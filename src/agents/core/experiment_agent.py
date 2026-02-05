"""
Experiment Data Agent (ExperimentDataAgent)
Refactored (v3.3) to use Service-Oriented Architecture and SQLite Database.
"""

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import warnings
import json

# Import Services
from src.core.logger import get_logger, log_exception
from src.services.experiment.parsers import DataParser, DataType
from src.services.experiment.scanner import DataScanner
from src.services.db.database import DatabaseService

logger = get_logger(__name__)

# Preserve Enum for compatibility (or alias it)
DataType = DataType

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


@dataclass
class ExperimentDataConfig:
    """
    Configuration for Experiment Data Agent.
    
    Attributes:
        output_dir (str): Directory for output files.
        reference_electrode (str): Reference electrode type (default: RHE).
        scan_rate_default (float): Default scan rate for LSV (mV/s).
    """
    output_dir: str = "data/experimental"
    reference_electrode: str = "RHE"
    scan_rate_default: float = 5.0
    cv_scan_rate: float = 50.0


@dataclass
class LSVResult:
    """
    Results from LSV (Linear Sweep Voltammetry) analysis.
    
    Attributes:
        sample_id (str): Unique identifier for the sample.
        overpotential_10mA (float): Overpotential at 10 mA/cm2 (mV).
        onset_potential (float): Potential where reaction starts (V).
    """
    sample_id: str
    overpotential_10mA: Optional[float] = None
    overpotential_1mA: Optional[float] = None
    current_density_max: Optional[float] = None
    onset_potential: Optional[float] = None
    exchange_current_density: Optional[float] = None
    data: Optional[Dict[str, List[float]]] = None


@dataclass
class RDEResult:
    """Results from RDE analysis across multiple rotation speeds."""
    sample_id: str
    jk_by_potential: Dict[str, float] = field(default_factory=dict)
    exchange_current_density: Optional[float] = None
    tafel_slope: Optional[float] = None
    mass_activity: Optional[float] = None
    reference_potential: Optional[float] = None
    rpm_values: List[int] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None

@dataclass
class TafelResult:
    """Results from Tafel analysis.""" # ... (rest of TafelResult unchanged but not in view)




class ExperimentDataAgent:
    """
    Experiment Data Agent for processing electrochemical data.
    
    Architecture:
        - Facade: Delegates to services.
        - Services: DataParser (IO), DataScanner (Discovery), DatabaseService (Storage).
    """
    
    def __init__(self, config: ExperimentDataConfig = None):
        """
        Initialize ExperimentDataAgent.
        
        Args:
            config (ExperimentDataConfig): Configuration object.
        """
        self.config = config or ExperimentDataConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.results: Dict[str, Any] = {}
        
        # Initialize Services
        self.parser = DataParser()
        self.scanner = DataScanner()
        self.db = DatabaseService() # v3.3: Database Integration
        
        logger.info("ExperimentDataAgent initialized with services and database.")
    
    @log_exception(logger)
    def scan_directory(self, folder_path: str) -> Dict[str, Any]:
        """
        Scan directory using DataScanner service.
        
        Args:
            folder_path (str): Path to directory to scan.
            
        Returns:
            Dict: Summary of found files by type.
        """
        return self.scanner.scan_directory(folder_path)

    @log_exception(logger)
    def process_rde_directory(
        self,
        data_dir: str = "data/experimental/rde_lsv",
        reference_potential: float = 0.2,
        loading_mg_cm2: float = 0.25,
        precious_fraction: float = 0.20,
    ) -> Dict[str, Any]:
        """Process RDE LSV series in a directory and store metrics."""
        if not os.path.exists(data_dir):
            logger.warning(f"RDE directory not found: {data_dir}")
            return {"processed": 0, "directory": data_dir}

        manifest_path = os.path.join(data_dir, "manifest.json")
        conditions = None
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                conditions = manifest.get("conditions")
            except Exception:
                conditions = None

        files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".csv")
        ]
        if not files:
            return {"processed": 0, "directory": data_dir}

        # Group by formula prefix
        groups: Dict[str, List[str]] = {}
        for path in files:
            name = os.path.basename(path)
            formula = name.split("_")[0].split("-")[0]
            if not formula:
                continue
            groups.setdefault(formula, []).append(path)

        processed = 0
        summaries = []
        for formula, paths in groups.items():
            if len(paths) < 2:
                continue
            result = self.analyze_rde_series(
                paths,
                sample_id=formula,
                reference_potential=reference_potential,
                loading_mg_cm2=loading_mg_cm2,
                precious_fraction=precious_fraction,
                conditions=conditions,
            )
            processed += 1
            summaries.append({
                "sample_id": formula,
                "j0": result.exchange_current_density,
                "ma": result.mass_activity,
                "tafel": result.tafel_slope,
            })

        return {
            "processed": processed,
            "directory": data_dir,
            "summaries": summaries,
        }

    # ========== Data Loading ==========
    
    @log_exception(logger)
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV/Excel/MPT file using DataParser.
        
        Args:
            file_path (str): Absolute path to data file.
            
        Returns:
            pd.DataFrame: Loaded data.
        """
        return self.parser.load_file(file_path)
    
    def detect_data_type(self, df: pd.DataFrame) -> DataType:
        """
        Detect experiment type from dataframe columns.
        
        Args:
            df (pd.DataFrame): Dataframe to inspect.
            
        Returns:
            DataType: Detected type (LSV, CV, EIS, etc).
        """
        return self.parser.detect_type(df)
        
    # ========== Analysis ==========
    
    def analyze_lsv(self, df: pd.DataFrame, sample_id: str = "sample") -> LSVResult:
        """
        Analyze LSV curve to extract overpotential and onset.
        
        Args:
            df (pd.DataFrame): LSV data (Potential vs Current).
            sample_id (str): Sample identifier.
            
        Returns:
            LSVResult: Dataclass containing extracted metrics.
        """
        columns = {c.lower(): c for c in df.columns}
        
        # Find potential and current columns robustly (include common shorthand V/I)
        pot_keys = ["potential", "voltage", "ewe", "e", "v"]
        curr_keys = ["current", "density", "current_density", "currentdensity", "j", "i"]
        pot_col = next((orig for key, orig in columns.items() if any(k in key for k in pot_keys)), None)
        curr_col = next((orig for key, orig in columns.items() if any(k in key for k in curr_keys)), None)
        
        if not pot_col or not curr_col:
            logger.warning(f"Could not identify columns for LSV: {df.columns}")
            return LSVResult(sample_id=sample_id)
        
        V = df[pot_col].values
        J = df[curr_col].values
        
        try:
            # Overpotential at 10 mA/cm2
            idx_10 = np.argmin(np.abs(J - 10.0))
            eta_10 = float(V[idx_10]) if abs(J[idx_10] - 10) <= 5 else None
            
            # Onset Potential (0.1 mA/cm2 threshold)
            idx_onset = np.argmin(np.abs(J - 0.1))
            onset = float(V[idx_onset])
            
            result = LSVResult(
                sample_id=sample_id,
                overpotential_10mA=eta_10,
                current_density_max=float(np.max(J)),
                onset_potential=onset,
                data={
                    "voltage": V.tolist(),
                    "current": J.tolist()
                }
            )
            return result
            
        except Exception as e:
            logger.error(f"LSV Analysis failed: {e}")
            return LSVResult(sample_id=sample_id)

    def _extract_potential_current(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        columns = {c.lower(): c for c in df.columns}
        pot_keys = ["potential", "voltage", "ewe"]
        curr_keys = ["current", "current_density", "currentdensity", "density"]
        pot_col = next((orig for key, orig in columns.items() if any(k in key for k in pot_keys)), None)
        if pot_col is None:
            pot_col = next((orig for key, orig in columns.items() if key in ["e", "v"]), None)
        curr_col = next((orig for key, orig in columns.items() if any(k in key for k in curr_keys)), None)
        if curr_col is None:
            curr_col = next((orig for key, orig in columns.items() if key in ["j", "i"]), None)
        if curr_col == pot_col:
            curr_col = next(
                (orig for key, orig in columns.items() if key not in [pot_col.lower()] and "current" in key),
                None,
            )
        if not pot_col or not curr_col:
            return None, None
        V = df[pot_col].values.astype(float)
        J = df[curr_col].values.astype(float)
        return V, J

    def _parse_rpm_from_name(self, filename: str) -> Optional[int]:
        import re
        match = re.search(r"(\d+)\s*rpm", filename.lower())
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
        return None

    def analyze_rde_series(
        self,
        file_paths: List[str],
        sample_id: str,
        reference_potential: float = 0.2,
        loading_mg_cm2: float = 0.25,
        precious_fraction: float = 0.20,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> RDEResult:
        """Analyze multiple LSV curves (different RPM) to estimate Jk, J0, and MA."""
        if not file_paths:
            return RDEResult(sample_id=sample_id)

        rpm_to_curve = {}
        for path in file_paths:
            df = self.load_csv(path)
            V, J = self._extract_potential_current(df)
            if V is None or J is None:
                continue
            rpm = self._parse_rpm_from_name(os.path.basename(path))
            if rpm is None:
                continue
            rpm_to_curve[rpm] = (V, J)

        rpm_values = sorted(rpm_to_curve.keys(), reverse=True)
        if len(rpm_values) < 2:
            return RDEResult(sample_id=sample_id, rpm_values=rpm_values)

        # Build a common potential grid
        all_v = np.concatenate([rpm_to_curve[r][0] for r in rpm_values])
        v_min, v_max = float(np.min(all_v)), float(np.max(all_v))
        grid = np.linspace(v_min, v_max, 6)

        jk_by_potential = {}
        details = {"fits": {}}

        for v in grid:
            js = []
            xs = []
            for rpm in rpm_values:
                V, J = rpm_to_curve[rpm]
                j_interp = np.interp(v, V, J)
                if j_interp == 0:
                    continue
                omega = rpm * 2 * np.pi / 60.0
                xs.append(1.0 / np.sqrt(omega))
                js.append(1.0 / j_interp)
            if len(xs) < 2:
                continue
            coeff = np.polyfit(xs, js, 1)
            slope, intercept = coeff[0], coeff[1]
            if intercept <= 0:
                continue
            jk = 1.0 / intercept
            jk_by_potential[f"{v:.3f}"] = float(jk)
            details["fits"][f"{v:.3f}"] = {"slope": float(slope), "intercept": float(intercept)}

        # Estimate J0 from low overpotential region (first 3 potentials)
        j0 = None
        tafel_slope = None
        if jk_by_potential:
            items = sorted(jk_by_potential.items(), key=lambda kv: float(kv[0]))
            low = items[:3]
            xs = []
            ys = []
            for v_str, jk in low:
                if jk <= 0:
                    continue
                xs.append(float(v_str))
                ys.append(np.log10(jk))
            if len(xs) >= 2:
                coef = np.polyfit(xs, ys, 1)
                slope = coef[0]
                intercept = coef[1]
                tafel_slope = float(1.0 / slope) if slope != 0 else None
                j0 = float(10 ** intercept)

        # Mass activity at reference potential
        jk_ref = None
        if jk_by_potential:
            # use nearest potential
            nearest = min(jk_by_potential.keys(), key=lambda k: abs(float(k) - reference_potential))
            jk_ref = jk_by_potential.get(nearest)
        mass_activity = None
        if jk_ref is not None:
            denom = loading_mg_cm2 * precious_fraction
            if denom > 0:
                mass_activity = float(jk_ref / denom)

        result = RDEResult(
            sample_id=sample_id,
            jk_by_potential=jk_by_potential,
            exchange_current_density=j0,
            tafel_slope=tafel_slope,
            mass_activity=mass_activity,
            reference_potential=reference_potential,
            rpm_values=rpm_values,
            data=details,
        )

        # Save experiment + metrics
        formula_guess = sample_id.split('_')[0].split('-')[0]
        material_rec = self.db.get_material_by_formula(formula_guess)
        material_id = material_rec["material_id"] if material_rec else None
        exp_id = self.db.save_experiment(
            name=sample_id,
            exp_type="RDE",
            raw_path=";".join(file_paths),
            results=asdict(result),
            material_id=material_id,
        )

        metric_conditions = conditions or {
            "reference_electrode": self.config.reference_electrode,
            "scan_rate_mV_s": self.config.scan_rate_default,
        }

        if material_id:
            if jk_ref is not None:
                self.db.save_activity_metric(
                    material_id=material_id,
                    metric_name="Jk_ref",
                    metric_value=jk_ref,
                    unit="mA/cm2",
                    conditions=metric_conditions,
                    source="experiment",
                    source_id=str(exp_id),
                    metadata={"reference_potential": reference_potential},
                )
            if j0 is not None:
                self.db.save_activity_metric(
                    material_id=material_id,
                    metric_name="exchange_current_density",
                    metric_value=j0,
                    unit="mA/cm2",
                    conditions=metric_conditions,
                    source="experiment",
                    source_id=str(exp_id),
                    metadata={"reference_potential": reference_potential},
                )
            if mass_activity is not None:
                self.db.save_activity_metric(
                    material_id=material_id,
                    metric_name="mass_activity",
                    metric_value=mass_activity,
                    unit="mA/mg",
                    conditions=metric_conditions,
                    source="experiment",
                    source_id=str(exp_id),
                    metadata={
                        "loading_mg_cm2": loading_mg_cm2,
                        "precious_fraction": precious_fraction,
                        "reference_potential": reference_potential,
                    },
                )
            self.db.save_evidence(
                material_id=material_id,
                source_type="experiment",
                source_id=str(exp_id),
                score=1.0,
                metadata={"exp_type": "RDE", "files": file_paths},
            )

        return result


    @log_exception(logger)
    def process_request(self, file_path: str, method: str) -> Any:
        """
        Process a user request on a file and save results to DB.
        
        Args:
            file_path (str): Path to experiment file.
            method (str): Analysis method (e.g., 'lsv', 'cv').
            
        Returns:
            Any: Analysis result object.
        """
        df = self.load_csv(file_path)
        result = None
        
        if method.lower() == "lsv":
            result = self.analyze_lsv(df, os.path.basename(file_path))
            
            # v3.3: Save to Database
            if result:
                 # Auto-Link: Try to find matching material by formula (filename)
                 # Heuristic: Remove extension, split by underscore/hyphen if complex
                 # 经验性关联: 用文件名猜测材料公式(弱关联, 后续可引入更严格映射)
                 formula_guess = os.path.splitext(os.path.basename(file_path))[0].split('_')[0].split('-')[0]
                 material_rec = self.db.get_material_by_formula(formula_guess)
                 material_id = material_rec["material_id"] if material_rec else None
                 
                 exp_id = self.db.save_experiment(
                     name=result.sample_id,
                     exp_type="LSV",
                     raw_path=file_path,
                     results=asdict(result),
                     material_id=material_id
                 )

                 # Activity metrics (HOR/HER indicators)
                 conditions = {
                     "reference_electrode": self.config.reference_electrode,
                     "scan_rate_mV_s": self.config.scan_rate_default,
                 }
                 metrics = [
                     ("overpotential_10mA", result.overpotential_10mA, "mV"),
                     ("overpotential_1mA", result.overpotential_1mA, "mV"),
                     ("onset_potential", result.onset_potential, "V"),
                     ("current_density_max", result.current_density_max, "mA/cm2"),
                     ("exchange_current_density", result.exchange_current_density, "mA/cm2"),
                 ]
                 for metric_name, metric_value, unit in metrics:
                     if metric_value is None:
                         continue
                     self.db.save_activity_metric(
                         material_id=material_id,
                         metric_name=metric_name,
                         metric_value=metric_value,
                         unit=unit,
                         conditions=conditions,
                         source="experiment",
                         source_id=str(exp_id),
                         metadata={"exp_type": "LSV", "sample_id": result.sample_id}
                     )
                 
                 # M3: Log Evidence Chain
                 if material_id:
                     self.db.save_evidence(
                         material_id=material_id,
                         source_type="experiment",
                         source_id=str(exp_id),
                         score=1.0,
                         metadata={"filename": file_path, "overpotential": result.overpotential_10mA}
                     )
                     logger.info(f"Linked Experiment {exp_id} to Material {material_id} (Evidence Chain created).")
                 else:
                     logger.info(f"Saved LSV {exp_id}, but no matching material found for formula '{formula_guess}'.")
                 
        return result
