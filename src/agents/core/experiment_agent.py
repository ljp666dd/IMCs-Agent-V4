"""
Experiment Data Agent (ExperimentDataAgent)
Refactored (v3.0) to use Service-Oriented Architecture.
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import warnings

# Import Services
from src.core.logger import get_logger, log_exception
from src.services.experiment.parsers import DataParser, DataType
from src.services.experiment.scanner import DataScanner

logger = get_logger(__name__)

# Preserve Enum for compatibility (or alias it)
# We alias the Service Enum to the Agent Enum to avoid breaking app.py if it imports from here
DataType = DataType

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


@dataclass
class ExperimentDataConfig:
    """Configuration for Experiment Data Agent."""
    output_dir: str = "data/experimental"
    reference_electrode: str = "RHE"  # Reference electrode type
    scan_rate_default: float = 5.0  # mV/s for LSV
    cv_scan_rate: float = 50.0  # mV/s for CV


@dataclass
class LSVResult:
    """Results from LSV analysis."""
    sample_id: str
    overpotential_10mA: Optional[float] = None  # mV @ 10 mA/cm2
    overpotential_1mA: Optional[float] = None   # mV @ 1 mA/cm2
    current_density_max: Optional[float] = None  # mA/cm2
    onset_potential: Optional[float] = None     # V vs RHE
    exchange_current_density: Optional[float] = None  # mA/cm2


@dataclass
class TafelResult:
    """Results from Tafel analysis."""
    sample_id: str
    tafel_slope: Optional[float] = None  # mV/dec
    exchange_current: Optional[float] = None  # mA/cm2
    r_squared: Optional[float] = None


@dataclass
class CVResult:
    """Results from CV analysis."""
    sample_id: str
    ecsa: Optional[float] = None  # cm2
    specific_capacity: Optional[float] = None  # mF/cm2
    rf: Optional[float] = None  # Roughness factor


@dataclass
class EISResult:
    """Results from EIS analysis."""
    sample_id: str
    rs: Optional[float] = None  # Solution resistance, Ohm
    rct: Optional[float] = None  # Charge transfer resistance, Ohm
    cdl: Optional[float] = None  # Double layer capacitance, F


@dataclass
class StabilityResult:
    """Results from stability test."""
    sample_id: str
    retention_percent: Optional[float] = None
    duration_hours: Optional[float] = None
    degradation_rate: Optional[float] = None  # %/hour


class ExperimentDataAgent:
    """
    Experiment Data Agent for processing electrochemical data.
    Delegates parsing and scanning to Services.
    """
    
    def __init__(self, config: ExperimentDataConfig = None):
        self.config = config or ExperimentDataConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.results: Dict[str, Any] = {}
        
        # Initialize Services
        self.parser = DataParser()
        self.scanner = DataScanner()
        logger.info("ExperimentDataAgent initialized with services.")
    
    @log_exception(logger)
    def scan_directory(self, folder_path: str) -> Dict[str, Any]:
        """Scan directory using DataScanner service."""
        return self.scanner.scan_directory(folder_path)

    # ========== Data Loading (Delegated) ==========
    
    @log_exception(logger)
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV using DataParser."""
        return self.parser.load_file(file_path) # DataParser handles CSV/Excel/MPT types
    
    def detect_data_type(self, df: pd.DataFrame) -> DataType:
        """Detect type using DataParser."""
        return self.parser.detect_type(df)
        
    # ========== Analysis (Keep internal for now) ==========
    
    def analyze_lsv(self, df: pd.DataFrame, sample_id: str = "sample") -> LSVResult:
        """
        Analyze LSV curve.
        TODO: Move to services.experiment.analysis in Phase 4.
        """
        # (Preserve existing logic for now to minimal change)
        columns = {c.lower(): c for c in df.columns}
        
        # Find potential and current columns
        pot_col = next((c for c in columns if 'potential' in c or 'voltage' in c), None)
        curr_col = next((c for c in columns if 'current' in c or 'density' in c), None)
        
        if not pot_col or not curr_col:
            logger.warning(f"Could not identify columns for LSV: {df.columns}")
            return LSVResult(sample_id=sample_id)
            
        V = df[pot_col].values
        J = df[curr_col].values
        
        # Logic: Find overpotential at 10 mA/cm2
        # Assuming RHE scale
        
        # Simple interpolation
        try:
            # Overpotential at 10 mA
            # Find V where J = 10 (or -10 depending on convention)
            # Assuming oxidation is positive
            idx_10 = np.argmin(np.abs(J - 10.0))
            eta_10 = V[idx_10] if abs(J[idx_10] - 10) < 5 else None # Sanity check
            
            # Onset: V where J crosses 0.1 mA
            idx_onset = np.argmin(np.abs(J - 0.1))
            onset = V[idx_onset]
            
            return LSVResult(
                sample_id=sample_id,
                overpotential_10mA=eta_10,
                current_density_max=np.max(J),
                onset_potential=onset
            )
        except Exception as e:
            logger.error(f"LSV Analysis failed: {e}")
            return LSVResult(sample_id=sample_id)

    def process_request(self, file_path: str, method: str) -> Any:
        """Process a user request on a file."""
        df = self.load_csv(file_path)
        if method.lower() == "lsv":
            return self.analyze_lsv(df, os.path.basename(file_path))
        return None
