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
        
        # Find potential and current columns robustly
        pot_col = next((c for c in columns if 'potential' in c or 'voltage' in c), None)
        curr_col = next((c for c in columns if 'current' in c or 'density' in c), None)
        
        if not pot_col or not curr_col:
            logger.warning(f"Could not identify columns for LSV: {df.columns}")
            return LSVResult(sample_id=sample_id)
            
        V = df[pot_col].values
        J = df[curr_col].values
        
        try:
            # Overpotential at 10 mA/cm2
            # Logic: Find voltage where |current - 10| is minimal
            idx_10 = np.argmin(np.abs(J - 10.0))
            eta_10 = float(V[idx_10]) if abs(J[idx_10] - 10) < 5 else None
            
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
                 self.db.save_experiment(
                     name=result.sample_id,
                     exp_type="LSV",
                     raw_path=file_path,
                     results=asdict(result)
                 )
                 logger.info(f"Saved LSV results for {result.sample_id} to DB.")
                 
        return result
