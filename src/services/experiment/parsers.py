import pandas as pd
import os
from enum import Enum
from typing import Optional
from src.core.logger import get_logger, log_exception

logger = get_logger(__name__)

class DataType(Enum):
    """Types of electrochemical data."""
    LSV = "lsv"
    CV = "cv"
    TAFEL = "tafel"
    EIS = "eis"
    CHRONOAMPEROMETRY = "chronoamperometry"
    UNKNOWN = "unknown"

class DataParser:
    """
    Parses experimental data files (CSV, Excel, TXT).
    Service layer implementation.
    """
    
    @log_exception(logger)
    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load data with automatic format detection."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_lower = file_path.lower()
        if file_lower.endswith(('.csv', '.txt', '.mpt')):
            return self._load_csv_robust(file_path)
        elif file_lower.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def _load_csv_robust(self, file_path: str) -> pd.DataFrame:
        """Try different delimiters."""
        for sep in [',', '\t', ';', ' ']:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) > 1:
                    return df
            except:
                continue
        # Fallback to default
        return pd.read_csv(file_path)

    def detect_type(self, df: pd.DataFrame) -> DataType:
        """Automatically detect the type of electrochemical data."""
        columns = [str(c).lower() for c in df.columns]
        
        # Check for EIS (frequency, Z', Z'')
        if any('freq' in c or 'z' in c or 'impedance' in c for c in columns):
            return DataType.EIS
        
        # Check for CV (multiple cycles)
        if 'cycle' in columns or ('potential' in str(columns) and len(df) > 1000):
            return DataType.CV
        
        # Check for chronoamperometry (time vs current)
        if any('time' in c for c in columns) and any('current' in c for c in columns):
            return DataType.CHRONOAMPEROMETRY
        
        # Default to LSV
        if any('potential' in c or 'voltage' in c or 'v' == c for c in columns):
            return DataType.LSV
        
        return DataType.UNKNOWN
