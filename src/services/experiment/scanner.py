import os
from typing import Dict, Any, List
from src.core.logger import get_logger, log_exception
from src.services.experiment.parsers import DataParser, DataType

logger = get_logger(__name__)

class DataScanner:
    """
    Scans directories for experimental data files.
    Service layer implementation.
    """
    
    def __init__(self):
        self.parser = DataParser()

    @log_exception(logger)
    def scan_directory(self, folder_path: str) -> Dict[str, Any]:
        """Scan a directory for experimental data files."""
        summary = {
            "total_files": 0,
            "valid_files": 0,
            "data_types": {},  # type -> list of files
            "errors": []
        }
        
        if not os.path.exists(folder_path):
            summary["errors"].append(f"Directory not found: {folder_path}")
            return summary
            
        logger.info(f"Scanning directory: {folder_path}")
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.csv', '.xlsx', '.xls', '.mpt', '.txt')):
                    summary["total_files"] += 1
                    full_path = os.path.join(root, file)
                    
                    try:
                        # Lightweight usage of parser
                        # Note: This loads the file, which might be slow for huge dirs.
                        # Optimization: Check extension or first few lines only?
                        # For V3.0 (Engineering), we rely on Robust Parser.
                        df = self.parser.load_file(full_path)
                        dtype = self.parser.detect_type(df)
                        
                        if dtype != DataType.UNKNOWN:
                            dtype_str = dtype.value
                            if dtype_str not in summary["data_types"]:
                                summary["data_types"][dtype_str] = []
                            summary["data_types"][dtype_str].append(file)
                            summary["valid_files"] += 1
                        
                    except Exception as e:
                        # Log but don't stop scanning
                        logger.debug(f"Skipping file {file}: {e}")
        
        logger.info(f"Scan complete. Found {summary['valid_files']} valid files.")
        return summary
