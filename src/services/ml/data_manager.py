import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.core.logger import get_logger, log_exception
from src.services.chemistry.descriptors import StructureFeaturizer

logger = get_logger(__name__)

class DataManager:
    """
    Manages data loading, preprocessing, and splitting for ML models.
    Service layer implementation.
    """
    
    def __init__(self, output_dir: str = "data/ml_agent"):
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        
        # Data state
        self.feature_names: List[str] = []
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        
        # Split data
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        
        # Metadata
        self.material_ids: Optional[np.ndarray] = None

    @log_exception(logger)
    def load_theory_data(self, data_path: str, target_col: str = "formation_energy") -> Tuple[np.ndarray, np.ndarray]:
        """Load theoretical data from JSON (using known features whitelist)."""
        logger.info(f"Loading theory data from {data_path}, target={target_col}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame.from_dict(data, orient='index')
        
        # Validate target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
        
        # Clean target
        df = df.dropna(subset=[target_col])
        y = df[target_col].values
        
        # Metadata
        if 'material_id' in df.columns:
            self.material_ids = df['material_id'].values
            
        # Feature Selection
        # Prefer features defined in StructureFeaturizer
        known_features = StructureFeaturizer.KNOWN_FEATURES
        feature_cols = [c for c in df.columns if c in known_features]
        
        if len(feature_cols) == 0:
            logger.warning("No known features found in dataset. Falling back to heuristic selection.")
            exclude_cols = ['material_id', 'formula', target_col, "d-band center", "d-band width", "DOS at Fermi level"]
            feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
            
        if not feature_cols:
             raise ValueError("No numeric feature columns found")
             
        self.feature_names = feature_cols
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        self.X = X
        self.y = y
        logger.info(f"Loaded {len(X)} samples, {len(feature_cols)} features.")
        return X, y

    @log_exception(logger)
    def load_experimental_data(self, file_path: str, target_col: str, feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load experimental data from CSV/Excel."""
        logger.info(f"Loading experimental data from {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported format. Use CSV or Excel.")
            
        # Validate Target
        if target_col not in df.columns:
             raise ValueError(f"Target '{target_col}' not found.")
             
        # Feature Selection
        if feature_cols:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            X_df = df[feature_cols]
            self.feature_names = feature_cols
        else:
            # Auto-select numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            X_df = df[numeric_cols]
            self.feature_names = numeric_cols
            logger.info(f"Auto-selected {len(numeric_cols)} numeric features.")
            
        X = X_df.values
        y = df[target_col].values
        X = np.nan_to_num(X, nan=0.0)
        
        self.X = X
        self.y = y
        return X, y

    def prepare_split(self, test_size: float = 0.2, random_state: int = 42):
        """Split first, then fit scaler on train to avoid leakage."""
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_* first.")

        # Split on raw features
        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Fit scaler only on train
        self.X_train = self.scaler.fit_transform(X_train_raw)
        self.X_test = self.scaler.transform(X_test_raw)

        logger.info(f"Split & Scaled data: Train={len(self.X_train)}, Test={len(self.X_test)}")
