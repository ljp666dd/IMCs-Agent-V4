"""
Machine Learning Agent (MLAgent)
Refactored (v3.0) to use Service-Oriented Architecture.
Delegates logic to src.services.ml modules.
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Import Services
from src.core.logger import get_logger, log_exception
from src.services.ml.data_manager import DataManager
from src.services.ml.registry import ModelRegistry
from src.services.ml.trainer import UnifiedTrainer
from src.services.ml.types import ModelResult, ModelType
from src.services.chemistry.descriptors import StructureFeaturizer

logger = get_logger(__name__)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


@dataclass
class MLAgentConfig:
    """Configuration for ML Agent."""
    output_dir: str = "data/ml_agent"
    test_size: float = 0.2
    random_state: int = 42


class MLAgent:
    """
    Unified Machine Learning Agent.
    v3.0: Acts as a facade/controller for ML Services.
    """
    
    # Delegate whitelist to the service
    KNOWN_FEATURES = StructureFeaturizer.KNOWN_FEATURES
    
    def __init__(self, config: MLAgentConfig = None):
        self.config = config or MLAgentConfig()
        
        # Services
        self.data_manager = DataManager(output_dir=self.config.output_dir)
        self.trainer = UnifiedTrainer()
        self.featurizer = StructureFeaturizer()
        
        # State
        self.results: List[ModelResult] = []
        self.best_model: Optional[ModelResult] = None
        
        logger.info("MLAgent initialized with v3.0 Services.")

    # ========== Properties (Proxy to DataManager) ==========
    # These ensure app.py continues to work without changes
    
    @property
    def feature_names(self) -> List[str]:
        return self.data_manager.feature_names
        
    @property
    def X(self) -> Optional[np.ndarray]:
        return self.data_manager.X
        
    @property
    def y(self) -> Optional[np.ndarray]:
        return self.data_manager.y
        
    @property
    def X_train(self) -> Optional[np.ndarray]:
        return self.data_manager.X_train
    
    @property
    def X_test(self) -> Optional[np.ndarray]:
        return self.data_manager.X_test
        
    @property
    def y_train(self) -> Optional[np.ndarray]:
        return self.data_manager.y_train
        
    @property
    def y_test(self) -> Optional[np.ndarray]:
        return self.data_manager.y_test
    
    # ========== Data Loading ==========
    
    @log_exception(logger)
    def load_data(self, data_path: str = None, target_col: str = "formation_energy"):
        """Load theoretical data (JSON)."""
        if data_path is None:
            data_path = os.path.join("data", "theory", "formation_energy_full.json")
            
        self.data_manager.load_theory_data(data_path, target_col)
        self.data_manager.prepare_split(self.config.test_size, self.config.random_state)
        
    @log_exception(logger)
    def load_generic_csv(self, file_path: str, target_col: str, feature_cols: List[str] = None):
        """Load experimental data (CSV/Excel)."""
        self.data_manager.load_experimental_data(file_path, target_col, feature_cols)
        self.data_manager.prepare_split(self.config.test_size, self.config.random_state)
        
    # ========== Training ==========
    
    @log_exception(logger)
    def train_traditional_models(self) -> List[ModelResult]:
        """Train all traditional ML models."""
        models = ModelRegistry.get_traditional_models()
        results = self.trainer.train_traditional(
            models, 
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        self.results.extend(results)
        return results
        
    @log_exception(logger)
    def train_deep_learning_models(self, epochs: int = 300) -> List[ModelResult]:
        """Train DNN models."""
        # Define configs to try
        configs = [
            "DNN_256_128_64",
            "DNN_512_256_128", 
            "DNN_128_64_32"
        ]
        results = self.trainer.train_dnn(
            configs, self.X_train.shape[1],
            self.X_train, self.y_train, 
            self.X_test, self.y_test,
            epochs
        )
        self.results.extend(results)
        return results
        
    @log_exception(logger)
    def train_transformer_models(self, epochs: int = 200) -> List[ModelResult]:
        """Train Transformer models."""
        configs = [
            "Transformer_64_2",
            "Transformer_128_3"
        ]
        results = self.trainer.train_dnn(
            configs, self.X_train.shape[1],
            self.X_train, self.y_train, 
            self.X_test, self.y_test,
            epochs
        )
        self.results.extend(results)
        return results
        
    @log_exception(logger)
    def train_gnn_models_v2(self, cif_dir: str, target_map: Dict[str, float], 
                         epochs: int = 100, model_types: list = None) -> List[ModelResult]:
        """Train GNN models."""
        model_names = model_types or ["CGCNN", "SchNet", "MEGNet"]
        results = self.trainer.train_gnn(
            model_names, cif_dir, target_map, epochs
        )
        self.results.extend(results)
        return results
        
    # ========== Utilities ==========
    
    def get_traditional_models(self) -> Dict[str, Any]:
        """Expose models for inspection if needed."""
        return ModelRegistry.get_traditional_models()
        
    def extract_structure_features(self, cif_path: str) -> Optional[np.ndarray]:
        """Extract features from CIF."""
        return self.featurizer.extract(cif_path)
        
    def interpret_model(self, model_result: ModelResult) -> Optional[Dict[str, Any]]:
        """Explain model prediction using SHAP."""
        from src.services.ml.explainer import ModelExplainer
        explainer = ModelExplainer()
        
        # Determine type
        m_type = model_result.model_type.value # "Traditional ML" etc.
        
        return explainer.explain_model(
            model=model_result.model,
            X_train=self.X_train,
            feature_names=self.feature_names,
            model_type=m_type
        )
