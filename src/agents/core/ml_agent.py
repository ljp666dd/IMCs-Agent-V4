"""
Machine Learning Agent (MLAgent)
Refactored (v3.3) to use Service-Oriented Architecture and SQLite Database.
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import warnings
import json

# Import Services
from src.core.logger import get_logger, log_exception
from src.services.ml.data_manager import DataManager
from src.services.ml.registry import ModelRegistry
from src.services.ml.trainer import UnifiedTrainer
from src.services.ml.types import ModelResult, ModelType
from src.services.chemistry.descriptors import StructureFeaturizer
from src.services.db.database import DatabaseService

logger = get_logger(__name__)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


@dataclass
class MLAgentConfig:
    """
    Configuration for ML Agent.
    
    Attributes:
        output_dir (str): Directory for model artifacts.
        test_size (float): Fraction of data for testing.
        random_state (int): Seed for reproducibility.
    """
    output_dir: str = "data/ml_agent"
    test_size: float = 0.2
    random_state: int = 42


class MLAgent:
    """
    Unified Machine Learning Agent.
    
    Architecture:
        - Facade: Controls ML workflow.
        - Services: DataManager, Trainer, Registry, Featurizer, DatabaseService.
    """
    
    KNOWN_FEATURES = StructureFeaturizer.KNOWN_FEATURES
    
    def __init__(self, config: MLAgentConfig = None):
        """
        Initialize ML Agent.
        
        Args:
            config (MLAgentConfig): Configuration object.
        """
        self.config = config or MLAgentConfig()
        
        # Services
        self.data_manager = DataManager(output_dir=self.config.output_dir)
        self.trainer = UnifiedTrainer()
        self.featurizer = StructureFeaturizer()
        self.db = DatabaseService() # v3.3: Database Integration
        
        # State
        self.results: List[ModelResult] = []
        self.best_model: Optional[ModelResult] = None
        
        logger.info("MLAgent initialized with services and database.")

    # ========== Properties (Proxy to DataManager) ==========
    
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
        """
        Load theoretical data (JSON).
        
        Args:
            data_path (str): Path to JSON file.
            target_col (str): Name of target column.
        """
        if data_path is None:
            data_path = os.path.join("data", "theory", "formation_energy_full.json")
            
        self.data_manager.load_theory_data(data_path, target_col)
        self.data_manager.prepare_split(self.config.test_size, self.config.random_state)
        
    @log_exception(logger)
    def load_generic_csv(self, file_path: str, target_col: str, feature_cols: List[str] = None):
        """
        Load experimental data (CSV/Excel).
        
        Args:
            file_path (str): Path to data file.
            target_col (str): Target column name.
            feature_cols (List[str]): List of feature columns.
        """
        self.data_manager.load_experimental_data(file_path, target_col, feature_cols)
        self.data_manager.prepare_split(self.config.test_size, self.config.random_state)
        
        # v3.3: Log experiment linkage to DB? 
        # Ideally we link this training session to an Experiment ID.
        # For now, we focus on saving the Model result.
        
    # ========== Training ==========
    
    def _save_models_to_db(self, results: List[ModelResult]):
        """Helper to save model results to DB."""
        for res in results:
            try:
                # Prepare metrics JSON
                metrics = {
                    "r2_test": res.r2_test,
                    "mae_test": res.mae_test,
                    "rmse_test": res.rmse_test
                }
                
                # Save to DB
                # Note: Model object itself is not pickled into DB, we save filepath if available or just metadata.
                # In v3.3 Trainer refactor, we should ensure models are saved to disk and filepath returned.
                # Assume Trainer handles disk save or we do it here.
                # Here we mock filepath as we haven't implemented disk serialization in Trainer yet.
                filepath = os.path.join(self.config.output_dir, f"{res.name}.pkl")
                
                self.db.save_model(
                    name=res.name,
                    model_type=res.model_type.value,
                    target="target_variable", # Needs context
                    metrics=metrics,
                    filepath=filepath
                )
                logger.debug(f"Saved model record {res.name} to DB.")
            except Exception as e:
                logger.error(f"Failed to save model {res.name} to DB: {e}")

    @log_exception(logger)
    def train_traditional_models(self) -> List[ModelResult]:
        """
        Train all traditional ML models (RF, XGB, etc.).
        
        Returns:
            List[ModelResult]: Training results.
        """
        models = ModelRegistry.get_traditional_models()
        results = self.trainer.train_traditional(
            models, 
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        self.results.extend(results)
        self._save_models_to_db(results) # v3.3 DB
        return results
        
    @log_exception(logger)
    def train_deep_learning_models(self, epochs: int = 300) -> List[ModelResult]:
        """
        Train DNN models.
        
        Args:
            epochs (int): Training epochs.
            
        Returns:
            List[ModelResult]: Results.
        """
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
        self._save_models_to_db(results) # v3.3 DB
        return results
        
    @log_exception(logger)
    def train_transformer_models(self, epochs: int = 200) -> List[ModelResult]:
        """
        Train Transformer models.
        
        Args:
            epochs (int): Training epochs.
            
        Returns:
            List[ModelResult]: Results.
        """
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
        self._save_models_to_db(results) # v3.3 DB
        return results
        
    @log_exception(logger)
    def train_gnn_models_v2(self, cif_dir: str, target_map: Dict[str, float], 
                         epochs: int = 100, model_types: list = None) -> List[ModelResult]:
        """
        Train GNN models.
        
        Args:
            cif_dir (str): Directory containing CIF files.
            target_map (Dict): Map of ID to target value.
            epochs (int): Epochs.
            model_types (list): List of model names.
            
        Returns:
            List[ModelResult]: Results.
        """
        model_names = model_types or ["CGCNN", "SchNet", "MEGNet"]
        results = self.trainer.train_gnn(
            model_names, cif_dir, target_map, epochs
        )
        self.results.extend(results)
        self._save_models_to_db(results) # v3.3 DB
        return results
        
    # ========== Utilities ==========
    
    def get_traditional_models(self) -> Dict[str, Any]:
        """Get dict of initialized model objects."""
        return ModelRegistry.get_traditional_models()
        
    def extract_structure_features(self, cif_path: str) -> Optional[np.ndarray]:
        """Extract features from CIF file."""
        return self.featurizer.extract(cif_path)
        
    def interpret_model(self, model_result: ModelResult) -> Optional[Dict[str, Any]]:
        """
        Explain model prediction using SHAP.
        
        Args:
            model_result (ModelResult): Model to explain.
            
        Returns:
            Dict: SHAP values and metadata.
        """
        from src.services.ml.explainer import ModelExplainer
        explainer = ModelExplainer()
        
        m_type = model_result.model_type.value
        
        return explainer.explain_model(
            model=model_result.model,
            X_train=self.X_train,
            feature_names=self.feature_names,
            model_type=m_type
        )
