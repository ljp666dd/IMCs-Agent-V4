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
        
        # Ideally we link this training session to an Experiment ID.
        # For now, we focus on saving the Model result.
        
    @log_exception(logger)
    def load_from_db(self, target_col: str = "formation_energy"):
        """Load training data directly from Database."""
        logger.info(f"Loading training data from DB (target={target_col})...")
        rows = self.db.fetch_training_set(target_col)
        
        if not rows:
            logger.warning("No data found in DB for training.")
            return

        X_list = []
        y_list = []
        
        for row in rows:
            cif_path = row.get("cif_path")
            target = row.get(target_col)
            
            if cif_path and os.path.exists(cif_path) and target is not None:
                # Extract features
                feats = self.featurizer.extract(cif_path)
                if feats is not None:
                    X_list.append(feats)
                    y_list.append(target)
        
        if X_list:
            X = np.array(X_list)
            y = np.array(y_list)
            feature_names = self.featurizer.feature_names
            
            # Populate DataManager manually
            self.data_manager.X = X
            self.data_manager.y = y
            self.data_manager.feature_names = feature_names
            self.data_manager.prepare_split(self.config.test_size, self.config.random_state)
            logger.info(f"Loaded {len(X)} samples from DB.")
        else:
            logger.warning("Failed to extract features for any DB records.")

    # ========== Training ==========
    
    def _save_models_to_db(self, results: List[ModelResult]):
        """Helper to save model results to DB and Disk."""
        import joblib
        
        # Ensure model directory exists
        model_dir = os.path.join(self.config.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        for res in results:
            try:
                # Prepare metrics JSON
                metrics = {
                    "r2_test": res.r2_test,
                    "mae_test": res.mae_test,
                    "rmse_test": res.rmse_test
                }
                
                # Save to Disk (Real Persistence)
                safe_name = "".join(x for x in res.name if x.isalnum() or x in "_-")
                filepath = os.path.join(model_dir, f"{safe_name}.pkl")
                joblib.dump(res.model, filepath)
                
                # Save to DB
                self.db.save_model(
                    name=res.name,
                    model_type=res.model_type.value,
                    target="target_variable", 
                    metrics=metrics,
                    filepath=filepath
                )
                logger.debug(f"Saved model {res.name} to {filepath} and DB.")
            except Exception as e:
                logger.error(f"Failed to save model {res.name}: {e}")

    def update_best_model(self):
        """Update self.best_model based on R2."""
        if not self.results:
            return
        # Sort by R2 descending
        sorted_models = sorted(self.results, key=lambda x: x.r2_test, reverse=True)
        self.best_model = sorted_models[0]
        logger.info(f"Best Model Updated: {self.best_model.name} (R2={self.best_model.r2_test:.3f})")

    @log_exception(logger)
    def select_features(self, top_n: int = 15):
        """
        Perform automated feature selection using Random Forest importance.
        Updates internal data state to use only selected features.
        """
        if self.X_train is None:
            return
            
        logger.info(f"Performing feature selection (target top_n={top_n})...")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import SelectFromModel
        
        # 1. Fit Selector
        # Use simple RF for robustness
        selector = SelectFromModel(
            estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            max_features=top_n
        )
        selector.fit(self.X_train, self.y_train)
        
        # 2. Get Mask
        mask = selector.get_support()
        selected_indices = np.where(mask)[0]
        
        # 3. Update DataManager State
        # We must filter X_train, X_test and feature_names
        original_feats = self.data_manager.feature_names
        new_feats = [original_feats[i] for i in selected_indices]
        
        logger.info(f"Selected {len(new_feats)} features: {new_feats}")
        
        self.data_manager.X = self.data_manager.X[:, mask]
        self.data_manager.X_train = self.data_manager.X_train[:, mask]
        self.data_manager.X_test = self.data_manager.X_test[:, mask]
        self.data_manager.feature_names = new_feats
        
        return new_feats

    @log_exception(logger)
    def train_traditional_models(self, auto_select_features: bool = True) -> List[ModelResult]:
        """
        Train all traditional ML models (RF, XGB, etc.).
        
        Args:
            auto_select_features (bool): If True, run feature selection first.
            
        Returns:
            List[ModelResult]: Training results.
        """
        if auto_select_features:
            self.select_features(top_n=12) # Reasonable default for small datasets
            
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
    
    def get_top_models(self, k: int = 3, metric: str = "r2_test") -> List[ModelResult]:
        """
        Get top K performing models.
        
        Args:
            k (int): Number of models to return.
            metric (str): Metric to sort by (descending for r2, ascending for rmse).
            
        Returns:
            List[ModelResult]: Top k models.
        """
        if not self.results:
            return []
            
        # Determine sort order
        reverse = True
        if "rmse" in metric or "mae" in metric:
            reverse = False
            
        sorted_models = sorted(
            self.results,
            key=lambda x: getattr(x, metric, -float('inf')) if getattr(x, metric) is not None else -float('inf'),
            reverse=reverse
        )
        return sorted_models[:k]

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
