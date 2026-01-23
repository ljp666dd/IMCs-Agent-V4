from typing import Dict, Any, Optional
from src.services.ml.models.traditional import get_traditional_models
from src.services.ml.models.dnn import get_dnn_model
from src.services.ml.models.gnn import get_gnn_model

class ModelRegistry:
    """
    Central registry for all ML models.
    Service layer implementation.
    """
    
    @staticmethod
    def get_traditional_models() -> Dict[str, Any]:
        """Get all configured traditional ML models."""
        return get_traditional_models()
    
    @staticmethod
    def get_dnn_model(name: str, input_dim: int):
        """Get a specific DNN architecture."""
        return get_dnn_model(name, input_dim)
        
    @staticmethod
    def get_gnn_model(name: str):
        """Get a specific GNN architecture."""
        return get_gnn_model(name)
