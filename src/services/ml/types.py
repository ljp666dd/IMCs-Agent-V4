from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

class ModelType(Enum):
    TRADITIONAL = "Traditional ML"
    DEEP_LEARNING = "Deep Learning"
    GNN = "Graph Neural Network"

@dataclass
class ModelResult:
    """Standardized result container for all models."""
    name: str
    model_type: ModelType
    r2_train: float
    r2_test: float
    mae_test: float
    rmse_test: float
    model: Any  # sklearn estimator or torch model
