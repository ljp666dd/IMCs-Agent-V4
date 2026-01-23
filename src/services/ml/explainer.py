import numpy as np
import pandas as pd
import warnings
from typing import Any, Optional, Dict
from src.core.logger import get_logger, log_exception

logger = get_logger(__name__)

# Optional import
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

class ModelExplainer:
    """
    Service for model interpretability (SHAP).
    """
    
    @log_exception(logger)
    def explain_model(self, model: Any, X_train: np.ndarray, feature_names: list, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Generate SHAP values for a trained model.
        
        Args:
            model: Trained model object (sklearn or torch)
            X_train: Training data (background dataset)
            feature_names: List of feature names
            model_type: "traditional", "deep_learning"
            
        Returns:
            Dict containing shap_values, explanation object, or None
        """
        if not HAS_SHAP:
            logger.warning("SHAP not installed. Skipping explanation.")
            return None
            
        logger.info(f"Generating SHAP explanation for {model_type} model...")
        
        try:
            # Select background sample (max 100 for speed)
            background = X_train
            if len(background) > 100:
                background = shap.kmeans(background, 100) if model_type != "deep_learning" else \
                             torch_sample(X_train, 100)
            
            explainer = None
            if model_type == "Traditional ML":
                # TreeExplainer is fastest for trees
                model_class = str(type(model)).lower()
                if 'ensemble' in model_class or 'boost' in model_class or 'tree' in model_class:
                     explainer = shap.TreeExplainer(model)
                else:
                     explainer = shap.KernelExplainer(model.predict, background)
            
            elif model_type == "Deep Learning":
                 # DeepExplainer (requires torch model)
                 import torch
                 if isinstance(background, np.ndarray):
                     bg_tensor = torch.FloatTensor(background)
                     # Need to move to same device as model
                     device = next(model.parameters()).device
                     bg_tensor = bg_tensor.to(device)
                     
                     embed_bg = bg_tensor # DeepExplainer handles it
                     explainer = shap.DeepExplainer(model, embed_bg)
            
            if explainer:
                # Calculate SHAP values
                # For KernelExplainer, use small sample of test data to explain?
                # Usually explain properties of specific instance or feature importance.
                # Here we return the explainer object or summary.
                
                # Simplified: Return explainer for now, or feature importance summary
                if hasattr(explainer, "shap_values"):
                    # Tree
                    shap_values = explainer.shap_values(X_train[:100]) # explain 100 samples
                else:
                    # Kernel / Deep - expensive
                    # Skip actual calculation for Service startup unless requested
                    shap_values = None
                
                return {
                    "explainer": explainer,
                    "shap_values": shap_values
                }
                
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return None

def torch_sample(X, n):
    import torch
    idx = np.random.choice(len(X), n, replace=False)
    return torch.FloatTensor(X[idx])
