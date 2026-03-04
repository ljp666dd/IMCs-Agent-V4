import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.agents.core.ml_agent import MLAgent
from src.core.logger import get_logger

logger = get_logger("TestHORModel")

def main():
    agent = MLAgent()
    
    logger.info("Starting ML Agent activity training pipeline...")
    # This should internally trigger train_physics_informed_dnn
    results = agent.train_activity_models("exchange_current_density")
    
    if not results:
        logger.error("Training failed to produce results.")
        return
        
    logger.info(f"Total models trained: {len(results)}")
    
    for res in results:
        logger.info(f"Model: {res.name}, R2_Test: {res.r2_test:.4f}, MAE_Test: {res.mae_test:.4f}")
        
    best = agent.get_top_models(k=1)[0]
    logger.info(f"Top model is: {best.name}")

    dnn_model = next((r for r in results if r.name == "HOR_Physics_DNN"), None)
    if dnn_model is None:
        logger.error("HOR_Physics_DNN not found in results!")
        return

    logger.info("Explaining HOR_Physics_DNN with SHAP...")
    try:
        explanation = agent.interpret_model(dnn_model)
        if explanation and explanation.get("shap_values") is not None:
            logger.info("SHAP explanation generated successfully.")
            
            # extract importances
            shap_vals = explanation["shap_values"]
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            if hasattr(shap_vals, 'values'):
                shap_vals = shap_vals.values
                
            import numpy as np
            importances = np.abs(shap_vals).mean(0)
            features = agent.data_manager.feature_names
            
            if len(importances.shape) > 1:
                importances = importances.mean(0)
                
            feat_imp = sorted(list(zip(features, importances)), key=lambda x: x[1], reverse=True)
            logger.info("Top SHAP Features for HOR_Physics_DNN:")
            for f, imp in feat_imp[:10]:
                logger.info(f"  {f}: {imp:.4f}")
        else:
            logger.info("SHAP run finished, but returned None or empty values string.")
    except Exception as e:
        logger.error(f"SHAP explanation failed with error: {e}")

if __name__ == "__main__":
    main()
