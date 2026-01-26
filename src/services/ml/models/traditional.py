from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from src.core.logger import get_logger

logger = get_logger(__name__)

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

def get_traditional_models() -> Dict[str, Any]:
    """Get dictionary of configured traditional ML models."""
    
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=1
        ),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "BayesianRidge": BayesianRidge(),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5),
        "SVR": SVR(kernel='rbf', C=10, gamma='scale'),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=1, verbosity=0
        )
    else:
        logger.warning("XGBoost not installed. Skipping.")
    
    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=1, verbose=-1
        )
    else:
        logger.warning("LightGBM not installed. Skipping.")
    
    return models
