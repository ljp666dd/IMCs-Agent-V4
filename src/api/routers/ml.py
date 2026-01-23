from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from services.ml.types import ModelType

# Service Instance
from agents.core.ml_agent import MLAgent
ml_service = MLAgent()

router = APIRouter()

class TrainRequest(BaseModel):
    model_types: List[str] = ["traditional"] # traditional, deep_learning, gnn
    epochs: int = 100

class PredictionRequest(BaseModel):
    features: List[float]
    model_name: Optional[str] = None

@router.post("/train")
async def train_models(req: TrainRequest, background_tasks: BackgroundTasks):
    """Start training task."""
    # This should be async in background
    def run_training():
        if "traditional" in req.model_types:
            ml_service.train_traditional_models()
        if "deep_learning" in req.model_types:
            ml_service.train_deep_learning_models(epochs=req.epochs)
    
    background_tasks.add_task(run_training)
    return {"message": "Training started", "config": req.model_types}

@router.get("/models")
async def list_models():
    """List trained models."""
    return [
        {
            "name": m.name,
            "type": m.model_type.value,
            "r2_test": m.r2_test
        }
        for m in ml_service.results
    ]

@router.post("/predict")
async def predict(req: PredictionRequest):
    """Predict using best model."""
    if not ml_service.best_model:
        raise HTTPException(status_code=400, detail="No model trained yet")
    
    # Mock prediction logic (MLAgent needs predict method exposed ideally)
    # ml_service.predict(...)
    return {"prediction": 0.0, "note": "Prediction not fully exposed in Agent Facade yet"}
