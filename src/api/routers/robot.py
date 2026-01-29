from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from src.services.db.database import DatabaseService
from src.agents.core.ml_agent import MLAgent, MLAgentConfig

router = APIRouter()
db = DatabaseService()

class RobotTaskRequest(BaseModel):
    task_type: str
    payload: Optional[Dict[str, Any]] = None
    external_id: Optional[str] = None

class RobotTaskStatus(BaseModel):
    task_id: int
    status: str
    result: Optional[Dict[str, Any]] = None
    external_id: Optional[str] = None

class RobotResultCallback(BaseModel):
    task_id: Optional[int] = None
    external_id: Optional[str] = None
    status: str
    result: Optional[Dict[str, Any]] = None
    auto_iterate: bool = False
    metric_name: Optional[str] = None


def _infer_material_id(result: Dict[str, Any]) -> Optional[str]:
    if not result:
        return None
    if result.get("material_id"):
        return result.get("material_id")
    material = result.get("material") or {}
    formula = None
    if isinstance(material, dict):
        mid = material.get("material_id")
        if mid:
            return mid
        formula = material.get("formula")
    if not formula:
        formula = result.get("formula")
    if not formula:
        return None
    rec = db.get_material_by_formula(formula)
    return rec["material_id"] if rec else None


def _save_metrics_to_db(result: Dict[str, Any]) -> int:
    if not result:
        return 0
    metrics = result.get("metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        return 0
    material_id = _infer_material_id(result)
    saved = 0
    for name, value in metrics.items():
        try:
            metric_value = float(value)
        except Exception:
            continue
        db.save_activity_metric(
            material_id=material_id,
            metric_name=str(name),
            metric_value=metric_value,
            unit=None,
            conditions=result.get("conditions") or {},
            source="robot",
            source_id=str(result.get("source_id") or ""),
            metadata={"origin": "robot_callback"},
        )
        saved += 1
    return saved


def _run_activity_iteration(metric_name: str) -> None:
    ml_agent = MLAgent(MLAgentConfig(output_dir="data/ml_agent"))
    ml_agent.load_activity_metrics_from_db(metric_name)
    if ml_agent.X_train is None:
        return
    ml_agent.train_traditional_models()


@router.post("/submit_task")
async def submit_task(req: RobotTaskRequest):
    task_id = db.create_robot_task(req.task_type, req.payload, req.external_id)
    return {"task_id": task_id, "status": "queued"}

@router.get("/task_status/{task_id}")
async def task_status(task_id: int):
    task = db.get_robot_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Robot task not found")
    return task

@router.post("/result_callback")
async def result_callback(req: RobotResultCallback, background_tasks: BackgroundTasks):
    task = None
    if req.task_id is not None:
        task = db.get_robot_task(req.task_id)
    elif req.external_id:
        task = db.get_robot_task_by_external(req.external_id)
    if not task:
        raise HTTPException(status_code=404, detail="Robot task not found")

    db.update_robot_task(task["id"], status=req.status, result=req.result, external_id=req.external_id)

    saved_metrics = 0
    try:
        saved_metrics = _save_metrics_to_db(req.result or {})
    except Exception:
        saved_metrics = 0

    if req.auto_iterate:
        metric_name = req.metric_name or "exchange_current_density"
        background_tasks.add_task(_run_activity_iteration, metric_name)

    return {"task_id": task["id"], "status": req.status, "saved_metrics": saved_metrics}

@router.get("/tasks")
async def list_tasks(limit: int = 50):
    return {"tasks": db.list_robot_tasks(limit=limit)}
