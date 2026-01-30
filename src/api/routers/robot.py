from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os

from src.services.db.database import DatabaseService
from src.agents.core.ml_agent import MLAgent, MLAgentConfig
from src.agents.core.experiment_agent import ExperimentDataAgent, ExperimentDataConfig

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
    record_predictions: bool = True
    top_n: Optional[int] = None
    auto_ingest: bool = False
    ingest_params: Optional[Dict[str, Any]] = None


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


def _extract_artifact_paths(result: Dict[str, Any]) -> List[str]:
    if not result:
        return []
    raw = result.get("artifacts") or result.get("files") or result.get("artifact_paths")
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    paths: List[str] = []
    for item in raw:
        if not item:
            continue
        if isinstance(item, dict):
            path = item.get("path") or item.get("file") or item.get("filepath")
        else:
            path = str(item)
        if path:
            paths.append(path)
    return paths


def _process_experiment_artifacts(result: Dict[str, Any], ingest_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    paths = _extract_artifact_paths(result)
    if not paths:
        return {"processed": 0, "message": "no artifacts"}

    params = ingest_params or {}
    data_dir = params.get("data_dir")
    reference_potential = params.get("reference_potential", 0.2)
    loading_mg_cm2 = params.get("loading_mg_cm2", 0.25)
    precious_fraction = params.get("precious_fraction", 0.20)
    method = params.get("method", "lsv")
    conditions = result.get("conditions") or params.get("conditions")

    exp_agent = ExperimentDataAgent(ExperimentDataConfig(output_dir="data/experimental"))

    if data_dir and os.path.isdir(data_dir):
        return exp_agent.process_rde_directory(
            data_dir=data_dir,
            reference_potential=reference_potential,
            loading_mg_cm2=loading_mg_cm2,
            precious_fraction=precious_fraction,
        )

    file_paths = [p for p in paths if os.path.exists(p)]
    if not file_paths:
        return {"processed": 0, "message": "artifact paths not found"}

    groups: Dict[str, List[str]] = {}
    for path in file_paths:
        if os.path.isdir(path):
            return exp_agent.process_rde_directory(
                data_dir=path,
                reference_potential=reference_potential,
                loading_mg_cm2=loading_mg_cm2,
                precious_fraction=precious_fraction,
            )
        name = os.path.basename(path)
        formula = name.split("_")[0].split("-")[0]
        if not formula:
            formula = "sample"
        groups.setdefault(formula, []).append(path)

    processed = 0
    summaries = []
    for formula, files in groups.items():
        if len(files) >= 2:
            result_obj = exp_agent.analyze_rde_series(
                files,
                sample_id=formula,
                reference_potential=reference_potential,
                loading_mg_cm2=loading_mg_cm2,
                precious_fraction=precious_fraction,
                conditions=conditions,
            )
            processed += 1
            summaries.append({
                "sample_id": formula,
                "j0": result_obj.exchange_current_density,
                "ma": result_obj.mass_activity,
                "tafel": result_obj.tafel_slope,
            })
        else:
            try:
                exp_agent.process_request(files[0], method)
                processed += 1
            except Exception:
                continue

    return {"processed": processed, "summaries": summaries}


def _run_activity_iteration(metric_name: str, record_predictions: bool = True, top_n: Optional[int] = None) -> None:
    ml_agent = MLAgent(MLAgentConfig(output_dir="data/ml_agent"))
    ml_agent.load_activity_metrics_from_db(metric_name)
    if ml_agent.X_train is None:
        return
    results = ml_agent.train_traditional_models()
    preds = ml_agent.predict_best()

    top_n = top_n or 10
    ranked = []
    if preds:
        try:
            ranked = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        except Exception:
            ranked = list(preds.items())[:top_n]

    if record_predictions and preds:
        for mid, score in preds.items():
            try:
                db.save_evidence(
                    material_id=str(mid),
                    source_type="ml_prediction",
                    source_id=f"robot_iter_{metric_name}",
                    score=0.6,
                    metadata={
                        "prediction": float(score),
                        "metric": metric_name,
                        "origin": "robot_iter",
                    },
                )
            except Exception:
                continue

    try:
        out_dir = os.path.join("data", "experimental")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"robot_iter_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report = {
            "metric": metric_name,
            "trained_models": [r.name for r in results] if results else [],
            "ranking_top_n": [{"material_id": mid, "score": score} for mid, score in ranked],
            "prediction_count": len(preds) if preds else 0,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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

    ingest_summary = None
    if req.auto_ingest:
        try:
            ingest_summary = _process_experiment_artifacts(req.result or {}, req.ingest_params)
        except Exception:
            ingest_summary = None

    if req.auto_iterate:
        metric_name = req.metric_name or "exchange_current_density"
        background_tasks.add_task(_run_activity_iteration, metric_name, req.record_predictions, req.top_n)

    response = {"task_id": task["id"], "status": req.status, "saved_metrics": saved_metrics}
    if ingest_summary is not None:
        response["ingest_summary"] = ingest_summary
    return response

@router.get("/tasks")
async def list_tasks(limit: int = 50):
    return {"tasks": db.list_robot_tasks(limit=limit)}
