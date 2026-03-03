from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os
import math

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
    callback_id: Optional[str] = None
    status: str
    result: Optional[Dict[str, Any]] = None
    auto_iterate: bool = False
    metric_name: Optional[str] = None
    record_predictions: bool = True
    top_n: Optional[int] = None
    auto_ingest: bool = False
    ingest_params: Optional[Dict[str, Any]] = None


class IterationToTaskPlanRequest(BaseModel):
    robot_task_id: int
    event_id: Optional[int] = None
    top_n: Optional[int] = None
    task_type: str = "catalyst_discovery"
    description: Optional[str] = None
    auto_execute: bool = True


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
        if not math.isfinite(metric_value):
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


def _run_activity_iteration(robot_task_id: int, metric_name: str, record_predictions: bool = True, top_n: Optional[int] = None) -> None:
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

    out_dir = os.path.join("data", "experimental")
    out_path = os.path.join(out_dir, f"robot_iter_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report = {
        "metric": metric_name,
        "trained_models": [r.name for r in results] if results else [],
        "ranking_top_n": [{"material_id": mid, "score": score} for mid, score in ranked],
        "prediction_count": len(preds) if preds else 0,
    }

    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Persist iteration report into robot event stream for reproducible closed-loop.
    try:
        if robot_task_id:
            event_payload = dict(report)
            event_payload["report_path"] = out_path
            event_payload["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db.log_robot_task_event_idempotent(
                task_id=int(robot_task_id),
                status="iteration_completed",
                payload=event_payload,
                external_id=None,
                callback_id=None,
                payload_hash=db.hash_payload(event_payload),
            )
    except Exception:
        pass


def _normalize_status(status: str) -> str:
    raw = (status or "").strip().lower()
    if raw in {"success", "succeeded", "ok", "done", "completed", "complete"}:
        return "completed"
    if raw in {"fail", "failed", "error", "failure"}:
        return "failed"
    if raw in {"running", "in_progress", "processing"}:
        return "running"
    if raw in {"queued", "pending"}:
        return "queued"
    if raw in {"canceled", "cancelled"}:
        return "canceled"
    return status or "unknown"


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

    normalized_status = _normalize_status(req.status)

    event_payload = {
        "status": normalized_status,
        "result": req.result,
        "auto_iterate": bool(req.auto_iterate),
        "metric_name": req.metric_name,
        "record_predictions": bool(req.record_predictions),
        "top_n": req.top_n,
        "auto_ingest": bool(req.auto_ingest),
        "ingest_params": req.ingest_params,
    }
    payload_hash = db.hash_payload(event_payload)
    logged = db.log_robot_task_event_idempotent(
        task_id=task["id"],
        status=normalized_status,
        payload=event_payload,
        external_id=req.external_id,
        callback_id=req.callback_id,
        payload_hash=payload_hash,
    )
    if logged.get("conflict"):
        raise HTTPException(status_code=409, detail="Duplicate callback_id with different payload")

    # Always update task state (idempotent); skip side-effects on duplicate callbacks.
    db.update_robot_task(task["id"], status=normalized_status, result=req.result, external_id=req.external_id)
    if not logged.get("inserted"):
        return {
            "task_id": task["id"],
            "status": normalized_status,
            "duplicate": True,
            "event_id": logged.get("event_id"),
            "saved_metrics": 0,
        }

    saved_metrics = 0
    if normalized_status == "completed":
        try:
            saved_metrics = _save_metrics_to_db(req.result or {})
        except Exception:
            saved_metrics = 0

    ingest_summary = None
    if req.auto_ingest and normalized_status == "completed":
        try:
            ingest_summary = _process_experiment_artifacts(req.result or {}, req.ingest_params)
        except Exception:
            ingest_summary = None

    if req.auto_iterate and normalized_status == "completed":
        metric_name = req.metric_name or "exchange_current_density"
        background_tasks.add_task(_run_activity_iteration, task["id"], metric_name, req.record_predictions, req.top_n)

    response = {
        "task_id": task["id"],
        "status": normalized_status,
        "saved_metrics": saved_metrics,
        "event_id": logged.get("event_id"),
        "duplicate": False,
    }
    if ingest_summary is not None:
        response["ingest_summary"] = ingest_summary
    return response

@router.get("/task_events/{task_id}")
async def task_events(task_id: int, limit: int = 50, status: Optional[str] = None):
    task = db.get_robot_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Robot task not found")
    events = db.list_robot_task_events(task_id=task_id, limit=limit, status=status)
    return {"task_id": task_id, "events": events}


@router.post("/iteration_to_taskplan")
async def iteration_to_taskplan(req: IterationToTaskPlanRequest, background_tasks: BackgroundTasks):
    task = db.get_robot_task(req.robot_task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Robot task not found")

    events = db.list_robot_task_events(task_id=req.robot_task_id, limit=200)
    if req.event_id is not None:
        event = next((e for e in events if e.get("id") == req.event_id), None)
    else:
        iter_events = [e for e in events if e.get("status") == "iteration_completed"]
        event = iter_events[0] if iter_events else None

    if not event:
        raise HTTPException(status_code=404, detail="Iteration event not found")

    payload = event.get("payload") or {}
    ranking = payload.get("ranking_top_n") or []
    if not isinstance(ranking, list) or not ranking:
        raise HTTPException(status_code=400, detail="Iteration event has no ranking_top_n")

    if req.top_n is not None:
        try:
            n = int(req.top_n)
        except Exception:
            n = None
        if n and n > 0:
            ranking = ranking[:n]

    preds: Dict[str, float] = {}
    for item in ranking:
        if not isinstance(item, dict):
            continue
        mid = (item.get("material_id") or "").strip()
        if not mid:
            continue
        try:
            score = float(item.get("score"))
        except Exception:
            continue
        if not math.isfinite(score):
            continue
        preds[mid] = score

    if not preds:
        raise HTTPException(status_code=400, detail="No valid (material_id, score) pairs in ranking_top_n")

    metric_name = payload.get("metric") or payload.get("metric_name") or "exchange_current_density"
    target_col = f"activity_metric:{metric_name}"

    try:
        from src.services.task.types import TaskPlan, TaskStep, TaskType
        task_type = TaskType(req.task_type)
    except Exception:
        from src.services.task.types import TaskPlan, TaskStep, TaskType
        task_type = TaskType.CATALYST_DISCOVERY

    plan_id = f"iter_{req.robot_task_id}_{event.get('id')}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    description = req.description or f"Iteration Top-N recommend from robot_task_id={req.robot_task_id} event_id={event.get('id')} metric={metric_name}"

    plan = TaskPlan(
        task_id=plan_id,
        task_type=task_type,
        description=description,
        steps=[
            TaskStep(
                step_id="step_1",
                agent="ml",
                action="seed_predictions",
                params={"predictions": preds, "target_col": target_col},
                dependencies=[],
            ),
            TaskStep(
                step_id="step_2",
                agent="task_manager",
                action="recommend",
                params={},
                dependencies=["step_1"],
            ),
            TaskStep(
                step_id="step_3",
                agent="task_manager",
                action="knowledge_pack",
                params={},
                dependencies=["step_1", "step_2"],
            ),
        ],
    )

    # Persist plan skeleton immediately to avoid 404 races when UI polls /tasks/{task_id}.
    db.create_plan(plan_id=plan_id, user_id=None, task_type=task_type.value, description=description)
    for step in plan.steps:
        db.log_plan_step(
            plan_id=plan_id,
            step_id=step.step_id,
            agent=step.agent,
            action=step.action,
            status="pending",
            dependencies=step.dependencies,
            params=step.params,
        )

    if req.auto_execute:
        db.update_plan_status(plan_id, "executing")

        def _do_execute():
            from src.services.task.executor_factory import new_plan_executor
            executor = new_plan_executor()
            executor.max_adaptive_rounds = 0
            executor.execute_plan(plan)

        background_tasks.add_task(_do_execute)

    return {
        "task_id": plan_id,
        "task_type": task_type.value,
        "from_robot_task_id": req.robot_task_id,
        "from_event_id": event.get("id"),
        "auto_execute": bool(req.auto_execute),
        "message": "TaskPlan created" + (" and execution started" if req.auto_execute else ""),
    }

@router.get("/tasks")
async def list_tasks(limit: int = 50):
    return {"tasks": db.list_robot_tasks(limit=limit)}
