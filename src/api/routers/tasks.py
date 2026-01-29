from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from src.services.task.planner import TaskPlanner
from src.services.task.executor import PlanExecutor
from src.services.task.types import TaskPlan, TaskType, TaskStep
from src.services.db.database import DatabaseService

# Initialize Services
# In a real app, use Dependency Injection (Depends)
planner = TaskPlanner()
# For Executor, we need actual agents. 
# This is a bit tricky in API without global state or DI.
# For v3.3 Prototype, we will instantiate fresh agents or shared singletons.
# We'll use a Singleton pattern in main.py or here.

# Ideally, we should import the initialized agents from a central container.
# For now, minimal instantiation:
from src.agents.core.task_manager import TaskManagerAgent
# TaskManagerAgent wraps everything. We can rely on it.
agent_instance = TaskManagerAgent() 
db = DatabaseService()

router = APIRouter()

class TaskRequest(BaseModel):
    query: str

class TaskResponse(BaseModel):
    task_id: str
    task_type: str
    description: str
    steps: List[Dict]
    status: str


def _restore_plan_from_db(task_id: str) -> Optional[TaskPlan]:
    plan_row = db.get_plan(task_id)
    if not plan_row:
        return None
    steps = db.list_plan_steps(task_id)
    latest = {}
    for s in steps:
        latest[s["step_id"]] = s

    try:
        task_type = TaskType(plan_row.get("task_type"))
    except Exception:
        task_type = TaskType.GENERAL

    plan = TaskPlan(
        task_id=task_id,
        task_type=task_type,
        description=plan_row.get("description") or "",
        steps=[],
        status=plan_row.get("status") or "pending",
    )

    def _step_rank(step_id: str) -> int:
        if isinstance(step_id, str) and step_id.startswith("step_"):
            try:
                return int(step_id.split("_", 1)[1])
            except Exception:
                return 10 ** 9
        return 10 ** 9

    for step_id in sorted(latest.keys(), key=_step_rank):
        row = latest[step_id]
        params = row.get("params") or {}
        deps = row.get("dependencies") or []
        status = row.get("status") or "pending"
        if status in ("completed", "replanned", "skipped"):
            step_status = "completed"
        elif status in ("failed", "blocked"):
            step_status = status
        else:
            step_status = "pending"
        step = TaskStep(
            step_id=step_id,
            agent=row.get("agent", ""),
            action=row.get("action", ""),
            params=params,
            dependencies=deps,
            status=step_status,
            result=row.get("result"),
            error=row.get("error"),
        )
        plan.steps.append(step)
        if step.result is not None:
            plan.results[step_id] = step.result
    return plan

@router.post("/create", response_model=TaskResponse)
async def create_task(req: TaskRequest):
    """Create a new task plan based on natural language query."""
    plan = agent_instance.create_plan(req.query)
    from dataclasses import asdict
    return {
        "task_id": plan.task_id,
        "task_type": plan.task_type.value,
        "description": plan.description,
        "steps": [asdict(step) for step in plan.steps],
        "status": plan.status
    }

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(req: ChatRequest):
    """Process a chat message (Conversation or Task Command)."""
    return agent_instance.process_chat_message(req.message)

@router.post("/execute/{task_id}")
async def execute_task(task_id: str, background_tasks: BackgroundTasks):
    """Execute a task (Async)."""
    # If not in memory, attempt to restore from DB for resume/confirmation.
    if not agent_instance.current_plan or agent_instance.current_plan.task_id != task_id:
        restored = _restore_plan_from_db(task_id)
        if not restored:
            raise HTTPException(status_code=404, detail="Task not found in memory or DB")
        agent_instance.current_plan = restored

    plan = agent_instance.current_plan
    if plan.status in ("completed", "failed", "blocked"):
        return {"message": f"Task already {plan.status}", "task_id": task_id}
    
    # Run in background
    background_tasks.add_task(agent_instance.execute_plan, agent_instance.current_plan)
    return {"message": "Task execution started", "task_id": task_id}


class GapConfirmRequest(BaseModel):
    run_step_ids: List[str] = []
    mark_complete: bool = False
    params_overrides: Optional[Dict[str, Any]] = None


@router.post("/{task_id}/confirm_gap")
async def confirm_gap_fill(task_id: str, req: GapConfirmRequest):
    """Confirm or skip evidence gap steps before continuing."""
    plan = db.get_plan(task_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Task not found")

    steps = db.list_plan_steps(task_id)
    latest = {}
    for s in steps:
        latest[s["step_id"]] = s

    pending_ids = [sid for sid, row in latest.items() if (row.get("status") in ("pending", "running"))]
    run_ids = set(req.run_step_ids or pending_ids)
    skip_ids = [sid for sid in pending_ids if sid not in run_ids]
    params_overrides = req.params_overrides or {}

    for sid in skip_ids:
        row = latest.get(sid) or {}
        db.log_plan_step(
            plan_id=task_id,
            step_id=sid,
            agent=row.get("agent", ""),
            action=row.get("action", ""),
            status="skipped",
            dependencies=row.get("dependencies") or [],
            params=row.get("params") or {},
            result={"note": "Skipped by user confirmation"},
        )

    for sid in run_ids:
        row = latest.get(sid) or {}
        params = params_overrides.get(sid, row.get("params") or {})
        db.log_plan_step(
            plan_id=task_id,
            step_id=sid,
            agent=row.get("agent", ""),
            action=row.get("action", ""),
            status="pending",
            dependencies=row.get("dependencies") or [],
            params=params,
            result={"note": "Confirmed by user"},
        )

    if req.mark_complete:
        db.update_plan_status(task_id, "completed")
        return {"message": "Gap fill skipped. Task marked completed.", "skipped": skip_ids}

    return {"message": "Gap fill confirmed.", "skipped": skip_ids, "run": list(run_ids)}

@router.get("/{task_id}")
async def get_task_status(task_id: str):
    """Get task status."""
    plan = db.get_plan(task_id)
    if plan:
        steps = db.list_plan_steps(task_id)
        # Reduce to latest status per step_id
        latest = {}
        for s in steps:
            latest[s["step_id"]] = s
        results = {
            step_id: (data.get("result") if data.get("result") is not None else None)
            for step_id, data in latest.items()
        }
        return {
            "task_id": task_id,
            "status": plan.get("status"),
            "steps": list(latest.values()),
            "results": results
        }

    if agent_instance.current_plan and agent_instance.current_plan.task_id == task_id:
        return {
            "task_id": task_id,
            "status": agent_instance.current_plan.status,
            "results": agent_instance.current_plan.results
        }
    return {"status": "unknown"}


@router.get("/{task_id}/report")
async def get_task_report(task_id: str):
    """Get task report with snapshot if available."""
    plan = db.get_plan(task_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Task not found")

    steps = db.list_plan_steps(task_id)
    latest = {}
    for s in steps:
        latest[s["step_id"]] = s
    results = {
        step_id: (data.get("result") if data.get("result") is not None else None)
        for step_id, data in latest.items()
    }
    report = {
        "task": plan,
        "steps": list(latest.values()),
        "results": results
    }

    # Attach snapshot if exists
    snap = db.get_snapshot_by_plan(task_id)
    if snap:
        items = db.list_snapshot_items(snap["id"])
        report["snapshot"] = {
            "meta": snap,
            "items": items
        }
        # Evidence chain for snapshot materials
        material_ids = [i.get("item_id") for i in items if i.get("item_type") == "material"]
        evidence_chain = []
        for mid in material_ids:
            evidence_chain.append({
                "material_id": mid,
                "evidence": db.get_evidence_for_material(mid),
                "adsorption_energies": db.list_adsorption_energies(mid),
                "activity_metrics": db.list_activity_metrics(mid),
            })
        report["evidence_chain"] = evidence_chain
    else:
        report["evidence_chain"] = []
    return report


@router.get("/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: int):
    """Get snapshot and items by id."""
    snap = db.get_snapshot(snapshot_id)
    if not snap:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    items = db.list_snapshot_items(snapshot_id)
    return {"meta": snap, "items": items}
