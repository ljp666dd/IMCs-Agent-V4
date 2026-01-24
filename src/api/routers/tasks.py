from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from src.services.task.planner import TaskPlanner
from src.services.task.executor import PlanExecutor
from src.services.task.types import TaskPlan, TaskType
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
    # In a real DB-backed app, we load plan from DB.
    # Here we use the in-memory current_plan from agent_instance (Singleton-ish).
    if not agent_instance.current_plan or agent_instance.current_plan.task_id != task_id:
        raise HTTPException(status_code=404, detail="Task not found in memory (DB not connected in API yet)")
    
    # Run in background
    background_tasks.add_task(agent_instance.execute_plan, agent_instance.current_plan)
    return {"message": "Task execution started", "task_id": task_id}

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
