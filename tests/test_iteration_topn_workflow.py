import asyncio
import json
import os
from datetime import datetime

from fastapi import BackgroundTasks

from src.services.db.database import DatabaseService
from src.services.task.executor import PlanExecutor
from src.services.task.types import TaskPlan, TaskStep, TaskType


def _cleanup_db_files(db_path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db_path + suffix)
        except Exception:
            pass


def _cleanup_file(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


class _DummyTheoryAgent:
    def __init__(self, db: DatabaseService):
        self._db = db

    def list_stored_materials(self, limit: int = 20):
        return self._db.list_materials(limit=limit)

    def get_status(self):
        return {"cif_files": 0}


def test_seed_predictions_generates_ordered_ranking_and_updates_knowledge_pack():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    db_path = os.path.join("data", f"test_seed_predictions_{stamp}.db")
    task_id = f"seed_preds_{stamp}"
    knowledge_path = os.path.join("data", "tasks", f"knowledge_{task_id}.json")

    db = DatabaseService(db_path=db_path)
    try:
        db.save_material("mp-1", "Pt", energy=-1.0)
        db.save_material("mp-2", "PtNi", energy=-0.5)
        db.save_material("mp-3", "PtCo", energy=-0.3)

        executor = PlanExecutor(
            agents={"theory": _DummyTheoryAgent(db), "ml": object()},
            db=db,
        )
        executor.meta_controller = None
        executor.max_adaptive_rounds = 0

        plan = TaskPlan(
            task_id=task_id,
            task_type=TaskType.CATALYST_DISCOVERY,
            description="seed predictions plan",
            steps=[
                TaskStep(
                    step_id="step_1",
                    agent="ml",
                    action="seed_predictions",
                    params={
                        "predictions": {"mp-1": 0.9, "mp-2": 0.8, "mp-3": 0.7},
                        "target_col": "activity_metric:exchange_current_density",
                    },
                    dependencies=[],
                ),
                TaskStep(
                    step_id="step_2",
                    agent="task_manager",
                    action="knowledge_pack",
                    params={},
                    dependencies=["step_1"],
                ),
            ],
        )

        executor.execute_plan(plan)
        assert plan.status == "completed"

        ranking = plan.results.get("ranking_current") or []
        assert [r.get("material_id") for r in ranking[:3]] == ["mp-1", "mp-2", "mp-3"]
        assert plan.results.get("candidate_material_ids") == ["mp-1", "mp-2", "mp-3"]

        assert os.path.exists(knowledge_path)
        with open(knowledge_path, "r", encoding="utf-8") as f:
            pack = json.load(f)
        assert [r.get("material_id") for r in (pack.get("ranking_current") or [])[:3]] == ["mp-1", "mp-2", "mp-3"]
    finally:
        _cleanup_file(knowledge_path)
        _cleanup_db_files(db_path)


def test_iteration_to_taskplan_persists_plan_skeleton_without_execution():
    import src.api.routers.robot as robot

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    db_path = os.path.join("data", f"test_iter_to_plan_{stamp}.db")
    test_db = DatabaseService(db_path=db_path)

    old_db = robot.db
    robot.db = test_db
    try:
        robot_task_id = test_db.create_robot_task("experiment", payload={"x": 1}, external_id="ext-iter")
        payload = {
            "metric": "exchange_current_density",
            "ranking_top_n": [
                {"material_id": "mp-1", "score": 1.0},
                {"material_id": "mp-2", "score": 0.9},
            ],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        test_db.log_robot_task_event_idempotent(
            task_id=robot_task_id,
            status="iteration_completed",
            payload=payload,
            external_id="ext-iter",
            callback_id=None,
            payload_hash=test_db.hash_payload(payload),
        )

        req = robot.IterationToTaskPlanRequest(
            robot_task_id=robot_task_id,
            auto_execute=False,
        )
        resp = asyncio.run(robot.iteration_to_taskplan(req, BackgroundTasks()))
        plan_id = resp.get("task_id")
        assert isinstance(plan_id, str) and plan_id.startswith(f"iter_{robot_task_id}_")

        plan_row = test_db.get_plan(plan_id)
        assert plan_row is not None
        assert plan_row.get("status") == "pending"

        latest = {row["step_id"]: row for row in test_db.list_latest_plan_steps(plan_id)}
        assert latest.get("step_1", {}).get("status") == "pending"
        assert latest.get("step_2", {}).get("status") == "pending"
        assert latest.get("step_3", {}).get("status") == "pending"
    finally:
        robot.db = old_db
        _cleanup_db_files(db_path)

