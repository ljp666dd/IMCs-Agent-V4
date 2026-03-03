import os
from datetime import datetime

from src.services.db.database import DatabaseService
from src.services.task.executor import PlanExecutor
from src.services.task.types import TaskPlan, TaskStep, TaskType


class _DummyPaper:
    def __init__(self, title: str):
        self.title = title


class _DummyLiteratureAgent:
    def search_all_sources(self, query: str, limit: int = 5):
        if len(query or "") > 80:
            raise TimeoutError("timeout while searching")
        return [_DummyPaper("ok")]

    def extract_knowledge(self, topic: str):
        return {"topic": topic, "notes": []}


class _DummyTheoryAgent:
    def download_structures(self, limit: int = 50):
        raise RuntimeError("403 forbidden api key")

    def download_formation_energy(self):
        return None

    def download_orbital_dos(self, material_ids=None):
        return None

    def download_adsorption_energies(self, adsorbates=None, limit: int = 50):
        return None

    def get_status(self):
        return {}


def _cleanup_db_files(db_path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db_path + suffix)
        except Exception:
            pass


def test_failure_policy_replan_overrides_max_replans_zero():
    db_path = os.path.join("data", f"test_failure_policy_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
    db = DatabaseService(db_path=db_path)
    try:
        executor = PlanExecutor(
            agents={"literature": _DummyLiteratureAgent()},
            db=db,
        )
        executor.max_adaptive_rounds = 0

        long_query = "this is a very long query " + ("with many tokens " * 30)
        plan = TaskPlan(
            task_id=f"task_failure_replan_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            task_type=TaskType.CATALYST_DISCOVERY,
            description="failure policy replan test",
            steps=[
                TaskStep(
                    step_id="step_1",
                    agent="literature",
                    action="search",
                    params={"query": long_query, "limit": 5},
                    dependencies=[],
                    max_retries=0,
                    max_replans=0,
                ),
                TaskStep(
                    step_id="step_2",
                    agent="task_manager",
                    action="recommend",
                    params={},
                    dependencies=["step_1"],
                ),
            ],
        )

        executor.execute_plan(plan)
        assert plan.status == "completed"

        latest = {row["step_id"]: row for row in db.list_latest_plan_steps(plan.task_id)}
        assert latest.get("step_1", {}).get("status") == "replanned"
        assert latest.get("step_2", {}).get("status") == "completed"
        # The replanned literature step should exist and complete.
        assert latest.get("step_3", {}).get("agent") == "literature"
        assert latest.get("step_3", {}).get("status") == "completed"
    finally:
        _cleanup_db_files(db_path)


def test_failure_policy_skip_marks_step_completed_and_adds_warning():
    db_path = os.path.join("data", f"test_failure_policy_skip_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db")
    db = DatabaseService(db_path=db_path)
    try:
        executor = PlanExecutor(
            agents={"theory": _DummyTheoryAgent()},
            db=db,
        )
        executor.max_adaptive_rounds = 0

        plan = TaskPlan(
            task_id=f"task_failure_skip_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            task_type=TaskType.CATALYST_DISCOVERY,
            description="failure policy skip test",
            steps=[
                TaskStep(
                    step_id="step_1",
                    agent="theory",
                    action="download",
                    params={"data_types": ["cif"], "limit": 1},
                    dependencies=[],
                    max_retries=0,
                    max_replans=0,
                ),
                TaskStep(
                    step_id="step_2",
                    agent="task_manager",
                    action="recommend",
                    params={},
                    dependencies=["step_1"],
                ),
            ],
        )

        executor.execute_plan(plan)
        assert plan.status == "completed"

        latest = {row["step_id"]: row for row in db.list_latest_plan_steps(plan.task_id)}
        assert latest.get("step_1", {}).get("status") == "skipped"
        assert latest.get("step_2", {}).get("status") == "completed"

        warnings = plan.results.get("warnings") or []
        assert any(w.get("step_id") == "step_1" and w.get("category") == "auth" for w in warnings)
    finally:
        _cleanup_db_files(db_path)

