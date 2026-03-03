import os
from datetime import datetime
import asyncio

from src.services.db.database import DatabaseService
from src.services.task.executor import PlanExecutor
from src.services.task.types import TaskPlan, TaskStep, TaskType
from src.api.routers.tasks import _restore_plan_from_db, GapConfirmRequest, confirm_gap_fill
from src.agents.core.theory_agent import TheoryDataAgent
from src.agents.core.ml_agent import MLAgent
from src.agents.core.experiment_agent import ExperimentDataAgent
from src.agents.core.literature_agent import LiteratureAgent


def _latest_step_map(steps):
    latest = {}
    for step in steps:
        latest[step["step_id"]] = step
    return latest


def test_gap_confirmation_flow():
    os.environ["IMCS_EVIDENCE_AUTO_FILL"] = "0"
    os.environ["IMCS_EVIDENCE_GAP_ROUNDS"] = "1"
    os.environ["IMCS_ENABLE_CLOUD_LLM"] = "0"

    db = DatabaseService()
    material_id = f"test_gap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.save_material(material_id=material_id, formula="Pt", energy=None, cif_path=None)

    agents = {
        "theory": TheoryDataAgent(),
        "ml": MLAgent(),
        "experiment": ExperimentDataAgent(),
        "literature": LiteratureAgent(),
    }

    plan = TaskPlan(
        task_id=f"task_gapflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        task_type=TaskType.CATALYST_DISCOVERY,
        description="HOR gap confirmation test",
        steps=[
            TaskStep(
                step_id="step_1",
                agent="task_manager",
                action="knowledge_pack",
                params={},
                dependencies=[],
            )
        ],
    )

    try:
        executor = PlanExecutor(agents)
        executor.max_adaptive_rounds = 0
        executor.execute_plan(plan)

        status = (db.get_plan(plan.task_id) or {}).get("status")
        assert status == "awaiting_confirmation"

        steps = db.list_plan_steps(plan.task_id)
        latest = _latest_step_map(steps)
        pending_ids = [
            sid for sid, row in latest.items()
            if row.get("status") in ("pending", "running")
        ]
        assert pending_ids

        run_ids = [
            sid for sid, row in latest.items()
            if row.get("agent") == "experiment"
        ]
        assert run_ids

        asyncio.run(
            confirm_gap_fill(
                plan.task_id,
                GapConfirmRequest(
                    run_step_ids=run_ids,
                    mark_complete=False,
                    params_overrides={},
                ),
            )
        )

        plan2 = _restore_plan_from_db(plan.task_id)
        executor2 = PlanExecutor(agents)
        executor2.max_adaptive_rounds = 0
        executor2.execute_plan(plan2)

        status2 = (db.get_plan(plan.task_id) or {}).get("status")
        assert status2 in ("completed", "awaiting_confirmation")

        steps2 = db.list_plan_steps(plan.task_id)
        latest2 = _latest_step_map(steps2)
        for sid in run_ids:
            assert latest2.get(sid, {}).get("status") == "completed"
    finally:
        with db._get_conn() as conn:
            conn.execute("DELETE FROM evidence WHERE material_id = ?", (material_id,))
            conn.execute("DELETE FROM activity_metrics WHERE material_id = ?", (material_id,))
            conn.execute("DELETE FROM adsorption_energies WHERE material_id = ?", (material_id,))
            conn.execute("DELETE FROM materials WHERE material_id = ?", (material_id,))


def test_gap_mark_complete_skips_pending_steps():
    os.environ["IMCS_EVIDENCE_AUTO_FILL"] = "0"
    os.environ["IMCS_EVIDENCE_GAP_ROUNDS"] = "1"
    os.environ["IMCS_ENABLE_CLOUD_LLM"] = "0"

    db = DatabaseService()
    material_id = f"test_gap_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.save_material(material_id=material_id, formula="Pt", energy=None, cif_path=None)

    agents = {
        "theory": TheoryDataAgent(),
        "ml": MLAgent(),
        "experiment": ExperimentDataAgent(),
        "literature": LiteratureAgent(),
    }

    plan = TaskPlan(
        task_id=f"task_gapcomplete_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        task_type=TaskType.CATALYST_DISCOVERY,
        description="HOR gap mark-complete test",
        steps=[
            TaskStep(
                step_id="step_1",
                agent="task_manager",
                action="knowledge_pack",
                params={},
                dependencies=[],
            )
        ],
    )

    try:
        executor = PlanExecutor(agents)
        executor.max_adaptive_rounds = 0
        executor.execute_plan(plan)

        assert (db.get_plan(plan.task_id) or {}).get("status") == "awaiting_confirmation"

        steps = db.list_plan_steps(plan.task_id)
        latest = _latest_step_map(steps)
        pending_ids = [
            sid for sid, row in latest.items()
            if row.get("status") in ("pending", "running")
        ]
        assert pending_ids

        asyncio.run(
            confirm_gap_fill(
                plan.task_id,
                GapConfirmRequest(run_step_ids=[], mark_complete=True, params_overrides={}),
            )
        )

        assert (db.get_plan(plan.task_id) or {}).get("status") == "completed"

        steps2 = db.list_plan_steps(plan.task_id)
        latest2 = _latest_step_map(steps2)
        for sid in pending_ids:
            assert latest2.get(sid, {}).get("status") == "skipped"
    finally:
        with db._get_conn() as conn:
            conn.execute("DELETE FROM evidence WHERE material_id = ?", (material_id,))
            conn.execute("DELETE FROM activity_metrics WHERE material_id = ?", (material_id,))
            conn.execute("DELETE FROM adsorption_energies WHERE material_id = ?", (material_id,))
            conn.execute("DELETE FROM materials WHERE material_id = ?", (material_id,))


def test_gap_confirm_supports_params_overrides_and_skip_all():
    os.environ["IMCS_EVIDENCE_AUTO_FILL"] = "0"
    os.environ["IMCS_EVIDENCE_GAP_ROUNDS"] = "1"
    os.environ["IMCS_ENABLE_CLOUD_LLM"] = "0"

    db = DatabaseService()
    material_id = f"test_gap_override_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.save_material(material_id=material_id, formula="Pt", energy=None, cif_path=None)

    agents = {
        "theory": TheoryDataAgent(),
        "ml": MLAgent(),
        "experiment": ExperimentDataAgent(),
        "literature": LiteratureAgent(),
    }

    plan = TaskPlan(
        task_id=f"task_gapoverride_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        task_type=TaskType.CATALYST_DISCOVERY,
        description="HOR gap params override test",
        steps=[
            TaskStep(
                step_id="step_1",
                agent="task_manager",
                action="knowledge_pack",
                params={},
                dependencies=[],
            )
        ],
    )

    try:
        executor = PlanExecutor(agents)
        executor.max_adaptive_rounds = 0
        executor.execute_plan(plan)

        assert (db.get_plan(plan.task_id) or {}).get("status") == "awaiting_confirmation"

        steps = db.list_plan_steps(plan.task_id)
        latest = _latest_step_map(steps)
        pending_ids = [
            sid for sid, row in latest.items()
            if row.get("status") in ("pending", "running")
        ]
        assert pending_ids

        target_id = pending_ids[0]
        override_params = {"reference_potential": 0.123, "loading_mg_cm2": 0.99}

        asyncio.run(
            confirm_gap_fill(
                plan.task_id,
                GapConfirmRequest(
                    run_step_ids=[target_id],
                    mark_complete=False,
                    params_overrides={target_id: override_params},
                ),
            )
        )

        steps2 = db.list_plan_steps(plan.task_id)
        latest2 = _latest_step_map(steps2)
        assert latest2.get(target_id, {}).get("status") == "pending"
        assert (latest2.get(target_id, {}).get("params") or {}).get("reference_potential") == 0.123
        assert (latest2.get(target_id, {}).get("params") or {}).get("loading_mg_cm2") == 0.99

        asyncio.run(
            confirm_gap_fill(
                plan.task_id,
                GapConfirmRequest(run_step_ids=[], mark_complete=False, params_overrides={}),
            )
        )

        steps3 = db.list_plan_steps(plan.task_id)
        latest3 = _latest_step_map(steps3)
        skipped = [
            sid for sid in pending_ids
            if latest3.get(sid, {}).get("status") == "skipped"
        ]
        assert skipped
    finally:
        with db._get_conn() as conn:
            conn.execute("DELETE FROM evidence WHERE material_id = ?", (material_id,))
            conn.execute("DELETE FROM activity_metrics WHERE material_id = ?", (material_id,))
            conn.execute("DELETE FROM adsorption_energies WHERE material_id = ?", (material_id,))
            conn.execute("DELETE FROM materials WHERE material_id = ?", (material_id,))
