from __future__ import annotations

from typing import Any, Dict, Optional

from src.services.task.executor import PlanExecutor

_executor_agents: Optional[Dict[str, Any]] = None


def get_executor_agents() -> Dict[str, Any]:
    """Lazy-init agent registry used by PlanExecutor.

    Keep imports inside the function to avoid heavy module load on startup.
    """
    global _executor_agents
    if _executor_agents is None:
        from src.agents.core.theory_agent import TheoryDataAgent
        from src.agents.core.ml_agent import MLAgent
        from src.agents.core.experiment_agent import ExperimentDataAgent
        from src.agents.core.literature_agent import LiteratureAgent

        _executor_agents = {
            "theory": TheoryDataAgent(),
            "ml": MLAgent(),
            "experiment": ExperimentDataAgent(),
            "literature": LiteratureAgent(),
        }
    return _executor_agents


def new_plan_executor() -> PlanExecutor:
    """Create a fresh PlanExecutor to avoid cross-task state contamination."""
    return PlanExecutor(get_executor_agents())

