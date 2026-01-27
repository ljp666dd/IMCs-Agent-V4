from typing import Dict, Any, List, Tuple

from src.services.db.database import DatabaseService
from src.services.task.types import TaskType


class MetaController:
    """
    Rule-based meta-controller for adaptive planning.
    Decides which agents to call based on evidence coverage.
    """

    def __init__(self, db: DatabaseService = None, thresholds: Dict[str, int] = None):
        self.db = db or DatabaseService()
        self.thresholds = thresholds or {
            "materials_min": 50,
            "literature_min": 10,
            "adsorption_min": 10,
            "activity_min": 5,
            "model_min": 1,
            "dos_min": 10,
        }

    def get_stats(self) -> Dict[str, Any]:
        try:
            from src.agents.core.theory_agent import TheoryDataConfig
            allowed = TheoryDataConfig().elements
        except Exception:
            allowed = None
        return self.db.get_evidence_stats(allowed_elements=allowed)

    def decide(self, task_type: TaskType, user_request: str, stats: Dict[str, Any]) -> Dict[str, bool]:
        request_lower = (user_request or "").lower()
        evidence = stats.get("evidence_by_source", {}) or {}

        total_materials = stats.get("total_materials", 0) or 0
        literature_cov = evidence.get("literature", 0)
        adsorption_cov = stats.get("adsorption_materials", 0) or 0
        activity_cov = stats.get("activity_materials", 0) or 0
        model_count = stats.get("model_count", 0) or 0
        dos_count = stats.get("dos_count", 0) or 0

        need_literature = literature_cov < self.thresholds["literature_min"] or "paper" in request_lower or "literature" in request_lower
        need_theory = total_materials < self.thresholds["materials_min"] or "download" in request_lower or "theory" in request_lower
        need_dos = dos_count < self.thresholds["dos_min"]
        need_adsorption = adsorption_cov < self.thresholds["adsorption_min"] or "adsorption" in request_lower
        need_activity = activity_cov < self.thresholds["activity_min"]
        need_ml = model_count < self.thresholds["model_min"] or "train" in request_lower or "model" in request_lower or "ml" in request_lower

        # Adjust for task type
        if task_type == TaskType.LITERATURE_REVIEW:
            need_theory = False
            need_ml = False
        if task_type == TaskType.PROPERTY_PREDICTION:
            need_literature = False
        if task_type == TaskType.PERFORMANCE_ANALYSIS:
            need_theory = False

        return {
            "need_literature": need_literature,
            "need_theory": need_theory,
            "need_dos": need_dos,
            "need_adsorption": need_adsorption,
            "need_activity": need_activity,
            "need_ml": need_ml,
        }

    def _build_step_specs(self, task_type: TaskType, user_request: str, decisions: Dict[str, bool]) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []

        if decisions.get("need_literature"):
            specs.append({
                "agent": "literature",
                "action": "search",
                "params": {"query": user_request, "limit": 10},
                "deps": [],
            })

        if decisions.get("need_theory") or decisions.get("need_dos") or decisions.get("need_adsorption"):
            data_types = ["cif", "formation_energy"]
            if decisions.get("need_dos"):
                data_types.append("dos")
            if decisions.get("need_adsorption"):
                data_types.append("adsorption")
            specs.append({
                "agent": "theory",
                "action": "download",
                "params": {"data_types": data_types, "limit": 50},
                "deps": [],
            })

        if task_type in (TaskType.CATALYST_DISCOVERY, TaskType.PROPERTY_PREDICTION) and decisions.get("need_ml"):
            deps = []
            for spec in specs:
                if spec["agent"] == "theory":
                    deps.append("$theory")
            specs.append({
                "agent": "ml",
                "action": "train",
                "params": {"include_deep_learning": True},
                "deps": deps,
            })

        if task_type == TaskType.PERFORMANCE_ANALYSIS:
            specs.append({
                "agent": "experiment",
                "action": "process",
                "params": {},
                "deps": [],
            })

        if task_type in (TaskType.CATALYST_DISCOVERY, TaskType.PERFORMANCE_ANALYSIS):
            deps = []
            for spec in specs:
                if spec["agent"] == "literature":
                    deps.append("$literature")
                if spec["agent"] == "ml":
                    deps.append("$ml")
                if spec["agent"] == "experiment":
                    deps.append("$experiment")
            specs.append({
                "agent": "task_manager",
                "action": "recommend",
                "params": {},
                "deps": deps,
            })

        if task_type == TaskType.LITERATURE_REVIEW:
            specs.append({
                "agent": "literature",
                "action": "extract_knowledge",
                "params": {"topic": user_request},
                "deps": ["$literature"] if decisions.get("need_literature") else [],
            })
            specs.append({
                "agent": "task_manager",
                "action": "summarize",
                "params": {},
                "deps": ["$literature"],
            })

        if not specs:
            specs.append({
                "agent": "task_manager",
                "action": "analyze",
                "params": {"request": user_request},
                "deps": [],
            })

        return specs

    def build_initial_steps(self, task_type: TaskType, user_request: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        stats = self.get_stats()
        decisions = self.decide(task_type, user_request, stats)
        return self._build_step_specs(task_type, user_request, decisions), stats

    def suggest_followups(self, task_type: TaskType, user_request: str, existing_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        stats = self.get_stats()
        decisions = self.decide(task_type, user_request, stats)

        existing = {(s.get("agent"), s.get("action")) for s in existing_steps}
        specs = self._build_step_specs(task_type, user_request, decisions)
        filtered: List[Dict[str, Any]] = []

        for spec in specs:
            key = (spec.get("agent"), spec.get("action"))
            if key in existing:
                continue
            filtered.append(spec)

        return filtered
