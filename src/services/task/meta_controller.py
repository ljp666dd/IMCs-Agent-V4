from typing import Dict, Any, List, Tuple
import os
import json

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
        self.gap_strategies = self._load_gap_strategies()

    def _load_gap_strategies(self) -> Dict[str, Any]:
        """Load gap strategy templates from configs/gap_strategies.json."""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            path = os.path.join(base_dir, "configs", "gap_strategies.json")
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _format_params(self, params: Any, user_request: str) -> Any:
        if isinstance(params, dict):
            return {k: self._format_params(v, user_request) for k, v in params.items()}
        if isinstance(params, list):
            return [self._format_params(v, user_request) for v in params]
        if isinstance(params, str):
            return params.replace("{query}", user_request or "")
        return params

    def _strategy_steps(self, summary: Dict[str, int], user_request: str, wants_hor: bool) -> Tuple[List[Dict[str, Any]], set]:
        steps: List[Dict[str, Any]] = []
        covered = set()
        if not self.gap_strategies:
            return steps, covered
        strategy = None
        if wants_hor and isinstance(self.gap_strategies.get("hor"), dict):
            strategy = self.gap_strategies.get("hor")
        if strategy is None:
            strategy = self.gap_strategies.get("default")
        if not isinstance(strategy, dict):
            return steps, covered
        for key, items in strategy.items():
            if summary.get(key, 0) <= 0:
                continue
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                step = dict(item)
                step["params"] = self._format_params(step.get("params") or {}, user_request)
                steps.append(step)
            covered.add(key)
        return steps, covered

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
        request_lower = (user_request or "").lower()
        wants_hor = "hor" in request_lower or "hydrogen oxidation" in request_lower

        if decisions.get("need_literature"):
            specs.append({
                "agent": "literature",
                "action": "search",
                "params": {"query": user_request, "limit": 10},
                "deps": [],
            })
            if wants_hor:
                specs.append({
                    "agent": "literature",
                    "action": "harvest_hor_seed",
                    "params": {
                        "query": user_request,
                        "limit": 12,
                        "max_pdfs": 5,
                        "min_elements": 2,
                        "persist": True,
                    },
                    "deps": [],
                })
            specs.append({
                "agent": "literature",
                "action": "extract_knowledge",
                "params": {"topic": user_request},
                "deps": ["$literature"],
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
            specs.append({
                "agent": "task_manager",
                "action": "knowledge_pack",
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

    def _detect_hor(self, user_request: str) -> bool:
        request_lower = (user_request or "").lower()
        return "hor" in request_lower or "hydrogen oxidation" in request_lower

    def _infer_activity_metric(self, user_request: str) -> str:
        request_lower = (user_request or "").lower()
        if "mass activity" in request_lower or "ma" in request_lower:
            return "mass_activity"
        if "jk" in request_lower or "kinetic" in request_lower:
            return "Jk_ref"
        if "tafel" in request_lower:
            return "tafel_slope"
        if "overpotential" in request_lower:
            return "overpotential_10mA"
        if "j0" in request_lower or "exchange" in request_lower:
            return "exchange_current_density"
        return "exchange_current_density"

    def _required_evidence(self, task_type: TaskType, user_request: str) -> Tuple[List[str], List[str]]:
        """Return required evidence sources and material fields for the task."""
        request_lower = (user_request or "").lower()
        wants_hor = self._detect_hor(user_request)

        required_sources = set()
        required_fields = set()

        # Default expectations by task type
        if task_type in (TaskType.CATALYST_DISCOVERY, TaskType.LITERATURE_REVIEW):
            required_sources.add("literature")
        if task_type in (TaskType.CATALYST_DISCOVERY, TaskType.PROPERTY_PREDICTION):
            required_sources.add("ml_prediction")
            required_fields.add("formation_energy")
        if task_type == TaskType.PERFORMANCE_ANALYSIS:
            required_sources.add("activity_metric")
            required_sources.add("experiment")

        # Query hints
        if "literature" in request_lower or "paper" in request_lower or "knowledge" in request_lower:
            required_sources.add("literature")
        if "adsorption" in request_lower:
            required_sources.add("adsorption_energy")
        if "dos" in request_lower:
            required_fields.add("dos_data")

        # HOR-specific evidence
        if wants_hor:
            required_sources.update(["activity_metric", "adsorption_energy"])
            required_fields.add("dos_data")

        return sorted(required_sources), sorted(required_fields)

    def analyze_evidence_gap(
        self,
        material_ids: List[str],
        task_type: TaskType,
        user_request: str,
    ) -> Dict[str, Any]:
        """Analyze evidence gaps for candidate materials."""
        if not material_ids:
            return {"summary": {}, "materials": {}, "recommended_steps": []}

        required_sources, required_fields = self._required_evidence(task_type, user_request)
        evidence_counts = self.db.get_evidence_counts(material_ids)
        feature_flags = self.db.get_material_feature_flags(material_ids)

        summary: Dict[str, int] = {k: 0 for k in (required_sources + required_fields)}
        materials: Dict[str, Any] = {}

        for mid in material_ids:
            counts = evidence_counts.get(mid, {})
            flags = feature_flags.get(mid, {})
            missing = []

            for src in required_sources:
                if counts.get(src, 0) <= 0:
                    missing.append(src)
            for field in required_fields:
                if not flags.get(field, False):
                    missing.append(field)

            for miss in missing:
                summary[miss] = summary.get(miss, 0) + 1

            materials[mid] = {
                "missing": missing,
                "counts": counts,
                "flags": flags,
            }

        summary["materials_total"] = len(material_ids)
        recommended_steps = self._recommend_gap_steps(summary, user_request)

        return {
            "required_sources": required_sources,
            "required_fields": required_fields,
            "summary": summary,
            "materials": materials,
            "recommended_steps": recommended_steps,
        }

    def _recommend_gap_steps(self, summary: Dict[str, int], user_request: str) -> List[Dict[str, Any]]:
        """Convert gap summary into recommended agent actions."""
        steps: List[Dict[str, Any]] = []
        wants_hor = self._detect_hor(user_request)
        stats = self.get_stats()
        activity_count = stats.get("activity_materials", 0) or 0
        strategy_steps, covered = self._strategy_steps(summary, user_request, wants_hor)
        if strategy_steps:
            steps.extend(strategy_steps)

        if summary.get("literature", 0) > 0 and "literature" not in covered:
            steps.append({
                "agent": "literature",
                "action": "search",
                "params": {"query": user_request, "limit": 10},
                "reason": "Missing literature evidence for candidate materials.",
            })
            if wants_hor:
                steps.append({
                    "agent": "literature",
                    "action": "harvest_hor_seed",
                    "params": {"query": user_request, "limit": 12, "max_pdfs": 5, "min_elements": 2, "persist": True},
                    "reason": "HOR-specific metrics missing; harvest seed evidence.",
                })

        data_types = []
        if summary.get("formation_energy", 0) > 0 and "formation_energy" not in covered:
            data_types.append("formation_energy")
        if summary.get("dos_data", 0) > 0 and "dos_data" not in covered:
            data_types.append("dos")
        if summary.get("adsorption_energy", 0) > 0 and "adsorption_energy" not in covered:
            data_types.append("adsorption")
        if data_types:
            data_types = ["cif"] + sorted(set(data_types))
            steps.append({
                "agent": "theory",
                "action": "download",
                "params": {"data_types": data_types, "limit": 50},
                "reason": "Missing theory/DOS/adsorption evidence.",
            })

        if summary.get("activity_metric", 0) > 0 and "activity_metric" not in covered:
            steps.append({
                "agent": "experiment",
                "action": "process",
                "params": {
                    "data_dir": "data/experimental/rde_lsv",
                    "reference_potential": 0.2,
                    "loading_mg_cm2": 0.25,
                    "precious_fraction": 0.20,
                },
                "reason": "Missing activity metrics; process local experiment data.",
            })

        if summary.get("ml_prediction", 0) > 0 and "ml_prediction" not in covered:
            metric_name = self._infer_activity_metric(user_request)
            target_col = f"activity_metric:{metric_name}" if activity_count >= 5 else None
            steps.append({
                "agent": "ml",
                "action": "train",
                "params": {"include_deep_learning": True, "target_col": target_col},
                "reason": "Missing ML predictions for candidate materials.",
            })

        # Ensure ML target_col filled if strategy produced ml step without target
        if activity_count >= 5:
            metric_name = self._infer_activity_metric(user_request)
            for step in steps:
                if step.get("agent") == "ml" and step.get("action") == "train":
                    params = step.get("params") or {}
                    if params.get("target_col") in (None, "", "auto"):
                        params["target_col"] = f"activity_metric:{metric_name}"
                        step["params"] = params

        return steps
