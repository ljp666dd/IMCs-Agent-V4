import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class StrategyTracker:
    """Track evidence-gap outcomes to inform future strategy ordering."""

    def __init__(self, stats_path: Optional[str] = None):
        self.stats_path = stats_path or os.path.join(self._repo_root(), "data", "strategy", "strategy_stats.json")

    def _repo_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    def _load(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.stats_path):
                return {"evidence_types": {}}
            with open(self.stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {"evidence_types": {}}
        except Exception:
            return {"evidence_types": {}}

    def _save(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _extract_counts(self, stats: Dict[str, Any]) -> Dict[str, int]:
        evidence_by_source = stats.get("evidence_by_source", {}) or {}
        return {
            "literature": int(evidence_by_source.get("literature", 0) or 0),
            "ml_prediction": int(evidence_by_source.get("ml_prediction", 0) or 0),
            "experiment": int(evidence_by_source.get("experiment", 0) or 0),
            "activity_metric": int(stats.get("activity_materials", 0) or 0),
            "adsorption_energy": int(stats.get("adsorption_materials", 0) or 0),
            "formation_energy": int(stats.get("formation_energy_count", 0) or 0),
            "dos_data": int(stats.get("dos_count", 0) or 0),
        }

    def update_from_plan(self, plan: Any) -> Optional[Dict[str, Any]]:
        if not plan or not getattr(plan, "results", None):
            return None
        results = plan.results or {}
        before = results.get("evidence_stats_before_gap")
        after = results.get("evidence_stats_after_gap")
        gap_report = results.get("evidence_gap") or {}
        summary = gap_report.get("summary") or {}
        if not before or not after or not summary:
            return None

        before_counts = self._extract_counts(before)
        after_counts = self._extract_counts(after)
        delta = {k: max(0, after_counts.get(k, 0) - before_counts.get(k, 0)) for k in before_counts.keys()}

        data = self._load()
        evidence_types = data.setdefault("evidence_types", {})

        for key, missing_count in summary.items():
            if key == "materials_total" or missing_count <= 0:
                continue
            entry = evidence_types.setdefault(key, {"attempts": 0, "gains": 0, "score": 0.0})
            entry["attempts"] += 1
            if delta.get(key, 0) > 0:
                entry["gains"] += delta.get(key, 0)
            entry["score"] = round(entry["gains"] / max(entry["attempts"], 1), 3)

        data["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        self._save(data)

        try:
            out_dir = os.path.join(os.path.dirname(self.stats_path), "feedback")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"feedback_{getattr(plan, 'task_id', 'unknown')}.json")
            payload = {
                "task_id": getattr(plan, "task_id", None),
                "summary": summary,
                "delta": delta,
                "updated_at": data["updated_at"],
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return data
