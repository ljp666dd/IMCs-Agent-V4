import json
from typing import Dict, Any, Optional, List

from src.services.db.database import DatabaseService


class PlanEvaluator:
    """
    Compute lightweight evaluation metrics for a completed plan.
    Metrics are heuristic and designed for system health checks.
    """

    def __init__(self, db: Optional[DatabaseService] = None):
        self.db = db or DatabaseService()

    def _best_model_r2(self, plan_results: Dict[str, Any]) -> Optional[float]:
        best = None
        for val in plan_results.values():
            if isinstance(val, dict) and "models" in val:
                models = val.get("models") or []
                for m in models:
                    try:
                        r2 = m.get("r2_test")
                    except Exception:
                        r2 = None
                    if isinstance(r2, (int, float)):
                        if best is None or r2 > best:
                            best = r2
        return best

    def evaluate(self, plan) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"version": "1.0"}
        results = getattr(plan, "results", {}) or {}
        if hasattr(plan, "status"):
            metrics["plan_status"] = plan.status

        try:
            from src.agents.core.theory_agent import TheoryDataConfig
            allowed = TheoryDataConfig().elements
        except Exception:
            allowed = None

        stats = self.db.get_evidence_stats(allowed_elements=allowed)
        metrics["evidence_stats"] = stats
        metrics["model_count"] = stats.get("model_count")
        metrics["candidate_count"] = len(results.get("candidate_material_ids") or [])

        gap = results.get("evidence_gap") or {}
        summary = gap.get("summary") or {}
        if summary:
            metrics["evidence_gap_summary"] = summary
            total_missing = 0
            for k, v in summary.items():
                if k == "materials_total":
                    continue
                if isinstance(v, int):
                    total_missing += v
            metrics["evidence_gap_total_missing"] = total_missing
            rec_steps = gap.get("recommended_steps") or []
            metrics["gap_recommendation_count"] = len(rec_steps)
            total_materials = summary.get("materials_total") or 0
            if total_materials:
                coverage = {}
                overall_vals = []
                for key, missing in summary.items():
                    if key == "materials_total":
                        continue
                    if isinstance(missing, int):
                        cov = round(1.0 - (missing / total_materials), 4)
                        coverage[key] = cov
                        overall_vals.append(cov)
                if coverage:
                    metrics["evidence_coverage_by_key"] = coverage
                    metrics["evidence_coverage_overall"] = round(
                        sum(overall_vals) / len(overall_vals), 4
                    )

        ranking = results.get("ranking_current") or results.get("ranking_before_gap") or []
        if ranking:
            top_ids = [r.get("material_id") for r in ranking if r.get("material_id")]
            if top_ids:
                counts = self.db.get_evidence_counts(top_ids)
                evidence_hits = sum(1 for mid in top_ids if counts.get(mid))
                activity_hits = sum(
                    1
                    for mid in top_ids
                    if (counts.get(mid) or {}).get("activity_metric", 0) > 0
                )
                metrics["ranking_top_n"] = len(top_ids)
                metrics["ranking_evidence_hit_rate"] = round(evidence_hits / len(top_ids), 3)
                metrics["ranking_activity_hit_rate"] = round(activity_hits / len(top_ids), 3)

        rag = results.get("knowledge_rag") or []
        rag_count = 0
        for item in rag:
            if isinstance(item, dict):
                rag_count += len(item.get("results") or [])
        metrics["rag_results_count"] = rag_count

        best_r2 = self._best_model_r2(results)
        if best_r2 is not None:
            metrics["best_model_r2"] = round(best_r2, 4)

        metrics["ranking_metric"] = results.get("ranking_metric")
        return metrics
