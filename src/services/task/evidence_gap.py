"""
IMCs Evidence Gap Engine — 证据缺口处理

从 executor.py 拆分，负责：
1. 追加 ML 活性训练步骤
2. 执行证据缺口补齐步骤
3. 缺口补齐后重新计算排名和 RAG
4. 合并 knowledge pack 持久化数据
"""

import json
import os
from typing import Dict, Any, Optional, List

from src.core.logger import get_logger
from src.services.task.types import TaskPlan, TaskStep
from src.services.task.replan_engine import next_step_id

logger = get_logger(__name__)


def append_activity_ml_step(
    plan: TaskPlan,
    pending: Dict[str, TaskStep],
    db,
    depends_on: str,
    metric_name: str = "exchange_current_density",
) -> Optional[str]:
    """Append an ML step to train on activity metrics if not already present."""
    for step in plan.steps:
        if step.agent == "ml" and step.action in ("train", "train_all"):
            params = step.params or {}
            target_col = params.get("target_col") if isinstance(params, dict) else None
            if isinstance(target_col, str) and target_col.startswith("activity_metric:"):
                return None

    step_id = next_step_id(plan)
    step = TaskStep(
        step_id=step_id,
        agent="ml",
        action="train",
        params={"include_deep_learning": True, "target_col": f"activity_metric:{metric_name}"},
        dependencies=[depends_on],
        max_replans=0,
    )
    plan.steps.append(step)
    pending[step.step_id] = step
    db.log_plan_step(
        plan_id=plan.task_id,
        step_id=step.step_id,
        agent=step.agent,
        action=step.action,
        status="pending",
        dependencies=step.dependencies,
        params=step.params,
    )
    return step.step_id


def execute_gap_steps(
    plan: TaskPlan,
    gap_steps: List[Dict[str, Any]],
    ml_predictions: Dict[str, Any],
    literature_papers: List[Any],
    ml_target: Optional[str],
    db,
    execute_step_fn,
) -> Dict[str, Any]:
    """Execute evidence gap steps sequentially (best-effort)."""
    success = True
    for item in gap_steps:
        step_id = next_step_id(plan)
        step = TaskStep(
            step_id=step_id,
            agent=item.get("agent", ""),
            action=item.get("action", ""),
            params=item.get("params") or {},
            dependencies=item.get("deps") or [],
            status="pending",
        )
        plan.steps.append(step)
        db.log_plan_step(
            plan_id=plan.task_id,
            step_id=step.step_id,
            agent=step.agent,
            action=step.action,
            status="pending",
            dependencies=step.dependencies,
            params=step.params,
        )

        try:
            db.log_plan_step(
                plan_id=plan.task_id,
                step_id=step.step_id,
                agent=step.agent,
                action=step.action,
                status="running",
                dependencies=step.dependencies,
                params=step.params,
            )
            result = execute_step_fn(step)
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(result.get("error"))
            step.result = result
            step.status = "completed"
            plan.results[step.step_id] = result

            if step.agent == "literature" and isinstance(result, list):
                if result and hasattr(result[0], "title"):
                    literature_papers = result
            if step.agent == "ml":
                if isinstance(result, dict) and "predictions" in result:
                    ml_predictions = result.get("predictions", {}) or {}
                if isinstance(step.params, dict) and step.params.get("target_col"):
                    ml_target = step.params.get("target_col")

            db.log_plan_step(
                plan_id=plan.task_id,
                step_id=step.step_id,
                agent=step.agent,
                action=step.action,
                status="completed",
                result=result,
                dependencies=step.dependencies,
                params=step.params,
            )
        except Exception as exc:
            step.status = "failed"
            db.log_plan_step(
                plan_id=plan.task_id,
                step_id=step.step_id,
                agent=step.agent,
                action=step.action,
                status="failed",
                error=str(exc),
                dependencies=step.dependencies,
                params=step.params,
            )
            success = False
            break

    return {
        "success": success,
        "ml_predictions": ml_predictions,
        "literature_papers": literature_papers,
        "ml_target": ml_target,
    }


def recompute_evidence_post_gap(
    plan: TaskPlan,
    ml_predictions: Dict[str, Any],
    ml_target: Optional[str],
    agents: Dict[str, Any],
    db,
) -> Dict[str, Any]:
    """Recompute ranking + knowledge RAG after gap auto-fill."""
    rag_results: List[Dict[str, Any]] = []
    candidate_ids_snapshot: List[str] = []
    materials_snapshot: List[Dict[str, Any]] = []
    ranking_current = None
    ranking_metric = None

    try:
        theory_agent = agents.get("theory")
        if not theory_agent:
            return {}
        plan_record = db.get_plan(plan.task_id)
        created_at = plan_record.get("created_at") if plan_record else None
        try:
            from src.agents.core.theory_agent import TheoryDataConfig
            allowed = TheoryDataConfig().elements
        except Exception:
            allowed = None
        if created_at:
            materials = db.list_materials_since(created_at, limit=50, allowed_elements=allowed)
        else:
            materials = theory_agent.list_stored_materials(limit=20)
        if not materials:
            materials = theory_agent.list_stored_materials(limit=20)

        candidate_ids = set()
        top_n = 10
        sorted_preds = None
        if ml_predictions:
            try:
                sorted_preds = sorted(
                    [(mid, score) for mid, score in ml_predictions.items() if mid],
                    key=lambda kv: kv[1],
                    reverse=True
                )
                candidate_ids_ordered = [mid for mid, _ in sorted_preds[:top_n] if mid]
                candidate_ids = set(candidate_ids_ordered)
            except Exception:
                candidate_ids_ordered = [mid for mid in ml_predictions.keys() if mid][:top_n]
                candidate_ids = set(candidate_ids_ordered)
        else:
            candidate_ids_ordered = []
            candidate_ids = {m.get("material_id") for m in materials if m.get("material_id")}

        candidate_ids_snapshot = candidate_ids_ordered or list(candidate_ids)
        if candidate_ids_snapshot:
            loaded = db.list_materials_by_ids(candidate_ids_snapshot, allowed_elements=allowed)
            materials = loaded if loaded else [m for m in materials if m.get("material_id") in candidate_ids]
        else:
            materials = [m for m in materials if m.get("material_id") in candidate_ids]
        candidate_ids_snapshot = candidate_ids_snapshot or []
        materials_snapshot = materials[:]

        if ml_predictions and not sorted_preds:
            sorted_preds = list(ml_predictions.items())
        if sorted_preds:
            ranking_current = []
            for idx, (mid, score) in enumerate(sorted_preds[:top_n], start=1):
                ranking_current.append({
                    "rank": idx,
                    "material_id": mid,
                    "score": score,
                })
            ranking_metric = ml_target or "formation_energy"

        try:
            from src.services.knowledge import KnowledgeRAG, KnowledgeService
            rag = KnowledgeRAG(db.db_path)
            ks = KnowledgeService(db.db_path)
            for mat in materials[:5]:
                mid = mat.get("material_id")
                if not mid:
                    continue
                rag_out = rag.query(
                    query_text=f"HOR activity evidence for {mid}",
                    top_k=3,
                    source_type="literature"
                )
                if rag_out:
                    rag_results.append({
                        "material_id": mid,
                        "results": rag_out
                    })
                    for item in rag_out:
                        source_id = item.get("source_id")
                        if not source_id:
                            continue
                        ks.upsert_material_evidence(
                            material_id=mid,
                            source_type=item.get("source_type") or "literature",
                            source_id=source_id,
                            score=item.get("score"),
                            metadata=item
                        )
        except Exception as e:
            logger.warning(f"Post-gap RAG failed: {e}")
    except Exception as e:
        logger.warning(f"Post-gap evidence recompute failed: {e}")

    return {
        "rag_results": rag_results,
        "candidate_ids_snapshot": candidate_ids_snapshot,
        "materials_snapshot": materials_snapshot,
        "ranking_current": ranking_current,
        "ranking_metric": ranking_metric,
    }


def merge_knowledge_pack_results(plan: TaskPlan) -> None:
    """Merge persisted knowledge pack fields into plan.results if present."""
    if not plan or not plan.task_id:
        return
    out_path = os.path.join("data", "tasks", f"knowledge_{plan.task_id}.json")
    if not os.path.exists(out_path):
        return
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            pack = json.load(f)
    except Exception:
        return
    if not isinstance(pack, dict):
        return
    for key in (
        "evidence_gap",
        "evidence_stats_before_gap",
        "candidate_material_ids",
        "ranking_before_gap",
        "ranking_after_gap",
        "ranking_current",
        "ranking_metric",
    ):
        if key in pack and key not in plan.results:
            plan.results[key] = pack[key]
