"""
IMCs Plan Executor — 任务计划执行器

核心调度引擎，负责 DAG 感知的步骤执行循环。
具体逻辑委托给：
- replan_engine: 重规划策略
- evidence_gap: 证据缺口处理
- step_dispatcher: 步骤分发到 Agent
"""

import json
import os
import re
from typing import Dict, Any, Optional, List

from src.core.logger import get_logger, log_exception
from src.services.task.types import TaskPlan, TaskStep
from src.services.db.database import DatabaseService
from src.services.task.failure_policy import FailurePolicyEngine
from src.services.task.replan_engine import (
    load_replan_strategies, build_replan_spec, apply_replan,
    append_dynamic_steps, resolve_gap_deps, next_step_id,
)
from src.services.task.evidence_gap import (
    append_activity_ml_step, execute_gap_steps,
    recompute_evidence_post_gap, merge_knowledge_pack_results,
)
from src.services.task.step_dispatcher import dispatch_step

logger = get_logger(__name__)


class PlanExecutor:
    """
    Service for executing task plans by dispatching to agents.
    Includes persistence to SQLite (v4.0).
    """

    def __init__(self, agents: Dict[str, Any], db: Optional[DatabaseService] = None):
        """
        Args:
            agents: Dictionary mapping agent names to agent instances.
        """
        self.agents = agents
        self.db = db or DatabaseService()
        try:
            from src.services.task.meta_controller import MetaController
            self.meta_controller = MetaController(self.db)
        except Exception:
            self.meta_controller = None
        self.replan_strategies = load_replan_strategies()
        self.failure_policy = FailurePolicyEngine()
        self.max_adaptive_rounds = 1
        self._active_plan: Optional[TaskPlan] = None

    # ------------------------------------------------------------------
    # Delegate helpers (keep thin wrappers for backward compatibility)
    # ------------------------------------------------------------------

    def _next_step_id(self, plan: TaskPlan) -> str:
        return next_step_id(plan)

    def _raise_for_error_result(self, result: Any) -> None:
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result.get("error"))

    def _simplify_query(self, query: str) -> str:
        if not query:
            return ""
        tokens = re.findall(r"[A-Za-z0-9\\-\\+]+", query)
        if not tokens:
            return query.strip()
        return " ".join(tokens[:8]).strip()

    def _repo_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    def _format_params(self, params: Any, template_vars: Dict[str, str]) -> Any:
        from src.services.task.replan_engine import format_params
        return format_params(params, template_vars)

    def _load_replan_strategies(self) -> Dict[str, Any]:
        return load_replan_strategies()

    def _replan_from_strategy(self, step: TaskStep) -> Optional[Dict[str, Any]]:
        from src.services.task.replan_engine import replan_from_strategy
        return replan_from_strategy(self.replan_strategies, step, self._active_plan)

    def _build_replan_spec(self, step: TaskStep) -> Optional[Dict[str, Any]]:
        return build_replan_spec(step, self.replan_strategies, self._active_plan)

    def _apply_replan(self, plan: TaskPlan, failed_step: TaskStep, spec: Dict[str, Any],
                      pending: Dict[str, TaskStep]) -> List[str]:
        return apply_replan(plan, failed_step, spec, pending, self.db)

    def _append_dynamic_steps(self, plan: TaskPlan, specs: List[Dict[str, Any]],
                               pending: Dict[str, TaskStep]) -> List[str]:
        return append_dynamic_steps(plan, specs, pending, self.db)

    def _resolve_gap_deps(self, plan: TaskPlan, deps: List[Any]) -> List[str]:
        return resolve_gap_deps(plan, deps)

    def _append_activity_ml_step(self, plan: TaskPlan, pending: Dict[str, TaskStep],
                                 depends_on: str, metric_name: str = "exchange_current_density") -> Optional[str]:
        return append_activity_ml_step(plan, pending, self.db, depends_on, metric_name)

    def _execute_gap_steps(self, plan, gap_steps, ml_predictions, literature_papers, ml_target):
        return execute_gap_steps(
            plan, gap_steps, ml_predictions, literature_papers, ml_target,
            self.db, self._execute_step,
        )

    def _recompute_evidence_post_gap(self, plan, ml_predictions, ml_target):
        return recompute_evidence_post_gap(plan, ml_predictions, ml_target, self.agents, self.db)

    def _merge_knowledge_pack_results(self, plan: TaskPlan) -> None:
        merge_knowledge_pack_results(plan)

    def _execute_step(self, step) -> Any:
        """Dispatch single step (TaskStep object)."""
        return dispatch_step(step, self.agents, self._active_plan, self.db)

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    @log_exception(logger)
    def execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """Execute all steps in the plan (DAG aware + Persisted)."""
        if not plan:
            return {"error": "No plan provided"}

        # 1. Create Plan Record (M2 Persistence)
        self.db.create_plan(
            plan_id=plan.task_id,
            user_id=None,
            task_type=plan.task_type.value,
            description=plan.description
        )

        plan.status = "executing"
        self.db.update_plan_status(plan.task_id, "executing")

        logger.info(f"Executing Plan: {plan.task_id}")
        self._active_plan = plan
        self._merge_knowledge_pack_results(plan)
        pending = {}
        completed = set()
        for step in plan.steps:
            if step.status in ("completed", "skipped"):
                completed.add(step.step_id)
            elif step.status in ("failed", "blocked"):
                continue
            else:
                pending[step.step_id] = step
        literature_papers = []
        ml_top_models = []
        ml_predictions = {}
        ml_target = None

        adaptive_rounds = 0
        awaiting_confirmation = False
        env_path = os.path.join(self._repo_root(), ".env")
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except Exception:
            pass

        def _read_env_file(key: str) -> Optional[str]:
            try:
                if not os.path.exists(env_path):
                    return None
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith(key + "="):
                            return line.split("=", 1)[1]
            except Exception:
                return None
            return None

        auto_fill_val = os.getenv("IMCS_EVIDENCE_AUTO_FILL") or _read_env_file("IMCS_EVIDENCE_AUTO_FILL") or "0"
        gap_rounds_val = os.getenv("IMCS_EVIDENCE_GAP_ROUNDS") or _read_env_file("IMCS_EVIDENCE_GAP_ROUNDS") or "1"
        auto_fill = str(auto_fill_val).lower() in ("1", "true", "yes")
        max_gap_rounds = int(gap_rounds_val or "1")
        gap_rounds = 0
        logger.info(f"Evidence auto-fill: {auto_fill} (rounds={max_gap_rounds})")

        # ------- DAG execution loop -------
        while True:
            while pending:
                ready = [
                    s for s in pending.values()
                    if all(dep in completed for dep in (s.dependencies or []))
                ]

                if not ready:
                    for step in pending.values():
                        step.status = "blocked"
                        self.db.log_plan_step(
                            plan_id=plan.task_id,
                            step_id=step.step_id,
                            agent=step.agent,
                            action=step.action,
                            status="blocked",
                            error="Dependencies not met",
                            dependencies=step.dependencies,
                            params=step.params
                        )
                    plan.status = "blocked"
                    self.db.update_plan_status(plan.task_id, "blocked")
                    return plan.results

                for step in ready:
                    logger.info(f"Step {step.step_id}: [{step.agent}] {step.action}")
                    max_retries = getattr(step, "max_retries", 0) or 0
                    attempt = 0

                    while True:
                        attempt += 1
                        step.attempts = attempt
                        step.status = "running"

                        self.db.log_plan_step(
                            plan_id=plan.task_id,
                            step_id=step.step_id,
                            agent=step.agent,
                            action=step.action,
                            status="running",
                            dependencies=step.dependencies,
                            params=step.params
                        )

                        try:
                            result = self._execute_step(step)
                            self._raise_for_error_result(result)
                            step.result = result
                            step.status = "completed"
                            plan.results[step.step_id] = result

                            # Capture outputs for evidence aggregation
                            if step.agent == "literature" and isinstance(result, list):
                                if result and hasattr(result[0], "title"):
                                    literature_papers = result
                            if step.agent == "ml":
                                if isinstance(result, list):
                                    ml_top_models = result[:3]
                                elif isinstance(result, dict) and "top_3" in result:
                                    ml_top_models = result.get("top_3", [])
                                if isinstance(result, dict) and "predictions" in result:
                                    ml_predictions = result.get("predictions", {}) or {}
                                if isinstance(step.params, dict) and step.params.get("target_col"):
                                    ml_target = step.params.get("target_col")

                            self.db.log_plan_step(
                                plan_id=plan.task_id,
                                step_id=step.step_id,
                                agent=step.agent,
                                action=step.action,
                                status="completed",
                                result=result,
                                dependencies=step.dependencies,
                                params=step.params
                            )

                            completed.add(step.step_id)
                            pending.pop(step.step_id, None)

                            # Auto-append ML training after experiment data processed
                            try:
                                if step.agent == "experiment" and step.action == "process":
                                    if isinstance(result, dict) and result.get("processed", 0) > 0:
                                        self._append_activity_ml_step(
                                            plan, pending, depends_on=step.step_id,
                                        )
                            except Exception as e:
                                logger.warning(f"Auto ML append failed: {e}")
                            break

                        except Exception as e:
                            logger.error(f"Step {step.step_id} failed (attempt {attempt}): {e}")
                            step.error = str(e)

                            if attempt <= max_retries:
                                self.db.log_plan_step(
                                    plan_id=plan.task_id,
                                    step_id=step.step_id,
                                    agent=step.agent,
                                    action=step.action,
                                    status="retrying",
                                    error=str(e),
                                    dependencies=step.dependencies,
                                    params=step.params
                                )
                                continue

                            decision = None
                            try:
                                if self.failure_policy:
                                    decision = self.failure_policy.decide(step, e)
                            except Exception as policy_err:
                                logger.warning(f"Failure policy decide failed: {policy_err}")
                                decision = None

                            if decision and decision.action == "skip":
                                step.status = "skipped"
                                payload = {
                                    "note": decision.note or "Skipped by failure policy",
                                    "failure_policy": decision.to_dict(),
                                }
                                self.db.log_plan_step(
                                    plan_id=plan.task_id,
                                    step_id=step.step_id,
                                    agent=step.agent,
                                    action=step.action,
                                    status="skipped",
                                    error=str(e),
                                    result=payload,
                                    dependencies=step.dependencies,
                                    params=step.params,
                                )
                                try:
                                    plan.results.setdefault("warnings", []).append(
                                        {
                                            "step_id": step.step_id,
                                            "agent": step.agent,
                                            "action": step.action,
                                            "category": decision.category,
                                            "note": decision.note,
                                            "error": str(e),
                                        }
                                    )
                                except Exception:
                                    pass
                                completed.add(step.step_id)
                                pending.pop(step.step_id, None)
                                break

                            # Replan on failure (best-effort)
                            replanned = False
                            replan_budget = getattr(step, "max_replans", 0) or 0
                            if decision and decision.action == "replan":
                                policy_budget = decision.max_replans or 1
                                replan_budget = max(replan_budget, policy_budget)

                            if step.replan_attempts < replan_budget:
                                spec = decision.spec if (decision and isinstance(decision.spec, dict)) else self._build_replan_spec(step)
                                if spec:
                                    try:
                                        self.db.update_plan_status(plan.task_id, "replanning")
                                    except Exception:
                                        pass

                                    new_step_ids = self._apply_replan(plan, step, spec, pending)
                                    if new_step_ids:
                                        step.replan_attempts += 1
                                        step.status = "replanned"
                                        self.db.log_plan_step(
                                            plan_id=plan.task_id,
                                            step_id=step.step_id,
                                            agent=step.agent,
                                            action=step.action,
                                            status="replanned",
                                            error=str(e),
                                            result={
                                                "note": spec.get("note") or (decision.note if decision else None),
                                                "new_steps": new_step_ids,
                                                "failure_policy": decision.to_dict() if decision else None,
                                            },
                                            dependencies=step.dependencies,
                                            params=step.params
                                        )
                                        replanned = True
                                        pending.pop(step.step_id, None)
                                    try:
                                        self.db.update_plan_status(plan.task_id, "executing")
                                    except Exception:
                                        pass

                            if replanned:
                                break

                            step.status = "failed"
                            self.db.log_plan_step(
                                plan_id=plan.task_id,
                                step_id=step.step_id,
                                agent=step.agent,
                                action=step.action,
                                status="failed",
                                error=str(e),
                                result={"failure_policy": decision.to_dict()} if decision else None,
                                dependencies=step.dependencies,
                                params=step.params
                            )

                            plan.status = "failed"
                            self.db.update_plan_status(plan.task_id, "failed")
                            return plan.results

            if self.meta_controller and adaptive_rounds < self.max_adaptive_rounds:
                try:
                    existing = [
                        {"agent": s.agent, "action": s.action, "params": s.params, "deps": s.dependencies}
                        for s in plan.steps
                    ]
                    followups = self.meta_controller.suggest_followups(
                        plan.task_type, plan.description, existing
                    )
                    if followups:
                        self._append_dynamic_steps(plan, followups, pending)
                        adaptive_rounds += 1
                        continue
                except Exception as e:
                    logger.warning(f"Meta-controller follow-up failed: {e}")

            break

        # ------- Evidence aggregation (best-effort) -------
        rag_results = []
        candidate_ids_snapshot = []
        materials_snapshot = []
        before_stats = None
        ranking_current = None
        ranking_metric = None
        try:
            theory_agent = self.agents.get("theory")
            if theory_agent:
                plan_record = self.db.get_plan(plan.task_id)
                created_at = plan_record.get("created_at") if plan_record else None
                try:
                    from src.agents.core.theory_agent import TheoryDataConfig
                    allowed = TheoryDataConfig().elements
                except Exception:
                    allowed = None
                if created_at:
                    materials = self.db.list_materials_since(created_at, limit=50, allowed_elements=allowed)
                else:
                    materials = theory_agent.list_stored_materials(limit=20)
                if not materials:
                    materials = theory_agent.list_stored_materials(limit=20)

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
                    loaded = self.db.list_materials_by_ids(candidate_ids_snapshot, allowed_elements=allowed)
                    materials = loaded if loaded else [m for m in materials if m.get("material_id") in candidate_ids]
                    plan.results["candidate_material_ids"] = candidate_ids_snapshot
                else:
                    materials = [m for m in materials if m.get("material_id") in candidate_ids]
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

                def _formula_aliases(formula: str):
                    aliases = set()
                    if not formula:
                        return aliases
                    aliases.add(formula)
                    try:
                        from pymatgen.core import Composition
                        comp = Composition(formula)
                        aliases.add(comp.reduced_formula)
                        elements = sorted([el.symbol for el in comp.elements])
                        if len(elements) > 1:
                            aliases.add("-".join(elements))
                            aliases.add("".join(elements))
                    except Exception:
                        pass
                    return {a for a in aliases if a and len(a) > 1}

                for mat in materials:
                    mid = mat.get("material_id")
                    if not mid:
                        continue
                    formula = mat.get("formula", "")
                    self.db.save_evidence(
                        material_id=mid,
                        source_type="theory",
                        source_id=str(mid),
                        score=1.0,
                        metadata={
                            "formation_energy": mat.get("formation_energy"),
                            "formula": formula
                        }
                    )
                    if literature_papers and formula:
                        aliases = _formula_aliases(formula)
                        patterns = [
                            re.compile(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", re.IGNORECASE)
                            for a in aliases
                        ]
                        for paper in literature_papers[:5]:
                            hay = f"{getattr(paper, 'title', '')} {getattr(paper, 'abstract', '')}"
                            if any(p.search(hay) for p in patterns):
                                self.db.save_evidence(
                                    material_id=mid,
                                    source_type="literature",
                                    source_id=getattr(paper, "doi", "") or getattr(paper, "url", "") or "paper",
                                    score=0.8,
                                    metadata={
                                        "title": getattr(paper, "title", ""),
                                        "year": getattr(paper, "year", None),
                                        "abstract": getattr(paper, "abstract", ""),
                                        "doi": getattr(paper, "doi", ""),
                                        "url": getattr(paper, "url", ""),
                                        "authors": getattr(paper, "authors", []),
                                        "citation_count": getattr(paper, "citation_count", None)
                                    }
                                )
                                break
                    if ml_predictions and mid in ml_predictions:
                        self.db.save_evidence(
                            material_id=mid,
                            source_type="ml_prediction",
                            source_id="ml_prediction",
                            score=0.7,
                            metadata={"prediction": ml_predictions.get(mid)}
                        )

                # Knowledge RAG summary (top-5)
                try:
                    from src.services.knowledge import KnowledgeRAG, KnowledgeService
                    rag = KnowledgeRAG(self.db.db_path)
                    ks = KnowledgeService(self.db.db_path)
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
                    logger.warning(f"Knowledge RAG summary failed: {e}")

                # Evidence gap analysis
                try:
                    if self.meta_controller and materials_snapshot:
                        mat_ids = [m.get("material_id") for m in materials_snapshot if m.get("material_id")]
                        gap_report = self.meta_controller.analyze_evidence_gap(
                            mat_ids, plan.task_type, plan.description,
                        )
                        if gap_report:
                            plan.results["evidence_gap"] = gap_report
                            try:
                                from src.agents.core.theory_agent import TheoryDataConfig
                                allowed = TheoryDataConfig().elements
                            except Exception:
                                allowed = None
                            before_stats = self.db.get_evidence_stats(allowed_elements=allowed)
                            if not plan.results.get("evidence_stats_before_gap"):
                                plan.results["evidence_stats_before_gap"] = before_stats
                            gap_steps = gap_report.get("recommended_steps") or []
                            if gap_steps:
                                if auto_fill and gap_rounds < max_gap_rounds:
                                    if ranking_current and not plan.results.get("ranking_before_gap"):
                                        plan.results["ranking_before_gap"] = ranking_current
                                    gap_rounds += 1
                                    fill_result = self._execute_gap_steps(
                                        plan, gap_steps, ml_predictions=ml_predictions,
                                        literature_papers=literature_papers, ml_target=ml_target,
                                    )
                                    ml_predictions = fill_result.get("ml_predictions", ml_predictions)
                                    literature_papers = fill_result.get("literature_papers", literature_papers)
                                    ml_target = fill_result.get("ml_target", ml_target)
                                    if fill_result.get("success", False):
                                        recomputed = self._recompute_evidence_post_gap(
                                            plan, ml_predictions=ml_predictions, ml_target=ml_target,
                                        ) or {}
                                        if recomputed.get("rag_results"):
                                            rag_results = recomputed.get("rag_results", rag_results)
                                        if recomputed.get("candidate_ids_snapshot"):
                                            candidate_ids_snapshot = recomputed.get("candidate_ids_snapshot", candidate_ids_snapshot)
                                            plan.results["candidate_material_ids"] = candidate_ids_snapshot
                                        if recomputed.get("materials_snapshot"):
                                            materials_snapshot = recomputed.get("materials_snapshot", materials_snapshot)
                                        if recomputed.get("ranking_current"):
                                            ranking_current = recomputed.get("ranking_current", ranking_current)
                                            ranking_metric = recomputed.get("ranking_metric", ranking_metric)
                                    else:
                                        awaiting_confirmation = True
                                else:
                                    for item in gap_steps:
                                        step_id = self._next_step_id(plan)
                                        deps = self._resolve_gap_deps(plan, item.get("deps") or [])
                                        step = TaskStep(
                                            step_id=step_id,
                                            agent=item.get("agent", ""),
                                            action=item.get("action", ""),
                                            params=item.get("params") or {},
                                            dependencies=deps,
                                            status="pending",
                                        )
                                        plan.steps.append(step)
                                        self.db.log_plan_step(
                                            plan_id=plan.task_id,
                                            step_id=step.step_id,
                                            agent=step.agent,
                                            action=step.action,
                                            status="pending",
                                            dependencies=step.dependencies,
                                            params=step.params
                                        )
                                    awaiting_confirmation = True
                except Exception as e:
                    logger.warning(f"Evidence gap analysis failed: {e}")
        except Exception as e:
            logger.warning(f"Evidence aggregation failed: {e}")

        if rag_results:
            plan.results["knowledge_rag"] = rag_results
        if ranking_current:
            plan.results["ranking_current"] = ranking_current
            plan.results["ranking_metric"] = ranking_metric
            if awaiting_confirmation and not plan.results.get("ranking_before_gap"):
                plan.results["ranking_before_gap"] = ranking_current
            if not awaiting_confirmation and plan.results.get("ranking_before_gap"):
                plan.results["ranking_after_gap"] = ranking_current

        # Dataset snapshot + reasoning report
        try:
            if candidate_ids_snapshot:
                snap_id = self.db.create_dataset_snapshot(
                    plan_id=plan.task_id,
                    name=f"snapshot_{plan.task_id}",
                    description="Auto snapshot of candidate materials for reproducibility",
                    metadata={"count": len(candidate_ids_snapshot)}
                )
                if snap_id:
                    for mid in candidate_ids_snapshot:
                        self.db.add_snapshot_item(
                            snapshot_id=snap_id,
                            item_type="material",
                            item_id=mid,
                            metadata=None
                        )
                    plan.results["dataset_snapshot_id"] = snap_id

            from src.services.knowledge import KnowledgeService
            ks = KnowledgeService(self.db.db_path)
            report_items = []
            for mat in materials_snapshot[:5]:
                mid = mat.get("material_id")
                if not mid:
                    continue
                ev = self.db.get_evidence_for_material(mid)
                ev_counts = {}
                for e in ev:
                    stype = e.get("source_type") or "unknown"
                    ev_counts[stype] = ev_counts.get(stype, 0) + 1
                ent = ks.get_entity_by_canonical("material", mid) or ks.get_entity_by_name("material", mid)
                trace = ks.trace_entity(ent["id"], depth=1) if ent else {}
                quality = ks.score_material(mid)
                report_items.append({
                    "material_id": mid,
                    "evidence_counts": ev_counts,
                    "trace": trace,
                    "knowledge_score": quality.get("score"),
                    "knowledge_score_detail": quality.get("counts", {})
                })
            if report_items:
                plan.results["reasoning_report"] = report_items
        except Exception as e:
            logger.warning(f"Reasoning report failed: {e}")

        # Evaluation metrics
        try:
            from src.services.task.evaluator import PlanEvaluator
            evaluator = PlanEvaluator(self.db)
            eval_metrics = evaluator.evaluate(plan)
            if eval_metrics:
                plan.results["evaluation_metrics"] = eval_metrics
        except Exception as e:
            logger.warning(f"Evaluation metrics failed: {e}")

        plan.results.setdefault("knowledge_rag", [])
        plan.results.setdefault("reasoning_report", [])

        # Persist knowledge pack
        try:
            if (
                plan.results.get("evidence_gap")
                or plan.results.get("evidence_stats_before_gap")
                or plan.results.get("ranking_current")
                or plan.results.get("ranking_before_gap")
                or plan.results.get("ranking_after_gap")
                or plan.results.get("ranking_metric")
                or plan.results.get("candidate_material_ids")
                or plan.results.get("evaluation_metrics")
            ):
                out_dir = os.path.join("data", "tasks")
                out_path = os.path.join(out_dir, f"knowledge_{plan.task_id}.json")
                if os.path.exists(out_path):
                    with open(out_path, "r", encoding="utf-8") as f:
                        pack = json.load(f)
                    pack["evidence_gap"] = plan.results.get("evidence_gap")
                    if plan.results.get("evidence_stats_before_gap"):
                        pack["evidence_stats_before_gap"] = plan.results.get("evidence_stats_before_gap")
                    if plan.results.get("evidence_stats_after_gap"):
                        pack["evidence_stats_after_gap"] = plan.results.get("evidence_stats_after_gap")
                    if plan.results.get("evidence_stats_delta"):
                        pack["evidence_stats_delta"] = plan.results.get("evidence_stats_delta")
                    if plan.results.get("candidate_material_ids"):
                        pack["candidate_material_ids"] = plan.results.get("candidate_material_ids")
                    if plan.results.get("ranking_before_gap"):
                        pack["ranking_before_gap"] = plan.results.get("ranking_before_gap")
                    if plan.results.get("ranking_after_gap"):
                        pack["ranking_after_gap"] = plan.results.get("ranking_after_gap")
                    if plan.results.get("ranking_current"):
                        pack["ranking_current"] = plan.results.get("ranking_current")
                    if plan.results.get("ranking_metric"):
                        pack["ranking_metric"] = plan.results.get("ranking_metric")
                    if plan.results.get("evaluation_metrics"):
                        pack["evaluation_metrics"] = plan.results.get("evaluation_metrics")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(pack, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update knowledge pack with evidence gap: {e}")

        # Re-evaluate pending state
        try:
            if any(step.status in ("pending", "running") for step in plan.steps):
                awaiting_confirmation = True
            else:
                awaiting_confirmation = False
        except Exception:
            pass

        # Stats delta
        try:
            if not awaiting_confirmation and plan.results.get("evidence_stats_before_gap"):
                try:
                    from src.agents.core.theory_agent import TheoryDataConfig
                    allowed = TheoryDataConfig().elements
                except Exception:
                    allowed = None
                after_stats = self.db.get_evidence_stats(allowed_elements=allowed)
                plan.results["evidence_stats_after_gap"] = after_stats
                delta = {}
                before = plan.results.get("evidence_stats_before_gap") or {}
                for key, before_val in before.items():
                    after_val = after_stats.get(key) if isinstance(after_stats, dict) else None
                    if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                        delta[key] = after_val - before_val
                plan.results["evidence_stats_delta"] = delta
        except Exception as e:
            logger.warning(f"Evidence stats delta failed: {e}")

        # Update knowledge pack with after-gap stats
        try:
            if plan.results.get("evidence_stats_after_gap") or plan.results.get("evidence_stats_delta"):
                out_dir = os.path.join("data", "tasks")
                out_path = os.path.join(out_dir, f"knowledge_{plan.task_id}.json")
                if os.path.exists(out_path):
                    with open(out_path, "r", encoding="utf-8") as f:
                        pack = json.load(f)
                    if plan.results.get("evidence_stats_after_gap"):
                        pack["evidence_stats_after_gap"] = plan.results.get("evidence_stats_after_gap")
                    if plan.results.get("evidence_stats_delta"):
                        pack["evidence_stats_delta"] = plan.results.get("evidence_stats_delta")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(pack, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update knowledge pack after gap stats: {e}")

        # Strategy feedback
        try:
            from src.services.task.strategy_tracker import StrategyTracker
            tracker = StrategyTracker()
            tracker.update_from_plan(plan)
        except Exception as e:
            logger.warning(f"Strategy tracker failed: {e}")

        if awaiting_confirmation:
            plan.status = "awaiting_confirmation"
            self.db.update_plan_status(plan.task_id, "awaiting_confirmation")
            self._active_plan = None
            return plan.results

        plan.status = "completed"
        self.db.update_plan_status(plan.task_id, "completed")
        self._active_plan = None
        return plan.results
