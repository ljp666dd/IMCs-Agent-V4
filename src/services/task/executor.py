import json
import os
from typing import Dict, Any, Optional, List
from src.core.logger import get_logger, log_exception
from src.services.task.types import TaskPlan, TaskStep
from src.services.db.database import DatabaseService

logger = get_logger(__name__)

class PlanExecutor:
    """
    Service for executing task plans by dispatching to agents.
    Includes persistence to SQLite (v4.0).
    """
    
    def __init__(self, agents: Dict[str, Any]):
        """
        Args:
            agents: Dictionary mapping agent names to agent instances.
        """
        self.agents = agents
        self.db = DatabaseService()
        try:
            from src.services.task.meta_controller import MetaController
            self.meta_controller = MetaController(self.db)
        except Exception:
            self.meta_controller = None
        self.replan_strategies = self._load_replan_strategies()
        self.max_adaptive_rounds = 1
        self._active_plan: Optional[TaskPlan] = None
    
    def _next_step_id(self, plan: TaskPlan) -> str:
        """Allocate a unique step_id within a plan."""
        max_idx = 0
        for step in plan.steps:
            sid = getattr(step, "step_id", "")
            if isinstance(sid, str) and sid.startswith("step_"):
                try:
                    max_idx = max(max_idx, int(sid.split("_", 1)[1]))
                except Exception:
                    continue
        return f"step_{max_idx + 1}"

    def _simplify_query(self, query: str) -> str:
        """Best-effort query simplification for fallback searches."""
        if not query:
            return ""
        import re
        tokens = re.findall(r"[A-Za-z0-9\\-\\+]+", query)
        if not tokens:
            return query.strip()
        return " ".join(tokens[:8]).strip()

    def _repo_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    def _load_replan_strategies(self) -> Dict[str, Any]:
        """Load replan/fallback strategies from configs/replan_strategies.json."""
        try:
            path = os.path.join(self._repo_root(), "configs", "replan_strategies.json")
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _format_params(self, params: Any, query: str) -> Any:
        if isinstance(params, dict):
            return {k: self._format_params(v, query) for k, v in params.items()}
        if isinstance(params, list):
            return [self._format_params(v, query) for v in params]
        if isinstance(params, str):
            return params.replace("{query}", query or "")
        return params

    def _replan_from_strategy(self, step: TaskStep) -> Optional[Dict[str, Any]]:
        if not self.replan_strategies:
            return None
        group = self.replan_strategies.get("default") or {}
        key = f"{step.agent}.{step.action}"
        spec = group.get(key)
        if not isinstance(spec, dict):
            return None
        try:
            query = None
            if isinstance(step.params, dict):
                query = step.params.get("query")
            if not query and self._active_plan:
                query = self._active_plan.description
            spec = json.loads(json.dumps(spec))
            spec["steps"] = [
                {
                    **item,
                    "params": self._format_params(item.get("params") or {}, query),
                }
                for item in (spec.get("steps") or [])
            ]
            return spec
        except Exception:
            return spec

    def _build_replan_spec(self, step: TaskStep) -> Optional[Dict[str, Any]]:
        """Return a minimal fallback plan when a step fails."""
        agent = step.agent
        action = step.action
        strategy_spec = self._replan_from_strategy(step)
        if strategy_spec:
            return strategy_spec

        if agent == "literature" and action == "search":
            query = (step.params or {}).get("query", "")
            simplified = self._simplify_query(query)
            if simplified and simplified.lower() != query.lower():
                fallback_query = simplified
            elif query:
                fallback_query = f"{query} catalyst"
            else:
                fallback_query = "HOR catalyst"
            return {
                "note": "fallback: simplified literature query",
                "steps": [
                    {
                        "agent": "literature",
                        "action": "search",
                        "params": {"query": fallback_query, "limit": 5},
                        "deps": []
                    }
                ]
            }

        if agent == "theory" and action == "download":
            data_types = (step.params or {}).get("data_types") or []
            fallback_types = ["cif"]
            if "formation_energy" in data_types:
                fallback_types = ["cif", "formation_energy"]
            return {
                "note": "fallback: reduced theory download scope",
                "steps": [
                    {
                        "agent": "theory",
                        "action": "download",
                        "params": {"data_types": fallback_types, "limit": 20},
                        "deps": []
                    }
                ]
            }

        if agent == "ml" and action in ("train", "train_all"):
            return {
                "note": "fallback: refresh theory data then retrain",
                "steps": [
                    {
                        "agent": "theory",
                        "action": "download",
                        "params": {"data_types": ["cif", "formation_energy"], "limit": 50},
                        "deps": []
                    },
                    {
                        "agent": "ml",
                        "action": action,
                        "params": step.params or {},
                        "deps": ["$prev"]
                    }
                ]
            }

        return None

    def _apply_replan(self, plan: TaskPlan, failed_step: TaskStep, spec: Dict[str, Any],
                      pending: Dict[str, TaskStep]) -> List[str]:
        """Insert replan steps and rewire dependencies. Returns new step_ids."""
        new_step_ids: List[str] = []
        prev_id: Optional[str] = None

        for item in spec.get("steps", []):
            step_id = self._next_step_id(plan)
            deps = []
            for dep in item.get("deps", []):
                if dep == "$prev":
                    if prev_id:
                        deps.append(prev_id)
                else:
                    deps.append(dep)

            step = TaskStep(
                step_id=step_id,
                agent=item.get("agent", ""),
                action=item.get("action", ""),
                params=item.get("params") or {},
                dependencies=deps,
                max_retries=item.get("max_retries", 0),
                max_replans=0
            )
            plan.steps.append(step)
            pending[step.step_id] = step
            new_step_ids.append(step.step_id)
            prev_id = step.step_id

            # Persist the new pending step
            self.db.log_plan_step(
                plan_id=plan.task_id,
                step_id=step.step_id,
                agent=step.agent,
                action=step.action,
                status="pending",
                dependencies=step.dependencies,
                params=step.params
            )

        if not new_step_ids:
            return new_step_ids

        anchor_id = new_step_ids[-1]
        for step in pending.values():
            deps = step.dependencies or []
            if failed_step.step_id in deps:
                step.dependencies = [anchor_id if d == failed_step.step_id else d for d in deps]
        return new_step_ids

    def _append_dynamic_steps(self, plan: TaskPlan, specs: List[Dict[str, Any]],
                               pending: Dict[str, TaskStep]) -> List[str]:
        """Append meta-controller suggested steps to the plan."""
        new_step_ids: List[str] = []
        step_id_map = {}

        for step in plan.steps:
            if step.agent:
                step_id_map[step.agent] = step.step_id

        for item in specs:
            step_id = self._next_step_id(plan)
            deps = []
            for dep in item.get("deps", []):
                if isinstance(dep, str) and dep.startswith("$"):
                    key = dep[1:]
                    if key in step_id_map:
                        deps.append(step_id_map[key])
                else:
                    deps.append(dep)

            step = TaskStep(
                step_id=step_id,
                agent=item.get("agent", ""),
                action=item.get("action", ""),
                params=item.get("params") or {},
                dependencies=deps,
                max_retries=item.get("max_retries", 0),
                max_replans=0
            )
            plan.steps.append(step)
            pending[step.step_id] = step
            new_step_ids.append(step.step_id)

            self.db.log_plan_step(
                plan_id=plan.task_id,
                step_id=step.step_id,
                agent=step.agent,
                action=step.action,
                status="pending",
                dependencies=step.dependencies,
                params=step.params
            )

            step_id_map[step.agent] = step.step_id

        return new_step_ids

    def _append_activity_ml_step(self, plan: TaskPlan, pending: Dict[str, TaskStep],
                                 depends_on: str, metric_name: str = "exchange_current_density") -> Optional[str]:
        """Append an ML step to train on activity metrics if not already present."""
        for step in plan.steps:
            if step.agent == "ml" and step.action in ("train", "train_all"):
                params = step.params or {}
                target_col = params.get("target_col") if isinstance(params, dict) else None
                if isinstance(target_col, str) and target_col.startswith("activity_metric:"):
                    return None

        step_id = self._next_step_id(plan)
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
        self.db.log_plan_step(
            plan_id=plan.task_id,
            step_id=step.step_id,
            agent=step.agent,
            action=step.action,
            status="pending",
            dependencies=step.dependencies,
            params=step.params,
        )
        return step.step_id

    def _execute_gap_steps(
        self,
        plan: TaskPlan,
        gap_steps: List[Dict[str, Any]],
        ml_predictions: Dict[str, Any],
        literature_papers: List[Any],
        ml_target: Optional[str],
    ) -> Dict[str, Any]:
        """Execute evidence gap steps sequentially (best-effort)."""
        success = True
        for item in gap_steps:
            step_id = self._next_step_id(plan)
            step = TaskStep(
                step_id=step_id,
                agent=item.get("agent", ""),
                action=item.get("action", ""),
                params=item.get("params") or {},
                dependencies=item.get("deps") or [],
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
                params=step.params,
            )

            try:
                self.db.log_plan_step(
                    plan_id=plan.task_id,
                    step_id=step.step_id,
                    agent=step.agent,
                    action=step.action,
                    status="running",
                    dependencies=step.dependencies,
                    params=step.params,
                )
                result = self._execute_step(step)
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

                self.db.log_plan_step(
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
                self.db.log_plan_step(
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

    def _recompute_evidence_post_gap(
        self,
        plan: TaskPlan,
        ml_predictions: Dict[str, Any],
        ml_target: Optional[str],
    ) -> Dict[str, Any]:
        """Recompute ranking + knowledge RAG after gap auto-fill."""
        rag_results: List[Dict[str, Any]] = []
        candidate_ids_snapshot: List[str] = []
        materials_snapshot: List[Dict[str, Any]] = []
        ranking_current = None
        ranking_metric = None

        try:
            theory_agent = self.agents.get("theory")
            if not theory_agent:
                return {}
            plan_record = self.db.get_plan(plan.task_id)
            created_at = plan_record.get("created_at") if plan_record else None
            if created_at:
                try:
                    from src.agents.core.theory_agent import TheoryDataConfig
                    allowed = TheoryDataConfig().elements
                except Exception:
                    allowed = None
                materials = self.db.list_materials_since(created_at, limit=50, allowed_elements=allowed)
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
                        ml_predictions.items(),
                        key=lambda kv: kv[1],
                        reverse=True
                    )
                    candidate_ids = {mid for mid, _ in sorted_preds[:top_n]}
                except Exception:
                    candidate_ids = set(ml_predictions.keys())
            else:
                candidate_ids = {m.get("material_id") for m in materials if m.get("material_id")}

            materials = [m for m in materials if m.get("material_id") in candidate_ids]
            candidate_ids_snapshot = list(candidate_ids)
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

    def _merge_knowledge_pack_results(self, plan: TaskPlan) -> None:
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

    @log_exception(logger)
    def execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """Execute all steps in the plan (DAG aware + Persisted)."""
        if not plan:
            return {"error": "No plan provided"}
            
        # 1. Create Plan Record (M2 Persistence)
        self.db.create_plan(
            plan_id=plan.task_id,
            user_id=None, # Context aware executor triggers this? Need to pass user_id later
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
        while True:
            while pending:
                # Find steps whose dependencies are satisfied
                # DAG 调度: 仅当依赖已完成时才进入就绪队列
                ready = [
                    s for s in pending.values()
                    if all(dep in completed for dep in (s.dependencies or []))
                ]

                if not ready:
                    # Deadlock or unmet deps -> mark remaining as blocked
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

                        # 2. Log Start
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

                            # 3. Log Success
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
                                            plan,
                                            pending,
                                            depends_on=step.step_id,
                                        )
                            except Exception as e:
                                logger.warning(f"Auto ML append failed: {e}")
                            break

                        except Exception as e:
                            logger.error(f"Step {step.step_id} failed (attempt {attempt}): {e}")
                            step.error = str(e)

                            if attempt <= max_retries:
                                # Retry
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

                            # Replan on failure (best-effort)
                            replanned = False
                            if step.replan_attempts < step.max_replans:
                                spec = self._build_replan_spec(step)
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
                                            result={"note": spec.get("note"), "new_steps": new_step_ids},
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

                            # 4. Log Failure
                            self.db.log_plan_step(
                                plan_id=plan.task_id,
                                step_id=step.step_id,
                                agent=step.agent,
                                action=step.action,
                                status="failed",
                                error=str(e),
                                dependencies=step.dependencies,
                                params=step.params
                            )

                            # Stop execution on failure
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

        # Evidence aggregation (best-effort)
        # 证据链聚合: 将理论/文献/ML 结果挂接到材料实体
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
                if created_at:
                    try:
                        from src.agents.core.theory_agent import TheoryDataConfig
                        allowed = TheoryDataConfig().elements
                    except Exception:
                        allowed = None
                    materials = self.db.list_materials_since(created_at, limit=50, allowed_elements=allowed)
                else:
                    materials = theory_agent.list_stored_materials(limit=20)
                if not materials:
                    materials = theory_agent.list_stored_materials(limit=20)

                # Candidate filter (prefer ML predictions top-N)
                candidate_ids = set()
                top_n = 10
                sorted_preds = None
                if ml_predictions:
                    try:
                        sorted_preds = sorted(
                            ml_predictions.items(),
                            key=lambda kv: kv[1],
                            reverse=True
                        )
                        candidate_ids = {mid for mid, _ in sorted_preds[:top_n]}
                    except Exception:
                        candidate_ids = set(ml_predictions.keys())
                else:
                    candidate_ids = {m.get("material_id") for m in materials if m.get("material_id")}

                materials = [m for m in materials if m.get("material_id") in candidate_ids]
                candidate_ids_snapshot = list(candidate_ids)
                materials_snapshot = materials[:]
                if candidate_ids_snapshot:
                    plan.results["candidate_material_ids"] = candidate_ids_snapshot
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
                    # Theory evidence
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
                    # Literature evidence
                    if literature_papers and formula:
                        import re
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
                    # ML evidence (only if per-material predictions exist)
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

                # Evidence gap analysis (post aggregation)
                try:
                    if self.meta_controller and materials_snapshot:
                        mat_ids = [m.get("material_id") for m in materials_snapshot if m.get("material_id")]
                        gap_report = self.meta_controller.analyze_evidence_gap(
                            mat_ids,
                            plan.task_type,
                            plan.description,
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
                                        plan,
                                        gap_steps,
                                        ml_predictions=ml_predictions,
                                        literature_papers=literature_papers,
                                        ml_target=ml_target,
                                    )
                                    ml_predictions = fill_result.get("ml_predictions", ml_predictions)
                                    literature_papers = fill_result.get("literature_papers", literature_papers)
                                    ml_target = fill_result.get("ml_target", ml_target)
                                    if fill_result.get("success", False):
                                        recomputed = self._recompute_evidence_post_gap(
                                            plan,
                                            ml_predictions=ml_predictions,
                                            ml_target=ml_target,
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
                                        deps = item.get("deps") or []
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

        # Dataset snapshot + reasoning report (best-effort)
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

            # Reasoning report: evidence counts + knowledge trace
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

        # Evaluation metrics (P1)
        try:
            from src.services.task.evaluator import PlanEvaluator
            evaluator = PlanEvaluator(self.db)
            eval_metrics = evaluator.evaluate(plan)
            if eval_metrics:
                plan.results["evaluation_metrics"] = eval_metrics
        except Exception as e:
            logger.warning(f"Evaluation metrics failed: {e}")

        # Ensure key outputs exist for downstream UI/reporting
        plan.results.setdefault("knowledge_rag", [])
        plan.results.setdefault("reasoning_report", [])
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
        # Re-evaluate pending state based on step statuses (post-confirmation runs)
        try:
            if any(step.status in ("pending", "running") for step in plan.steps):
                awaiting_confirmation = True
            else:
                awaiting_confirmation = False
        except Exception:
            pass

        # After gap fill, compute stats delta if available
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

        # Update knowledge pack with after-gap stats if available
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

        # Strategy feedback (P2) - run after stats delta is available
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

    def _execute_step(self, step) -> Any:
        """Dispatch single step (TaskStep object)."""
        agent_name = step.agent
        action = step.action
        params = step.params
        
        if agent_name not in self.agents and agent_name != "task_manager":
            return {"error": f"Unknown agent: {agent_name}"}
            
        target_agent = self.agents.get(agent_name)
        
        # Dispatch Logic (Hardcoded per Phase 4 scope to match original TaskManager)
        # Ideally this would be dynamic, but adhering to original logic for safety.
        
        if agent_name == "literature":
            if action == "search":
                return target_agent.search_all_sources(params.get("query", ""), params.get("limit"))
            elif action == "extract_knowledge":
                # Assuming LiteratureAgentv3.1 has extract_knowledge(topic)
                return target_agent.extract_knowledge(params.get("topic", ""))
            elif action == "harvest_hor_seed":
                return target_agent.harvest_hor_seed(
                    query=params.get("query", ""),
                    limit=params.get("limit", 10),
                    max_pdfs=params.get("max_pdfs", 5),
                    min_elements=params.get("min_elements", 2),
                    persist=bool(params.get("persist", False)),
                )
            elif action == "ingest_local_library":
                return target_agent.ingest_local_library(
                    min_elements=params.get("min_elements", 2)
                )
        
        elif agent_name == "theory":
            if action == "load_data":
                return target_agent.get_status()
            elif action == "download":
                data_types = params.get("data_types") or []
                limit = params.get("limit", 50)
                # 默认下载结构, 再按需补充 formation_energy / dos / adsorption
                if not data_types or "cif" in data_types:
                    target_agent.download_structures(limit=limit) # Increased limit for valid ML
                if "formation_energy" in data_types:
                    target_agent.download_formation_energy()
                if "dos" in data_types:
                    mats = target_agent.list_stored_materials(limit=20)
                    mat_ids = [
                        m.get("material_id")
                        for m in mats
                        if m.get("material_id") and str(m.get("material_id")).startswith("mp-")
                    ]
                    if mat_ids:
                        target_agent.download_orbital_dos(material_ids=mat_ids)
                if "adsorption" in data_types:
                    try:
                        target_agent.download_adsorption_energies(adsorbates=["H*", "OH*"], limit=limit)
                    except TypeError:
                        target_agent.download_adsorption_energies()
                return {"message": "Downloaded theory data to DB", "data_types": data_types}
        
        elif agent_name == "experiment":
             if action == "process":
                 data_dir = params.get("data_dir") if isinstance(params, dict) else None
                 reference_potential = params.get("reference_potential", 0.2) if isinstance(params, dict) else 0.2
                 loading_mg_cm2 = params.get("loading_mg_cm2", 0.25) if isinstance(params, dict) else 0.25
                 precious_fraction = params.get("precious_fraction", 0.20) if isinstance(params, dict) else 0.20
                 return target_agent.process_rde_directory(
                     data_dir=data_dir or "data/experimental/rde_lsv",
                     reference_potential=reference_potential,
                     loading_mg_cm2=loading_mg_cm2,
                     precious_fraction=precious_fraction,
                 )
                 
        elif agent_name == "ml":
            # AUTO-LOAD DATA from DB before training
            target_col = params.get("target_col") if isinstance(params, dict) else None
            if isinstance(target_col, str) and target_col.startswith("activity_metric:"):
                metric = target_col.split(":", 1)[1] or "exchange_current_density"
                target_agent.load_activity_metrics_from_db(metric)
            else:
                target_agent.load_from_db(target_col=target_col or "formation_energy")
            
            if action == "train":
                results = target_agent.train_traditional_models()
                # Serialize for API safely (remove raw model objects)
                models = [
                    {
                        "name": r.name,
                        "r2_test": r.r2_test,
                        "rmse_test": r.rmse_test
                    } for r in results
                ]
                pred_map = target_agent.predict_best()
                return {"models": models, "predictions": pred_map}
            elif action == "train_all":
                # Sequence
                r1 = target_agent.train_traditional_models()
                # Assuming deep learning might be added here in future
                
                # Get Top 3
                top_models = target_agent.get_top_models(k=3)
                top_summary = [f"{m.name} (R2={m.r2_test:.3f})" for m in top_models]
                pred_map = target_agent.predict_best()
                return {"trained": len(r1), "top_3": top_summary, "predictions": pred_map}
            elif action == "shap_analysis":
                return {"message": "SHAP analysis ready"}
            elif action == "predict":
                return {"message": "Prediction ready"}
                
        elif agent_name == "task_manager":
            # Self-referential actions usually handled by UI or simple summaries
            if action == "recommend":
                # Synthesize Report from Agent States
                lit_agent = self.agents.get("literature")
                ml_agent = self.agents.get("ml")
                theory_agent = self.agents.get("theory")
                
                report = []
                report.append("### 🔬 Research Synthesis")
                
                # 1. Literature Insights
                if lit_agent and hasattr(lit_agent, 'papers') and lit_agent.papers:
                    titles = [p.title for p in lit_agent.papers[:3]]
                    report.append(f"**📚 Literature**: Analyzed {len(lit_agent.papers)} papers. Key sources:\n" + "\n".join([f"- *{t}*" for t in titles]))
                
                # 2. Theory Data status
                if theory_agent:
                    # We can't easily get count without querying DB or keeping state. 
                    # But status check is cheap.
                    status = theory_agent.get_status()
                    cif_count = status.get("cif_files", 0)
                    report.append(f"**💎 Theory Data**: Integrated {cif_count} crystal structures from Materials Project.")

                # 3. ML Model Performance
                if ml_agent:
                    top = ml_agent.get_top_models(k=3)
                    if top:
                        best = top[0]
                        report.append(f"**🤖 AI Models**: Trained {len(ml_agent.results)} candidates. Best Performer:\n" + \
                                      f"- **{best.name}** (R²={best.r2_test:.3f})")
                        if best.r2_test > 0.7:
                            report.append(f"> ✅ High confidence model detected. Ready for prediction tasks.")
                        else:
                            report.append(f"> ⚠️ Model performance needs improvement (R²<0.7). Suggest adding more data.")

                # 4. Final Recommendation
                report.append("\n### 💡 Final Recommendation")
                report.append("1. **Experiment**: Proceed with synthesis of materials identified in the downloaded CIF set.")
                report.append("2. **Analysis**: Use the trained High-Performance Model to screen new candidates.")
                report.append("3. **Next Step**: Upload experimental characterization data to `/experiments` for validation.")
                
                return {"recommendation": "\n\n".join(report)}
            elif action == "knowledge_pack":
                try:
                    plan = self._active_plan
                    try:
                        from src.agents.core.theory_agent import TheoryDataConfig
                        allowed = TheoryDataConfig().elements
                    except Exception:
                        allowed = None
                    pack = {
                        "task_id": plan.task_id if plan else None,
                        "query": plan.description if plan else None,
                        "task_type": plan.task_type.value if plan else None,
                        "evidence_stats": self.db.get_evidence_stats(allowed_elements=allowed),
                        "knowledge_rag": (plan.results.get("knowledge_rag") if plan else []),
                        "reasoning_report": (plan.results.get("reasoning_report") if plan else []),
                        "evidence_gap": (plan.results.get("evidence_gap") if plan else None),
                        "candidate_material_ids": (plan.results.get("candidate_material_ids") if plan else []),
                        "ranking_before_gap": (plan.results.get("ranking_before_gap") if plan else []),
                        "ranking_after_gap": (plan.results.get("ranking_after_gap") if plan else []),
                        "ranking_current": (plan.results.get("ranking_current") if plan else []),
                        "ranking_metric": (plan.results.get("ranking_metric") if plan else None),
                        "evaluation_metrics": (plan.results.get("evaluation_metrics") if plan else None),
                    }
                    out_dir = os.path.join("data", "tasks")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"knowledge_{plan.task_id if plan else 'unknown'}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(pack, f, ensure_ascii=False, indent=2)
                    return {"knowledge_pack": pack, "path": out_path}
                except Exception as e:
                    return {"error": f"knowledge_pack failed: {e}"}
            elif action == "summarize":
                 return {"summary": "Task completed."}
                 
        return {"status": "executed", "action": action}
