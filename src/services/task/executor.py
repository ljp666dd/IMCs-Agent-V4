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
        self.max_adaptive_rounds = 1
    
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

    def _build_replan_spec(self, step: TaskStep) -> Optional[Dict[str, Any]]:
        """Return a minimal fallback plan when a step fails."""
        agent = step.agent
        action = step.action

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
                dependencies=step.dependencies
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
                dependencies=step.dependencies
            )

            step_id_map[step.agent] = step.step_id

        return new_step_ids

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
        pending = {s.step_id: s for s in plan.steps}
        completed = set()
        literature_papers = []
        ml_top_models = []
        ml_predictions = {}

        adaptive_rounds = 0
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
                            dependencies=step.dependencies
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
                            dependencies=step.dependencies
                        )

                        try:
                            result = self._execute_step(step)
                            step.result = result
                            step.status = "completed"
                            plan.results[step.step_id] = result

                            # Capture outputs for evidence aggregation
                            if step.agent == "literature" and isinstance(result, list):
                                literature_papers = result
                            if step.agent == "ml":
                                if isinstance(result, list):
                                    ml_top_models = result[:3]
                                elif isinstance(result, dict) and "top_3" in result:
                                    ml_top_models = result.get("top_3", [])
                                if isinstance(result, dict) and "predictions" in result:
                                    ml_predictions = result.get("predictions", {}) or {}

                            # 3. Log Success
                            self.db.log_plan_step(
                                plan_id=plan.task_id,
                                step_id=step.step_id,
                                agent=step.agent,
                                action=step.action,
                                status="completed",
                                result=result,
                                dependencies=step.dependencies
                            )

                            completed.add(step.step_id)
                            pending.pop(step.step_id, None)
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
                                    dependencies=step.dependencies
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
                                            dependencies=step.dependencies
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
                                dependencies=step.dependencies
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

        plan.status = "completed"
        self.db.update_plan_status(plan.task_id, "completed")

        # Evidence aggregation (best-effort)
        # 证据链聚合: 将理论/文献/ML 结果挂接到材料实体
        rag_results = []
        candidate_ids_snapshot = []
        materials_snapshot = []
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
        except Exception as e:
            logger.warning(f"Evidence aggregation failed: {e}")

        if rag_results:
            plan.results["knowledge_rag"] = rag_results

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

        # Ensure key outputs exist for downstream UI/reporting
        plan.results.setdefault("knowledge_rag", [])
        plan.results.setdefault("reasoning_report", [])
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
                    mat_ids = [m.get("material_id") for m in mats if m.get("material_id")]
                    if mat_ids:
                        target_agent.download_orbital_dos(material_ids=mat_ids)
                if "adsorption" in data_types:
                    target_agent.download_adsorption_energies(adsorbates=["H*", "OH*"], limit=limit)
                return {"message": "Downloaded theory data to DB", "data_types": data_types}
        
        elif agent_name == "experiment":
             if action == "process":
                 return {"message": "Scanning for experiment data..."}
                 
        elif agent_name == "ml":
            # AUTO-LOAD DATA from DB before training
            target_agent.load_from_db() 
            
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
            elif action == "summarize":
                 return {"summary": "Task completed."}
                 
        return {"status": "executed", "action": action}
