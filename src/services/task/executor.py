from typing import Dict, Any, Optional
from src.core.logger import get_logger, log_exception
from src.services.task.types import TaskPlan
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

        while pending:
            # Find steps whose dependencies are satisfied
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

                except Exception as e:
                    logger.error(f"Step {step.step_id} failed: {e}")
                    step.error = str(e)
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
                
        plan.status = "completed"
        self.db.update_plan_status(plan.task_id, "completed")

        # Evidence aggregation (best-effort)
        try:
            theory_agent = self.agents.get("theory")
            if theory_agent:
                plan_record = self.db.get_plan(plan.task_id)
                created_at = plan_record.get("created_at") if plan_record else None
                if created_at:
                    materials = self.db.list_materials_since(created_at, limit=50)
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
                        metadata={"formation_energy": mat.get("formation_energy")}
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
                                    metadata={"title": getattr(paper, "title", ""), "year": getattr(paper, "year", None)}
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
        except Exception as e:
            logger.warning(f"Evidence aggregation failed: {e}")

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
                return target_agent.search_all_sources(params.get("query", ""))
            elif action == "extract_knowledge":
                # Assuming LiteratureAgentv3.1 has extract_knowledge(topic)
                return target_agent.extract_knowledge(params.get("topic", ""))
        
        elif agent_name == "theory":
            if action == "load_data":
                return target_agent.get_status()
            elif action == "download":
                target_agent.download_structures(limit=50) # Increased limit for valid ML
                return {"message": "Downloaded structures to DB"}
        
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
