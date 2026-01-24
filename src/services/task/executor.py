from typing import Dict, Any, Optional
from src.core.logger import get_logger, log_exception
from src.services.task.types import TaskPlan

logger = get_logger(__name__)

class PlanExecutor:
    """
    Service for executing task plans by dispatching to agents.
    """
    
    def __init__(self, agents: Dict[str, Any]):
        """
        Args:
            agents: Dictionary mapping agent names to agent instances.
                    e.g. {"ml": ml_agent, "theory": theory_agent}
        """
        self.agents = agents

    @log_exception(logger)
    def execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """Execute all steps in the plan."""
        if not plan:
            return {"error": "No plan provided"}
            
        plan.status = "executing"
        results = {}
        logger.info(f"Executing Plan: {plan.task_id}")
        
        for i, step in enumerate(plan.steps, 1):
            agent_name = step.get("agent")
            action = step.get("action")
            logger.info(f"Step {i}: [{agent_name}] {action}")
            
            try:
                result = self._execute_step(step)
                results[f"step_{i}"] = result
                plan.results[f"step_{i}"] = result # Update plan state logic
            except Exception as e:
                logger.error(f"Step {i} failed: {e}")
                results[f"step_{i}"] = {"error": str(e)}
                
        plan.status = "completed"
        return results

    def _execute_step(self, step: Dict[str, Any]) -> Any:
        """Dispatch single step to appropriate agent."""
        agent_name = step.get("agent")
        action = step.get("action")
        params = step.get("params", {})
        
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
                return [
                    {
                        "name": r.name,
                        "r2_test": r.r2_test,
                        "rmse_test": r.rmse_test
                    } for r in results
                ]
            elif action == "train_all":
                # Sequence
                r1 = target_agent.train_traditional_models()
                # Assuming deep learning might be added here in future
                
                # Get Top 3
                top_models = target_agent.get_top_models(k=3)
                top_summary = [f"{m.name} (R2={m.r2_test:.3f})" for m in top_models]
                return {"trained": len(r1), "top_3": top_summary}
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
