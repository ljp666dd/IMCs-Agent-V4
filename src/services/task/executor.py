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
                return target_agent.search_semantic_scholar(params.get("query", ""))
            elif action == "extract_knowledge":
                # Assuming LiteratureAgentv3.1 has extract_knowledge(topic)
                return target_agent.extract_knowledge(params.get("topic", ""))
        
        elif agent_name == "theory":
            if action == "load_data":
                return target_agent.get_status()
            elif action == "download":
                return {"message": "Download invoked (requires user confirmation in UI)"}
        
        elif agent_name == "experiment":
             if action == "process":
                 return {"message": "Scanning for experiment data..."}
                 
        elif agent_name == "ml":
            if action == "train":
                return target_agent.train_traditional_models()
            elif action == "train_all":
                # Sequence
                r1 = target_agent.train_traditional_models()
                return {"traditional": len(r1)}
            elif action == "shap_analysis":
                return {"message": "SHAP analysis ready"}
            elif action == "predict":
                return {"message": "Prediction ready"}
                
        elif agent_name == "task_manager":
            # Self-referential actions usually handled by UI or simple summaries
            if action == "recommend":
                return {"recommendation": "Review literature and trained models."}
            elif action == "summarize":
                 return {"summary": "Task completed."}
                 
        return {"status": "executed", "action": action}
