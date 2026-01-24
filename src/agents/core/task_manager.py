"""
Task Manager Agent (TaskManagerAgent)
Refactored (v3.1) to use Service-Oriented Architecture.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.core.logger import get_logger, log_exception
from src.services.task.types import TaskPlan, TaskType
from src.services.task.planner import TaskPlanner
from src.services.task.executor import PlanExecutor

# Sub-Agents
from src.agents.core.ml_agent import MLAgent, MLAgentConfig
from src.agents.core.theory_agent import TheoryDataAgent, TheoryDataConfig
from src.agents.core.experiment_agent import ExperimentDataAgent, ExperimentDataConfig
from src.agents.core.literature_agent import LiteratureAgent, LiteratureConfig

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


class TaskManagerAgent:
    """
    Task Manager Agent - Central Orchestrator.
    Delegates to TaskPlanner and PlanExecutor.
    """
    
    def __init__(self, output_dir: str = "data/tasks"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Sub-Agents
        logger.info("Initializing Sub-Agents...")
        self.ml_agent = MLAgent(MLAgentConfig(output_dir="data/ml_agent"))
        self.theory_agent = TheoryDataAgent(TheoryDataConfig(output_dir="data/theory"))
        self.experiment_agent = ExperimentDataAgent(ExperimentDataConfig(output_dir="data/experimental"))
        self.literature_agent = LiteratureAgent(LiteratureConfig())
        
        # Initialize Services
        self.planner = TaskPlanner()
        self.executor = PlanExecutor(agents={
            "ml": self.ml_agent,
            "theory": self.theory_agent,
            "experiment": self.experiment_agent,
            "literature": self.literature_agent,
            "task_manager": self # For self-referential actions
        })
        
        # State
        self.current_plan: Optional[TaskPlan] = None
        self.task_history: List[TaskPlan] = []
        
        logger.info("TaskManagerAgent initialized (v3.1).")
    
    # ========== Task Analysis & Planning ==========
    
    @log_exception(logger)
    def analyze_request(self, user_request: str) -> TaskType:
        """Analyze user request."""
        return self.planner.analyze_request(user_request)
    
    @log_exception(logger)
    def create_plan(self, user_request: str) -> TaskPlan:
        """Create execution plan."""
        plan = self.planner.create_plan(user_request)
        self.current_plan = plan
        return plan
    
    def format_plan(self, plan: TaskPlan = None) -> str:
        """Format plan for display."""
        plan = plan or self.current_plan
        if not plan: return "No plan created."
        
        output = f"TASK PLAN: {plan.task_id} [{plan.task_type.value}]\n"
        output += f"Description: {plan.description}\nSteps:\n"
        for i, step in enumerate(plan.steps, 1):
            output += f"  {i}. [{step.get('agent', '').upper()}] {step.get('action', '')}\n"
        return output
    
    # ========== Execution ==========
    
    @log_exception(logger)
    def execute_plan(self, plan: TaskPlan = None) -> Dict[str, Any]:
        """Execute plan."""
        plan = plan or self.current_plan
        results = self.executor.execute_plan(plan)
        
        self.task_history.append(plan)
        return results
        
    # ========== Interaction ==========
    
    @log_exception(logger)
    def process_chat_message(self, message: str) -> Dict[str, Any]:
        """
        Process a natural language message from the user.
        Decides whether to create a Task Plan or return a simple Chat Response.
        """
        msg_lower = message.lower().strip()
        
        # 1. Heuristic: Is this a Command?
        # Commands usually start with verbs or are long sentences describing a goal.
        # Follow-ups are usually short ("2", "yes", "ok") or questions ("why?").
        
        is_command = False
        keywords = ["create plan", "find", "search", "train", "analyze", "discover"]
        
        if len(message.split()) > 3 and any(k in msg_lower for k in keywords):
            is_command = True
        elif "plan" in msg_lower:
            is_command = True
            
        # 2. Handle Command -> Create Plan
        if is_command:
            plan = self.create_plan(message)
            return {
                "type": "plan",
                "content": plan
            }
            
        # 3. Handle Follow-up / Conversation
        # If we have a previous plan context.. specifically 'recommend' step output.
        # Since we don't store "last_recommendation" in memory explicitly, we just echo.
        # Future: Store conversation context.
        
        response_text = f"I received: '{message}'.\n"
        
        if msg_lower in ["2", "2.", "analysis"]:
            response_text = "To proceed with Analysis (Step 2), please click the 'Execute' button on the generated plan above. Or say 'Create plan for Analysis' to start a specifc task."
        elif msg_lower in ["1", "literature"]:
             response_text = "To review Literature (Step 1), check the plan above."
        else:
            response_text += "If this is a new research goal, please describe it in a full sentence (e.g., 'Find catalysts for CO2RR')."
            
        return {
            "type": "chat",
            "content": response_text
        }

    def chat(self, message: str) -> str:
        """Simple chat interface (Legacy/Diagnostic)."""
        # In a real app, this would use an LLM.
        # Here we just analyze and basic response.
        task_type = self.analyze_request(message)
        return f"I understand you want to perform a {task_type.value} task. Please say 'create plan' to proceed."

    def confirm_and_execute(self) -> str:
        if self.current_plan:
             self.execute_plan(self.current_plan)
             return "Plan executed."
        return "No plan to execute."
        
    def save_state(self):
        # Placeholder for persistence
        pass

# ========== Convenience ==========

def create_catalyst_research_agent():
    return TaskManagerAgent()

def main():
    agent = TaskManagerAgent()
    print("TaskManager initialized.")

if __name__ == "__main__":
    main()
