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
from src.services.db.database import DatabaseService

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
        # Persist plan + initial steps (pending)
        try:
            db = DatabaseService()
            db.create_plan(
                plan_id=plan.task_id,
                user_id=None,
                task_type=plan.task_type.value,
                description=plan.description
            )
            for step in plan.steps:
                db.log_plan_step(
                    plan_id=plan.task_id,
                    step_id=step.step_id,
                    agent=step.agent,
                    action=step.action,
                    status="pending",
                    dependencies=step.dependencies,
                    params=step.params
                )
        except Exception as e:
            logger.warning(f"Failed to persist plan {plan.task_id}: {e}")
        return plan
    
    def format_plan(self, plan: TaskPlan = None) -> str:
        """Format plan for display."""
        plan = plan or self.current_plan
        if not plan: return "No plan created."
        
        output = f"TASK PLAN: {plan.task_id} [{plan.task_type.value}]\n"
        output += f"Description: {plan.description}\nSteps:\n"
        for i, step in enumerate(plan.steps, 1):
            output += f"  {i}. [{step.agent.upper()}] {step.action}\n"
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
        task_type = self.analyze_request(message)
        
        # Determine State (Basic State Machine)
        if not hasattr(self, 'conversation_state'):
            self.conversation_state = 'idle'
            self.draft_plan_context = {}

        # 1. Start Planning
        PLAN_KEYWORDS = [
            "create plan", "find", "search", "train", "analyze", "discover",
            "创建", "任务", "计划", "搜索", "训练", "分析", "发现", "筛选"
        ]
        is_start_command = any(k in msg_lower for k in PLAN_KEYWORDS) and len(message.split()) > 2
        should_start = task_type != TaskType.GENERAL

        if self.conversation_state == 'idle' and (is_start_command or should_start):
            self.conversation_state = 'planning'
            self.draft_plan_context['objective'] = message
            return {
                "type": "chat",
                "content": f"I understand you want to: '{message}'.\n\nBefore I execute, let's refine the plan:\n1. **Literature**: Should I search specifically on arXiv or ChemRxiv?\n2. **Theory**: Do you have target elements in mind (e.g. Pt, Ni)?"
            }

        # 2. Refine Plan (Planning Mode)
        if self.conversation_state == 'planning':
            if "execute" in msg_lower or "go" in msg_lower or "yes" in msg_lower:
                # Finalize
                final_request = f"{self.draft_plan_context.get('objective')} {message}"
                plan = self.create_plan(final_request)
                self.conversation_state = 'idle'
                self.draft_plan_context = {}
                from dataclasses import asdict
                plan_dict = asdict(plan)
                plan_dict["task_type"] = plan.task_type.value
                return {
                    "type": "plan",
                    "content": plan_dict
                }
            else:
                # Accumulate context
                self.draft_plan_context['details'] = message
                return {
                    "type": "chat",
                    "content": "Noted. Any specific requirements for the Machine Learning model? (e.g. 'Use Random Forest' or 'Deep Learning')\n\nType **'Execute'** to start."
                }

        # 3. Idle / Follow-up Mode
        response_text = f"I received: '{message}'.\n"
        if msg_lower in ["2", "2.", "analysis"]:
            response_text = "To proceed with Analysis (Step 2), please click the 'Execute' button on the plan card."
        elif msg_lower in ["reset", "cancel"]:
            self.conversation_state = 'idle'
            self.draft_plan_context = {}
            response_text = "Conversation reset. How can I help?"
        else:
            response_text += "You are in IDLE mode. Type 'Find catalysts...' to start a new task."
            
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
