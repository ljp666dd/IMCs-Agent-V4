"""
Task Manager Agent (TaskManagerAgent)
Refactored (v4.0) to use AgentOrchestrator and LLM.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.core.logger import get_logger, log_exception
from src.services.task.types import TaskPlan, TaskType
from src.services.task.planner import TaskPlanner
from src.services.db.database import DatabaseService
from src.services.task.executor_factory import new_plan_executor

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


class TaskManagerAgent:
    """
    Task Manager Agent - Central Orchestrator.
    Delegates execution to AgentOrchestrator via IterativeSession.
    """
    
    def __init__(self, output_dir: str = "data/tasks"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Services
        self.planner = TaskPlanner()
        # Sub-agents are managed directly by AgentOrchestrator now.
        
        # State
        self.current_plan: Optional[TaskPlan] = None
        self.task_history: List[TaskPlan] = []
        
        logger.info("TaskManagerAgent initialized (v4.0 - Orchestrator Driven).")
    
    # ========== Task Analysis & Planning ==========
    
    @log_exception(logger)
    def analyze_request(self, user_request: str) -> TaskType:
        """Analyze user request."""
        return self.planner.analyze_request(user_request)
    
    @log_exception(logger)
    def create_plan(self, user_request: str) -> TaskPlan:
        """Create an intent-based execution plan."""
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
        """Execute plan using PlanExecutor (single source of truth: DB + persisted steps)."""
        plan = plan or self.current_plan
        if not plan:
            return {"error": "No plan provided"}

        executor = new_plan_executor()
        results = executor.execute_plan(plan)
        self.current_plan = plan
        self.task_history.append(plan)
        return results
        
    # ========== Interaction ==========
    
    @log_exception(logger)
    def process_chat_message(self, message: str) -> Dict[str, Any]:
        """
        Process a natural language message from the user using LLM.
        Decides whether to create a Task Plan or return a simple Chat Response.
        """
        msg_lower = message.lower().strip()
        task_type = self.analyze_request(message)
        
        # Determine State
        if not hasattr(self, 'conversation_state'):
            self.conversation_state = 'idle'
            self.draft_plan_context = {}
            
        from src.services.llm.expert_reasoning import get_expert_reasoning
        llm = get_expert_reasoning()

        # Check if user wants to execute
        if self.conversation_state == 'planning' and any(k in msg_lower for k in ["execute", "go", "yes", "执行", "开始"]):
            final_request = f"{self.draft_plan_context.get('objective', '')} {message}"
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
        
        # State update
        if self.conversation_state == 'idle':
            self.conversation_state = 'planning'
            self.draft_plan_context['objective'] = message
            
        prompt = f'''你是一个智能催化剂发现平台 (IMCs) 的课题组长 (PI)。
用户现在的需求是：{self.draft_plan_context.get('objective')}
用户刚才的回复是：{message}

请以课题组长的身份，用中文进行简短、专业的回复（1-3句话）。
- 澄清任何缺失的关键信息（如特定元素、需要何种性能分析等）。
- 或者确认你已经清楚需求。
- 必须在内容结尾提示用户，如果确认无误，请回复“执行”或“Execute”以开始自动编排流水线。'''
        try:
            reply = llm.generate_response(prompt)
        except Exception as e:
            logger.warning(f"LLM chat failed: {e}")
            reply = f"我已经了解您的需求：'{self.draft_plan_context.get('objective')}'。准备好后请回复 '执行'。"
            
        return {
            "type": "chat",
            "content": reply
        }

    def chat(self, message: str) -> str:
        res = self.process_chat_message(message)
        return res.get("content", str(res))

    def confirm_and_execute(self) -> str:
        if self.current_plan:
             self.execute_plan(self.current_plan)
             return "Plan executed."
        return "No plan to execute."
        
    def save_state(self):
        pass

# ========== Convenience ==========

def create_catalyst_research_agent():
    return TaskManagerAgent()

def main():
    agent = TaskManagerAgent()
    print("TaskManager initialized.")

if __name__ == "__main__":
    main()
