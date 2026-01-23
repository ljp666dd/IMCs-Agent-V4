from datetime import datetime
from src.core.logger import get_logger, log_exception
from src.services.task.types import TaskPlan, TaskType

logger = get_logger(__name__)

class TaskPlanner:
    """
    Service for generating execution plans from user requests.
    """
    
    @log_exception(logger)
    def analyze_request(self, user_request: str) -> TaskType:
        """Analyze user request to determine task type."""
        request_lower = user_request.lower()
        
        if any(kw in request_lower for kw in ["discover", "find", "screen", "candidate"]):
            return TaskType.CATALYST_DISCOVERY
        elif any(kw in request_lower for kw in ["predict", "train", "model", "machine learning"]):
            return TaskType.PROPERTY_PREDICTION
        elif any(kw in request_lower for kw in ["analyze", "performance", "test", "experiment"]):
            return TaskType.PERFORMANCE_ANALYSIS
        elif any(kw in request_lower for kw in ["literature", "paper", "review", "knowledge"]):
            return TaskType.LITERATURE_REVIEW
        else:
            return TaskType.GENERAL

    @log_exception(logger)
    def create_plan(self, user_request: str) -> TaskPlan:
        """Create an execution plan based on user request."""
        task_type = self.analyze_request(user_request)
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        plan = TaskPlan(
            task_id=task_id,
            task_type=task_type,
            description=user_request
        )
        
        # Template-based Planning (Rule Engine)
        if task_type == TaskType.CATALYST_DISCOVERY:
            plan.steps = [
                {"agent": "literature", "action": "search", "params": {"query": user_request}},
                {"agent": "theory", "action": "download", "params": {"data_types": ["cif", "formation_energy", "dos"]}},
                {"agent": "ml", "action": "train", "params": {"include_deep_learning": True}},
                {"agent": "task_manager", "action": "recommend", "params": {}}
            ]
        
        elif task_type == TaskType.PROPERTY_PREDICTION:
            plan.steps = [
                {"agent": "theory", "action": "load_data", "params": {}},
                {"agent": "ml", "action": "train_all", "params": {"include_deep_learning": True}},
                {"agent": "ml", "action": "shap_analysis", "params": {}},
                {"agent": "task_manager", "action": "summarize", "params": {}}
            ]
        
        elif task_type == TaskType.PERFORMANCE_ANALYSIS:
            plan.steps = [
                {"agent": "experiment", "action": "process", "params": {}},
                {"agent": "ml", "action": "predict", "params": {}},
                {"agent": "literature", "action": "compare", "params": {}},
                {"agent": "task_manager", "action": "recommend", "params": {}}
            ]
        
        elif task_type == TaskType.LITERATURE_REVIEW:
            plan.steps = [
                {"agent": "literature", "action": "search", "params": {"query": user_request}},
                {"agent": "literature", "action": "extract_knowledge", "params": {"topic": user_request}},
                {"agent": "task_manager", "action": "summarize", "params": {}}
            ]
        
        else:
            # General Query
            plan.steps = [
                {"agent": "task_manager", "action": "analyze", "params": {"request": user_request}}
            ]
            
        logger.info(f"Created plan {task_id} ([{task_type.value}]) for: {user_request}")
        return plan
