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
        """Create an execution plan based on user request (DAG)."""
        from src.services.task.types import TaskStep # Import locally to avoid circulars if any
        
        task_type = self.analyze_request(user_request)
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        plan = TaskPlan(
            task_id=task_id,
            task_type=task_type,
            description=user_request,
            steps=[]
        )
        
        # Helper to add step
        count = 1
        def add_step(agent, action, params, deps=None):
            nonlocal count
            step_id = f"step_{count}" # Simple IDs for now
            step = TaskStep(
                step_id=step_id,
                agent=agent,
                action=action,
                params=params,
                dependencies=deps or []
            )
            plan.steps.append(step)
            count += 1
            return step_id

        # Template-based Planning (Rule Engine)
        if task_type == TaskType.CATALYST_DISCOVERY:
            # 1. Literature Search
            s1 = add_step("literature", "search", {"query": user_request})
            
            # 2. Theory Data (Depends on Lit slightly, but mainly parallel start)
            # Actually, theory usually runs independent or refined by lit.
            # Let's make it dependent for "pipeline" demo.
            s2 = add_step("theory", "download", {"data_types": ["cif", "formation_energy", "dos"]})
            
            # 3. ML Training (Depends on Theory Data)
            s3 = add_step("ml", "train", {"include_deep_learning": True}, deps=[s2])
            
            # 4. Recommendation (Depends on ML results and Lit insights)
            s4 = add_step("task_manager", "recommend", {}, deps=[s1, s3])
        
        elif task_type == TaskType.PROPERTY_PREDICTION:
            s1 = add_step("theory", "load_data", {})
            s2 = add_step("ml", "train_all", {"include_deep_learning": True}, deps=[s1])
            s3 = add_step("ml", "shap_analysis", {}, deps=[s2]) # Explainer needs trained model
            s4 = add_step("task_manager", "summarize", {}, deps=[s3])
        
        elif task_type == TaskType.PERFORMANCE_ANALYSIS:
            s1 = add_step("experiment", "process", {})
            s2 = add_step("ml", "predict", {}, deps=[s1]) # Predict based on experimental input structures?
            s3 = add_step("literature", "compare", {}, deps=[s1])
            s4 = add_step("task_manager", "recommend", {}, deps=[s2, s3])
        
        elif task_type == TaskType.LITERATURE_REVIEW:
            s1 = add_step("literature", "search", {"query": user_request})
            s2 = add_step("literature", "extract_knowledge", {"topic": user_request}, deps=[s1])
            s3 = add_step("task_manager", "summarize", {}, deps=[s2])
        
        else:
            # General Query
            add_step("task_manager", "analyze", {"request": user_request})
            
        logger.info(f"Created plan {task_id} ([{task_type.value}]) for: {user_request}")
        return plan
