from datetime import datetime
from typing import List
from src.core.logger import get_logger, log_exception
from src.services.task.types import TaskPlan, TaskType
import re

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except Exception:
    LangDetectException = Exception
    HAS_LANGDETECT = False



def _has_cjk(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None

def _detect_language(text: str) -> str:
    if HAS_LANGDETECT:
        try:
            lang = detect(text)
            if lang.startswith("zh"):
                return "zh"
            if lang.startswith("en"):
                return "en"
            return lang
        except LangDetectException:
            pass
    if _has_cjk(text):
        return "zh"
    return "en"

def _extract_element_symbols(text: str) -> List[str]:
    """Extract element-like symbols from text (best-effort)."""
    if not text:
        return []
    # Match element symbols by capital letter + optional lowercase.
    return re.findall(r"[A-Z][a-z]?", text)

logger = get_logger(__name__)

class TaskPlanner:
    """
    Service for generating execution plans from user requests.
    """

    def __init__(self):
        try:
            from src.services.task.meta_controller import MetaController
            self.meta_controller = MetaController()
        except Exception:
            self.meta_controller = None
    
    @log_exception(logger)
    def analyze_request(self, user_request: str) -> TaskType:
        """Analyze user request to determine task type."""
        request_lower = user_request.lower()
        try:
            from src.agents.core.theory_agent import TheoryDataConfig
            allowed_elements = set(TheoryDataConfig().elements)
        except Exception:
            allowed_elements = set()
        element_hits = [el for el in _extract_element_symbols(user_request) if el in allowed_elements]
        has_material_hint = ("mp-" in request_lower) or (len(set(element_hits)) >= 2)
        force_en = any(kw in request_lower for kw in [
            "discover", "find", "screen", "candidate", "recommend", "design", "search", "explore",
            "predict", "train", "model", "machine learning", "ml",
            "analyze", "analysis", "performance", "activity", "overpotential", "experiment", "kinetics", "lsv", "cv", "stability", "polarization", "adsorption",
            "literature", "paper", "review", "survey", "knowledge", "rag",
            "catalyst", "alloy", "material", "hor", "her"
        ])
        lang = "en" if force_en else _detect_language(user_request)
        
        # ??????????????????; ???????????????(??? HOR/??????)
        en_discovery = ["discover", "find", "screen", "candidate", "recommend", "design", "search", "explore"]
        en_ml = ["predict", "train", "model", "machine learning", "ml"]
        en_perf = ["analyze", "analysis", "performance", "activity", "overpotential", "experiment", "kinetics", "lsv", "cv", "stability", "polarization", "adsorption"]
        en_lit = ["literature", "paper", "review", "survey", "knowledge", "rag"]
        en_context = ["catalyst", "alloy", "material", "hor", "her"]
        cn_discovery = ["发现", "筛选", "候选", "寻找", "搜索"]
        cn_context = ["合金", "催化", "材料", "HOR", "HER", "有序"]
        cn_ml = ["预测", "训练", "模型", "机器学习"]
        cn_perf = ["分析", "性能", "测试", "实验", "表征"]
        cn_lit = ["文献", "论文", "综述", "调研", "知识"]
        if lang == "en":
            if any(kw in request_lower for kw in en_discovery):
                return TaskType.CATALYST_DISCOVERY
            if any(kw in request_lower for kw in en_ml):
                return TaskType.PROPERTY_PREDICTION
            if any(kw in request_lower for kw in en_perf) or any(kw in request_lower for kw in en_context):
                return TaskType.PERFORMANCE_ANALYSIS
            if any(kw in request_lower for kw in en_lit):
                return TaskType.LITERATURE_REVIEW
            if has_material_hint:
                return TaskType.CATALYST_DISCOVERY
        elif lang == "zh":
            if (any(kw in user_request for kw in cn_discovery) and any(ctx in user_request for ctx in cn_context)):
                return TaskType.CATALYST_DISCOVERY
            elif any(kw in user_request for kw in cn_ml):
                return TaskType.PROPERTY_PREDICTION
            elif any(kw in user_request for kw in cn_perf):
                return TaskType.PERFORMANCE_ANALYSIS
            elif any(kw in user_request for kw in cn_lit):
                return TaskType.LITERATURE_REVIEW
            elif has_material_hint:
                return TaskType.CATALYST_DISCOVERY
        else:
            if any(kw in request_lower for kw in en_discovery) or (any(kw in user_request for kw in cn_discovery) and any(ctx in user_request for ctx in cn_context)):
                return TaskType.CATALYST_DISCOVERY
            if any(kw in request_lower for kw in en_ml) or any(kw in user_request for kw in cn_ml):
                return TaskType.PROPERTY_PREDICTION
            if any(kw in request_lower for kw in en_perf) or any(kw in user_request for kw in cn_perf):
                return TaskType.PERFORMANCE_ANALYSIS
            if any(kw in request_lower for kw in en_lit) or any(kw in user_request for kw in cn_lit):
                return TaskType.LITERATURE_REVIEW
            if has_material_hint:
                return TaskType.CATALYST_DISCOVERY
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
        def add_step(agent, action, params, deps=None, max_retries: int = 0, max_replans: int = 0):
            nonlocal count
            step_id = f"step_{count}" # Simple IDs for now
            step = TaskStep(
                step_id=step_id,
                agent=agent,
                action=action,
                params=params,
                dependencies=deps or [],
                max_retries=max_retries,
                max_replans=max_replans
            )
            plan.steps.append(step)
            count += 1
            return step_id

        # Template-based Planning (Rule Engine)
        dynamic_specs = []
        if self.meta_controller:
            try:
                dynamic_specs, _stats = self.meta_controller.build_initial_steps(task_type, user_request)
            except Exception:
                dynamic_specs = []

        # If we have dynamic specs, build plan from them
        if dynamic_specs:
            step_id_map = {}
            for spec in dynamic_specs:
                deps = []
                for dep in spec.get("deps", []):
                    if dep == "$literature":
                        if "literature" in step_id_map:
                            deps.append(step_id_map["literature"])
                    elif dep == "$theory":
                        if "theory" in step_id_map:
                            deps.append(step_id_map["theory"])
                    elif dep == "$ml":
                        if "ml" in step_id_map:
                            deps.append(step_id_map["ml"])
                    elif dep == "$experiment":
                        if "experiment" in step_id_map:
                            deps.append(step_id_map["experiment"])
                    else:
                        deps.append(dep)

                step_id = add_step(
                    spec.get("agent"),
                    spec.get("action"),
                    spec.get("params") or {},
                    deps=deps
                )
                # Track last step id per agent for dependency mapping
                step_id_map[spec.get("agent")] = step_id

            logger.info(f"Created plan {task_id} ([{task_type.value}]) for: {user_request} (dynamic)")
            return plan

        # Fallback to static template
        if task_type == TaskType.CATALYST_DISCOVERY:
            # 1. Literature Search
            s1 = add_step("literature", "search", {"query": user_request}, max_retries=1, max_replans=1)
            
            # 2. Theory Data (Depends on Lit slightly, but mainly parallel start)
            # Actually, theory usually runs independent or refined by lit.
            # Let's make it dependent for "pipeline" demo.
            s2 = add_step("theory", "download", {"data_types": ["cif", "formation_energy", "dos", "adsorption"]}, max_replans=1)
            
            # 3. ML Training (Depends on Theory Data)
            s3 = add_step("ml", "train", {"include_deep_learning": True}, deps=[s2], max_replans=1)
            
            # 4. Recommendation (Depends on ML results and Lit insights)
            s4 = add_step("task_manager", "recommend", {}, deps=[s1, s3])
        
        elif task_type == TaskType.PROPERTY_PREDICTION:
            s1 = add_step("theory", "load_data", {}, max_replans=1)
            s2 = add_step("ml", "train_all", {"include_deep_learning": True}, deps=[s1], max_replans=1)
            s3 = add_step("ml", "shap_analysis", {}, deps=[s2]) # Explainer needs trained model
            s4 = add_step("task_manager", "summarize", {}, deps=[s3])
        
        elif task_type == TaskType.PERFORMANCE_ANALYSIS:
            s1 = add_step("experiment", "process", {}, max_replans=1)
            s2 = add_step("ml", "predict", {}, deps=[s1], max_replans=1) # Predict based on experimental input structures?
            s3 = add_step("literature", "compare", {}, deps=[s1])
            s4 = add_step("task_manager", "recommend", {}, deps=[s2, s3])
        
        elif task_type == TaskType.LITERATURE_REVIEW:
            s1 = add_step("literature", "search", {"query": user_request}, max_retries=1, max_replans=1)
            s2 = add_step("literature", "extract_knowledge", {"topic": user_request}, deps=[s1])
            s3 = add_step("task_manager", "summarize", {}, deps=[s2])
        
        else:
            # General Query
            add_step("task_manager", "analyze", {"request": user_request})
            
        logger.info(f"Created plan {task_id} ([{task_type.value}]) for: {user_request}")
        return plan
