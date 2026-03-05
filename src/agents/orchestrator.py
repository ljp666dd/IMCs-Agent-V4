"""
IMCs Agent Orchestrator - 多智能体协调器

实现智能体协同决策机制：
1. 收集各智能体能力评估
2. 按置信度智能排序
3. 迭代执行并收集贡献
4. 动态重规划
5. 知识融合推荐

Version: 1.0
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import concurrent.futures
import numpy as np

from src.agents.protocol import (
    AgentProtocol, AgentCapability, ResourceStatus, 
    QueryContext, AgentContribution, ContributionType
)
from src.agents.protocol_impl import (
    TheoryAgentProtocolMixin, MLAgentProtocolMixin,
    ExperimentAgentProtocolMixin, LiteratureAgentProtocolMixin
)
from src.agents.query_parser import QueryParser as LLMQueryParser # Renamed to avoid conflict with new QueryParser
from src.agents.fusion import AdvancedFusionEngine, create_fusion_report
from src.agents.conflict_detector import ConflictDetector
from src.agents.decision_logger import DecisionLogger
from src.services.task.meta_controller import MetaController
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecommendationResult:
    """推荐结果"""
    success: bool
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    decision_session_id: str = ""
    contributions: Dict[str, Any] = field(default_factory=list)
    execution_order: List[Any] = field(default_factory=list)
    iteration: int = 1
    explanations: List[Any] = field(default_factory=list)  # RecommendationExplanation list
    parsed_intent: Dict[str, Any] = field(default_factory=dict)  # QueryParser output
    debate_record: Optional[Any] = None           # DebateRecord from ConflictDetector
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "candidates": self.candidates,
            "reasoning": self.reasoning,
            "execution_order": self.execution_order,
            "iteration": self.iteration,
            "contributions": {k: v.to_dict() if hasattr(v, "to_dict") else str(v) for k, v in self.contributions.items()}
        }


class AgentOrchestrator:
    """
    多智能体协调器
    
    负责协调各智能体的协同工作，实现：
    - 能力评估收集
    - 执行顺序调度
    - 迭代执行
    - 动态重规划
    """
    
    def __init__(self, on_progress=None):
        """初始化协调器"""
        self.on_progress = on_progress
        self.agents: Dict[str, Any] = {}
        self.query_parser = LLMQueryParser()
        self.fusion_engine = AdvancedFusionEngine()
        self.conflict_detector = ConflictDetector()
        self.decision_logger = DecisionLogger()
        self.meta_controller = MetaController()
        
        # 初始化默认智能体
        self._init_agents()
    
    def _init_agents(self):
        """初始化所有智能体（使用混入类扩展）"""
        try:
            from src.agents.core.theory_agent import TheoryDataAgent, TheoryDataConfig
            
            class TheoryAgentWithProtocol(TheoryAgentProtocolMixin, TheoryDataAgent):
                pass
            
            self.agents["theory"] = TheoryAgentWithProtocol(TheoryDataConfig())
            logger.info("TheoryAgent initialized with protocol")
        except Exception as e:
            logger.warning(f"Failed to init TheoryAgent: {e}")
        
        try:
            from src.agents.core.ml_agent import MLAgent
            
            class MLAgentWithProtocol(MLAgentProtocolMixin, MLAgent):
                pass
            
            self.agents["ml"] = MLAgentWithProtocol()
            logger.info("MLAgent initialized with protocol")
        except Exception as e:
            logger.warning(f"Failed to init MLAgent: {e}")
        
        try:
            from src.agents.core.experiment_agent import ExperimentDataAgent
            
            class ExperimentAgentWithProtocol(ExperimentAgentProtocolMixin, ExperimentDataAgent):
                pass
            
            self.agents["experiment"] = ExperimentAgentWithProtocol()
            logger.info("ExperimentAgent initialized with protocol")
        except Exception as e:
            logger.warning(f"Failed to init ExperimentAgent: {e}")
        
        try:
            from src.agents.core.literature_agent import LiteratureAgent
            
            class LiteratureAgentWithProtocol(LiteratureAgentProtocolMixin, LiteratureAgent):
                pass
            
            self.agents["literature"] = LiteratureAgentWithProtocol()
            logger.info("LiteratureAgent initialized with protocol")
        except Exception as e:
            logger.warning(f"Failed to init LiteratureAgent: {e}")
    
    def collect_capabilities(self, query: str, context: Optional[QueryContext] = None) -> Dict[str, AgentCapability]:
        """
        收集所有智能体的能力评估
        
        Args:
            query: 用户查询
            context: 查询上下文
            
        Returns:
            {agent_name: AgentCapability}
        """
        capabilities = {}
        for name, agent in self.agents.items():
            try:
                cap = agent.assess_capability(query, context)
                capabilities[name] = cap
                logger.info(f"[{name}] can_contribute={cap.can_contribute}, confidence={cap.confidence:.2f}")
            except Exception as e:
                logger.warning(f"Failed to assess {name}: {e}")
                capabilities[name] = AgentCapability(
                    can_contribute=False,
                    reason=f"Error: {e}"
                )
        return capabilities

    def _get_agent_dependencies(self, agent_name: str) -> List[str]:
        """获取智能体的显式依赖关系 (V5.4)"""
        deps = {
            "ml": ["theory"],
            "experiment": ["ml", "theory"],
            "literature": [],
            "theory": [],
        }
        return deps.get(agent_name, [])

    def schedule_execution(self, capabilities: Dict[str, AgentCapability]) -> List[List[str]]:
        """基于能力评估与依赖关系生成执行计划 (V5.4 DAG 层级调度)"""
        to_execute = [name for name, cap in capabilities.items() if cap.can_contribute]
        if not to_execute:
            return []
            
        adj = {name: [] for name in to_execute}
        in_degree = {name: 0 for name in to_execute}
        for name in to_execute:
            for dep in self._get_agent_dependencies(name):
                if dep in to_execute:
                    adj[dep].append(name)
                    in_degree[name] += 1
        
        layers = []
        queue = [n for n in to_execute if in_degree[n] == 0]
        while queue:
            current_layer = sorted(queue, key=lambda x: -capabilities[x].confidence)
            layers.append(current_layer)
            queue = []
            for node in current_layer:
                for neighbor in adj[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        return layers
    
    def should_replan(self, context: QueryContext, capabilities: Dict[str, AgentCapability]) -> bool:
        """
        判断是否需要重规划
        
        触发条件：
        - 某个高优先级智能体失败
        - 发现新的候选需要其他智能体验证
        - 资源状态发生变化
        """
        # 简单实现：如果有失败的贡献，考虑重规划
        for name, contrib in context.contributions.items():
            if not contrib.success:
                logger.info(f"[{name}] failed, considering replan")
                return True
        return False
    
    def _emit_progress(self, message: str, stage: str = "info"):
        """V6 UI Callback Hook"""
        if hasattr(self, "on_progress") and callable(self.on_progress):
            self.on_progress({"stage": stage, "message": message})

    def orchestrate(self, query: str, max_iterations: int = 3) -> RecommendationResult:
        """
        执行完整的协同推荐流程
        
        Args:
            query: 用户查询
            max_iterations: 最大迭代次数
            
        Returns:
            RecommendationResult: 推荐结果
        """
        self._emit_progress(f"Starting orchestration for: {query}", "start")
        logger.info(f"Orchestrating query: {query}")
        start_time = time.time()
        
        # 开始决策链日志
        session_id = self.decision_logger.create_session(query)
        
        # 解析用户意图
        parsed_intent = self.query_parser.parse(query)
        self.decision_logger.log_intent(session_id, parsed_intent)
        
        # 创建查询上下文 (填充 target_elements)
        context = QueryContext(
            user_query=query,
            task_type="catalyst_discovery",
            target_elements=parsed_intent.get("target_elements", []),
            target_properties=list(parsed_intent.get("constraints", {}).keys()),
        )
        
        for iteration in range(1, max_iterations + 1):
            context.iteration = iteration
            self._emit_progress(f"Starting Multi-Agent Iteration {iteration}/{max_iterations}", "iteration")
            logger.info(f"=== Iteration {iteration} ===")
            
            # 1. 收集能力评估
            self._emit_progress("Collecting agent capabilities...", "planning")
            capabilities = self.collect_capabilities(query, context)
            self.decision_logger.log_capabilities(session_id, capabilities)
            
            # 2. 决定执行顺序
            self._emit_progress("Scheduling execution DAG...", "planning")
            execution_plan = self.schedule_execution(capabilities)
            logger.info(f"Execution plan (DAG Layers): {execution_plan}")
            
            if not execution_plan:
                logger.warning("No agent can contribute")
                return RecommendationResult(
                    success=False,
                    reasoning="没有智能体能够处理此查询",
                    decision_session_id=session_id,
                )
            
            # 3. 按层级执行 (V5.4 DAG Executor)
            for layer_idx, layer in enumerate(execution_plan):
                logger.info(f"Executing Layer {layer_idx}: {layer}")
                
                def _execute_agent(agent_name: str):
                    if agent_name not in self.agents:
                        return None
                    agent = self.agents[agent_name]
                    try:
                        contribution = agent.contribute(context)
                        return (agent_name, contribution)
                    except Exception as e:
                        logger.error(f"[{agent_name}] failed: {e}")
                        return (agent_name, AgentContribution(
                            agent_name=agent_name, success=False, reasoning=str(e)
                        ))

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(layer)) as executor:
                    futures = {executor.submit(_execute_agent, name): name for name in layer}
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res:
                            agent_name, contribution = res
                            context.add_contribution(contribution)
                            self.decision_logger.log_contribution(session_id, agent_name, contribution)
            
            # 4. 检查是否需要重规划
            if iteration < max_iterations and self.should_replan(context, capabilities):
                logger.info("Replanning...")
                continue
            else:
                break
        
        # 4.5 冲突检测
        debate_record = self.conflict_detector.detect(context.contributions, query)
        self.decision_logger.log_conflicts(session_id, debate_record)
        
        # 5. 融合推荐 (使用 AdvancedFusionEngine)
        logger.info("Fusing contributions with AdvancedFusionEngine...")
        candidates, explanations = self.fusion_engine.synthesize(context.contributions)
        
        # V5.4+ Global Self-Audit
        from src.services.llm.expert_reasoning import get_expert_reasoning
        llm_service = get_expert_reasoning()
        
        audit_results = {"consistent": True, "findings": []}
        try:
            logger.info("Performing global self-audit for consistency...")
            audit_results = llm_service.audit_consistency(context.contributions, candidates)
            if not audit_results.get("consistent"):
                logger.warning(f"Self-audit found {len(audit_results.get('findings', []))} issues")
        except Exception as e:
            logger.error(f"Self-audit failed: {e}")

        # 生成融合报告
        fusion_report = create_fusion_report(explanations, top_n=10)
        self.decision_logger.log_fusion(session_id, candidates, explanations)
        
        elapsed = time.time() - start_time
        logger.info(f"Orchestration completed in {elapsed:.2f}s, {len(candidates)} candidates")
        
        # === ACTIVE LEARNING PROTOCOL (V5 Bayesian AL) ===
        from src.services.ml.bayesian_al import get_bayesian_al_service
        al_service = get_bayesian_al_service(strategy="EI")
        
        # Try to fit GP if we have prior data
        try:
            from src.services.db.database import DatabaseService
            db = DatabaseService()
            prior_metrics = db.list_robot_tasks(limit=100)
            # Collect prior observations for GP fitting if sufficient
            # This is a lightweight attempt; real production would use activity_metrics
        except Exception:
            pass
            
        # V6 MO-AL: Use select_for_experiment which prefers Pareto front
        selected_al = al_service.select_for_experiment(candidates[:20], n_select=5)
        active_learning_requests = []
        
        for al_cand in selected_al:
            orig = next(
                (c for c in candidates if c.get("material_id") == al_cand.material_id), 
                None
            )
            if orig:
                prefix = "[最优性价比(Pareto)] " if al_cand.is_pareto_optimal else ""
                orig["active_learning_reason"] = f"{prefix}{al_cand.reason}"
                orig["ei_score"] = al_cand.ei_score
                orig["pi_score"] = al_cand.pi_score
                active_learning_requests.append(orig)
        
        logger.info(f"Bayesian AL: {len(active_learning_requests)} candidates selected (strategy={al_service.strategy})")
                
        from src.services.llm.expert_reasoning import get_expert_reasoning
        llm_service = get_expert_reasoning()

        if active_learning_requests:
            logger.warning(f"Active Learning Triggered! {len(active_learning_requests)} uncertain champions.")
            self.decision_logger.log_active_learning(session_id, True, active_learning_requests)
            expert_report = llm_service.generate_report(active_learning_requests, is_active_learning=True)
            result = RecommendationResult(
                success=True,
                candidates=active_learning_requests,
                reasoning=f"[Active Learning]\n{expert_report}",
                contributions=context.contributions,
                execution_order=execution_plan,
                iteration=context.iteration,
                explanations=explanations,
                parsed_intent=capabilities, # Using capabilities as proxy for intent here or keep it dummy
                debate_record=debate_record,
                decision_session_id=session_id,
            )
            self.decision_logger.log_final(session_id, result)
            
            # strategy feedback
            try:
                self.meta_controller.strategy_feedback(session_id, "partial", len(active_learning_requests))
            except Exception as e:
                logger.error(f"Strategy feedback failed: {e}")
            return result
        
        self.decision_logger.log_active_learning(session_id, False)
        expert_report = llm_service.generate_report(candidates[:20], is_active_learning=False)
        
        # 构建冲突摘要
        conflict_summary = ""
        if debate_record and hasattr(debate_record, "total_conflicts") and debate_record.total_conflicts > 0:
            conflict_summary = f"\n\n[冲突检测]\n{debate_record.summary}"
        
        result = RecommendationResult(
            success=True,
            candidates=candidates[:20],
            reasoning=f"[Director 深度剖析]\n{expert_report}\n\n[融合报告]\n{fusion_report}{conflict_summary}",
            contributions=context.contributions,
            execution_order=execution_plan,
            iteration=context.iteration,
            explanations=explanations,
            debate_record=debate_record,
            decision_session_id=session_id,
        )
        self.decision_logger.log_final(session_id, result)
        
        # strategy feedback integration for general success
        try:
            val_yield = len(candidates)
            outcome = "success" if val_yield > 0 else "failure"
            self.meta_controller.strategy_feedback(session_id, outcome, val_yield)
        except Exception as e:
            logger.error(f"Strategy feedback failed: {e}")
            
        return result

    def submit_task(self, query: str) -> int:
        """提交一个异步发现任务 (V5.6)"""
        from src.services.db.database import DatabaseService
        db = DatabaseService()
        task_id = db.create_robot_task(
            task_type="catalyst_discovery",
            payload={"query": query}
        )
        logger.info(f"Submitted async task {task_id} for query: {query}")
        return task_id
    
    def iterate_with_feedback(
        self, 
        previous_result: RecommendationResult, 
        feedback: Dict[str, Any]
    ) -> RecommendationResult:
        """
        基于反馈进行迭代优化
        
        Args:
            previous_result: 上一次推荐结果
            feedback: 用户反馈（如实验结果）
            
        Returns:
            更新后的推荐结果
        """
        logger.info("Iterating with feedback...")
        
        # 创建新的上下文
        context = QueryContext(
            user_query=feedback.get("query", ""),
            task_type="catalyst_discovery",
            iteration=previous_result.iteration + 1,
            previous_candidates=[c.get("material_id") for c in previous_result.candidates]
        )
        
        # 复制之前的贡献
        context.contributions = previous_result.contributions.copy()
        
        # 调用支持迭代的智能体
        for name, agent in self.agents.items():
            if hasattr(agent, 'can_iterate') and agent.can_iterate():
                try:
                    contribution = agent.iterate(context, feedback)
                    context.add_contribution(contribution)
                    logger.info(f"[{name}] iterated with feedback")
                except Exception as e:
                    logger.warning(f"[{name}] iteration failed: {e}")
        
        # 重新融合
        candidates, explanations = self.fusion_engine.synthesize(context.contributions)
        
        return RecommendationResult(
            success=True,
            candidates=candidates[:20],
            reasoning=f"基于实验反馈的第 {context.iteration} 轮优化推荐",
            contributions=context.contributions,
            execution_order=previous_result.execution_order,
            iteration=context.iteration
        )


# 便捷函数
def recommend_catalysts(query: str) -> RecommendationResult:
    """
    推荐催化剂的便捷函数
    
    Args:
        query: 用户查询
        
    Returns:
        RecommendationResult
    """
    orchestrator = AgentOrchestrator()
    return orchestrator.orchestrate(query)
