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
    AgentProtocol, ResourceStatus
)
from src.agents.protocol_impl import (
    TheoryAgentProtocolMixin, MLAgentProtocolMixin,
    ExperimentAgentProtocolMixin, LiteratureAgentProtocolMixin
)
from src.agents.types import AgentCapability, QueryContext, AgentContribution, ContributionType
from src.agents.query_parser import QueryParser as LLMQueryParser # Renamed to avoid conflict with new QueryParser
from src.agents.fusion import FusionEngine, create_fusion_report
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
    contributions: Dict[str, AgentContribution] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    iteration: int = 1
    explanations: List[Any] = field(default_factory=list)  # RecommendationExplanation list
    parsed_intent: Dict[str, Any] = field(default_factory=dict)  # QueryParser output
    debate_record: Optional[Any] = None           # DebateRecord from ConflictDetector
    decision_session_id: str = ""                 # DecisionLogger session ID
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "candidates": self.candidates,
            "reasoning": self.reasoning,
            "execution_order": self.execution_order,
            "iteration": self.iteration,
            "contributions": {k: v.to_dict() for k, v in self.contributions.items()}
        }


class FusionEngine:
    """
    知识融合引擎
    
    融合多智能体贡献，生成综合推荐
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        初始化融合引擎
        
        Args:
            weights: 各智能体权重 {agent_name: weight}
        """
        self.weights = weights or {
            "theory": 0.3,
            "ml": 0.3,
            "experiment": 0.25,
            "literature": 0.15,
        }
    
    def synthesize(self, contributions: Dict[str, AgentContribution]) -> List[Dict[str, Any]]:
        """
        融合多智能体贡献，生成排序后的候选列表
        
        Args:
            contributions: 各智能体的贡献
            
        Returns:
            排序后的候选材料列表
        """
        # 收集所有候选材料
        all_candidates: Dict[str, Dict[str, Any]] = {}
        
        # 从各智能体收集候选
        for agent_name, contrib in contributions.items():
            if not contrib.success:
                continue
                
            weight = self.weights.get(agent_name, 0.1)
            
            # 处理 candidates
            for cand in contrib.candidates:
                mat_id = cand.get("material_id")
                if not mat_id:
                    continue
                    
                if mat_id not in all_candidates:
                    all_candidates[mat_id] = {
                        "material_id": mat_id,
                        "formula": cand.get("formula"),
                        "scores": {},
                        "sources": [],
                        "properties": {},
                    }
                
                all_candidates[mat_id]["sources"].append(agent_name)
                all_candidates[mat_id]["scores"][agent_name] = weight * contrib.confidence
                
                # 合并属性
                for k, v in cand.items():
                    if k not in ["material_id", "formula"]:
                        all_candidates[mat_id]["properties"][k] = v
            
            # 处理 predictions
            for mat_id, score in contrib.predictions.items():
                if mat_id not in all_candidates:
                    all_candidates[mat_id] = {
                        "material_id": mat_id,
                        "scores": {},
                        "sources": [],
                        "properties": {},
                    }
                all_candidates[mat_id]["sources"].append(f"{agent_name}_prediction")
                all_candidates[mat_id]["scores"][f"{agent_name}_pred"] = weight * score
            
            # 处理 properties
            for mat_id, props in contrib.properties.items():
                if mat_id in all_candidates:
                    all_candidates[mat_id]["properties"].update(props)
        
        # 计算综合评分
        for mat_id, data in all_candidates.items():
            scores = data["scores"]
            # 综合评分 = 各智能体评分之和 + 来源数量奖励
            total_score = sum(scores.values())
            source_bonus = len(set(data["sources"])) * 0.1  # 多源奖励
            data["final_score"] = total_score + source_bonus
        
        # 排序
        sorted_candidates = sorted(
            all_candidates.values(),
            key=lambda x: x.get("final_score", 0),
            reverse=True
        )
        
        return sorted_candidates


class AgentOrchestrator:
    """
    多智能体协调器
    
    负责协调各智能体的协同工作，实现：
    - 能力评估收集
    - 执行顺序调度
    - 迭代执行
    - 动态重规划
    """
    
    def __init__(self):
        """初始化协调器"""
        self.agents: Dict[str, Any] = {}
        self.query_parser = LLMQueryParser(LLMQueryParser())
        self.fusion_engine = FusionEngine()
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
    
    def schedule_execution(self, capabilities: Dict[str, AgentCapability]) -> List[str]:
        """
        根据能力评估决定执行顺序
        
        排序规则：
        1. 只选择 can_contribute=True 的智能体
        2. 按置信度降序排序
        3. 同等置信度时按预估时间升序
        
        Args:
            capabilities: 各智能体能力评估
            
        Returns:
            执行顺序的智能体名称列表
        """
        # 过滤可贡献的智能体
        valid_agents = [
            (name, cap) for name, cap in capabilities.items()
            if cap.can_contribute
        ]
        
        # 排序
        sorted_agents = sorted(
            valid_agents,
            key=lambda x: (-x[1].confidence, x[1].estimated_time_seconds)
        )
        
        return [name for name, _ in sorted_agents]
    
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
    
    def orchestrate(self, query: str, max_iterations: int = 3) -> RecommendationResult:
        """
        执行完整的协同推荐流程
        
        Args:
            query: 用户查询
            max_iterations: 最大迭代次数
            
        Returns:
            RecommendationResult: 推荐结果
        """
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
            logger.info(f"=== Iteration {iteration} ===")
            
            # 1. 收集能力评估
            capabilities = self.collect_capabilities(query, context)
            self.decision_logger.log_capabilities(session_id, capabilities)
            
            # 2. 决定执行顺序
            execution_order = self.schedule_execution(capabilities)
            logger.info(f"Execution order: {execution_order}")
            
            if not execution_order:
                logger.warning("No agent can contribute")
                return RecommendationResult(
                    success=False,
                    reasoning="没有智能体能够处理此查询",
                    decision_session_id=session_id,
                )
            
            # 3. 并发执行各智能体 (不再严格依赖串行，而是并发查询)
            # 注意: 如果有依赖关系(比如 ml 需要 theory 的数据)，则需要更复杂的 DAG 调度
            # 当前架构中 QueryContext 会传入各 Agent，各 Agent 尝试计算它的可获取候选
            
            def _execute_agent(agent_name: str):
                if agent_name not in self.agents:
                    return None
                    
                agent = self.agents[agent_name]
                logger.info(f"Executing {agent_name} in parallel...")
                
                try:
                    contribution = agent.contribute(context)
                    logger.info(f"[{agent_name}] contributed {len(contribution.candidates)} candidates")
                    return (agent_name, contribution)
                except Exception as e:
                    logger.error(f"[{agent_name}] failed: {e}")
                    failed_contrib = AgentContribution(
                        agent_name=agent_name,
                        contribution_type=ContributionType.CANDIDATES,
                        success=False,
                        reasoning=str(e)
                    )
                    return (agent_name, failed_contrib)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(execution_order) or 1) as executor:
                futures = {executor.submit(_execute_agent, name): name for name in execution_order}
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
        
        # 生成融合报告
        fusion_report = create_fusion_report(explanations, top_n=10)
        self.decision_logger.log_fusion(session_id, candidates, explanations)
        
        elapsed = time.time() - start_time
        logger.info(f"Orchestration completed in {elapsed:.2f}s, {len(candidates)} candidates")
        
        # === ACTIVE LEARNING PROTOCOL (自适应阈值) ===
        active_learning_requests = []
        uncertainties = [
            c.get("properties", {}).get("uncertainty", 0)
            for c in candidates[:20]
            if c.get("properties", {}).get("uncertainty", 0) > 0
        ]
        al_threshold = float(np.percentile(uncertainties, 90)) if len(uncertainties) >= 3 else 0.8
        logger.info(f"Active Learning threshold (P90): {al_threshold:.4f}")
        
        for cand in candidates[:10]:
            props = cand.get("properties", {})
            score = props.get("predicted_activity", 0)
            uncertainty = props.get("uncertainty", 0)
            
            if score > 0 and uncertainty > al_threshold:
                cand["active_learning_reason"] = (
                    f"模型对 {cand.get('formula', '')} 的预测不确定度 ({uncertainty:.4f}) "
                    f"超过自适应阈值 ({al_threshold:.4f})，需实验验证。"
                )
                active_learning_requests.append(cand)
                
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
                execution_order=execution_order,
                iteration=context.iteration,
                explanations=explanations,
                parsed_intent=parsed_intent,
                debate_record=debate_record,
                decision_session_id=session_id,
            )
            self.decision_logger.log_final(session_id, result)
            
            # strategy feedback integration for active learning
            try:
                self.meta_controller.strategy_feedback(session_id, "partial", len(active_learning_requests))
            except Exception as e:
                logger.error(f"Strategy feedback failed: {e}")
                
            return result
        
        self.decision_logger.log_active_learning(session_id, False)
        expert_report = llm_service.generate_report(candidates[:20], is_active_learning=False)
        
        # 构建冲突摘要
        conflict_summary = ""
        if debate_record.total_conflicts > 0:
            conflict_summary = f"\n\n[冲突检测]\n{debate_record.summary}"
        
        result = RecommendationResult(
            success=True,
            candidates=candidates[:20],
            reasoning=f"[Director 深度剖析]\n{expert_report}\n\n[融合报告]\n{fusion_report}{conflict_summary}",
            contributions=context.contributions,
            execution_order=execution_order,
            iteration=context.iteration,
            explanations=explanations,
            parsed_intent=parsed_intent,
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
