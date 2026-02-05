"""
IMCs Advanced Fusion Engine - 高级知识融合引擎

实现智能多源知识融合：
1. 动态权重调整
2. 材料属性加权
3. 证据质量评估
4. 可解释性推荐理由
5. 贡献来源追踪

Version: 1.0
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from src.agents.protocol import AgentContribution, ContributionType
from src.core.logger import get_logger

logger = get_logger(__name__)


class RecommendationReason(Enum):
    """推荐理由类型"""
    THEORY_SUPPORT = "theory_support"           # 理论计算支持
    ML_PREDICTION = "ml_prediction"             # ML预测高分
    EXPERIMENT_VALIDATED = "experiment_validated"  # 实验验证
    LITERATURE_MENTION = "literature_mention"   # 文献提及
    MULTI_SOURCE = "multi_source"               # 多源证据
    LOW_FORMATION_ENERGY = "low_formation_energy"  # 低形成能
    OPTIMAL_D_BAND = "optimal_d_band"           # 最优d带中心
    HIGH_ACTIVITY = "high_activity"             # 高活性


@dataclass
class RecommendationExplanation:
    """推荐解释"""
    material_id: str
    final_score: float
    rank: int
    
    # 各来源得分
    source_scores: Dict[str, float] = field(default_factory=dict)
    
    # 推荐理由
    reasons: List[RecommendationReason] = field(default_factory=list)
    reason_details: Dict[str, str] = field(default_factory=dict)
    
    # 材料属性
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # 置信度
    confidence: float = 0.0
    confidence_explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "material_id": self.material_id,
            "final_score": self.final_score,
            "rank": self.rank,
            "source_scores": self.source_scores,
            "reasons": [r.value for r in self.reasons],
            "reason_details": self.reason_details,
            "properties": self.properties,
            "confidence": self.confidence,
            "confidence_explanation": self.confidence_explanation,
        }
    
    def get_summary(self) -> str:
        """生成人类可读的推荐摘要"""
        reasons_cn = {
            RecommendationReason.THEORY_SUPPORT: "理论计算支持",
            RecommendationReason.ML_PREDICTION: "ML预测高分",
            RecommendationReason.EXPERIMENT_VALIDATED: "实验验证通过",
            RecommendationReason.LITERATURE_MENTION: "文献高频提及",
            RecommendationReason.MULTI_SOURCE: "多源证据一致",
            RecommendationReason.LOW_FORMATION_ENERGY: "低形成能稳定",
            RecommendationReason.OPTIMAL_D_BAND: "d带中心接近最优",
            RecommendationReason.HIGH_ACTIVITY: "高催化活性",
        }
        
        reason_strs = [reasons_cn.get(r, r.value) for r in self.reasons[:3]]
        return f"第{self.rank}名 ({self.final_score:.3f}): " + ", ".join(reason_strs)


class AdvancedFusionEngine:
    """
    高级知识融合引擎
    
    特点：
    - 动态权重调整
    - 材料属性加权
    - 证据质量评估
    - 可解释性推荐
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化融合引擎
        
        Args:
            config: 配置参数
        """
        config = config or {}
        
        # 基础权重
        self.base_weights = config.get("weights", {
            "theory": 0.30,
            "ml": 0.25,
            "experiment": 0.30,
            "literature": 0.15,
        })
        
        # HOR催化最优参数（用于属性评分）
        self.optimal_params = config.get("optimal_params", {
            "d_band_center": -2.0,      # eV, 相对于费米能级
            "formation_energy": -0.5,    # eV/atom
            "hydrogen_binding": -0.3,    # eV
            "overpotential": 0.05,       # V
        })
        
        # 多源奖励系数
        self.multi_source_bonus = config.get("multi_source_bonus", 0.15)
        
        # 证据质量阈值
        self.quality_thresholds = config.get("quality_thresholds", {
            "min_confidence": 0.3,
            "min_sources": 1,
        })
    
    def synthesize(
        self, 
        contributions: Dict[str, AgentContribution],
        query_context: Dict[str, Any] = None
    ) -> Tuple[List[Dict[str, Any]], List[RecommendationExplanation]]:
        """
        融合多智能体贡献，生成可解释的推荐
        
        Args:
            contributions: 各智能体的贡献
            query_context: 查询上下文（用于动态权重调整）
            
        Returns:
            (candidates, explanations): 排序后的候选列表和解释
        """
        logger.info("Starting advanced fusion...")
        
        # 1. 动态调整权重
        weights = self._adjust_weights(contributions, query_context)
        logger.info(f"Adjusted weights: {weights}")
        
        # 2. 收集所有候选及其属性
        all_candidates = self._collect_candidates(contributions)
        logger.info(f"Collected {len(all_candidates)} candidates")
        
        # 3. 计算多维度评分
        scored_candidates = self._calculate_scores(all_candidates, weights, contributions)
        
        # 4. 生成推荐解释
        explanations = self._generate_explanations(scored_candidates, contributions)
        
        # 5. 排序
        sorted_candidates = sorted(
            scored_candidates.values(),
            key=lambda x: x.get("final_score", 0),
            reverse=True
        )
        
        sorted_explanations = sorted(
            explanations,
            key=lambda x: x.final_score,
            reverse=True
        )
        
        # 设置排名
        for i, exp in enumerate(sorted_explanations):
            exp.rank = i + 1
        
        return sorted_candidates, sorted_explanations
    
    def _adjust_weights(
        self, 
        contributions: Dict[str, AgentContribution],
        query_context: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """根据实际贡献动态调整权重"""
        weights = self.base_weights.copy()
        
        # 计算各智能体的有效贡献
        effective_agents = []
        for name, contrib in contributions.items():
            if contrib.success and (contrib.candidates or contrib.predictions or contrib.insights):
                effective_agents.append(name)
        
        if not effective_agents:
            return weights
        
        # 重新分配失效智能体的权重
        total_failed_weight = sum(
            weights.get(name, 0) 
            for name in weights.keys() 
            if name not in effective_agents
        )
        
        if total_failed_weight > 0 and effective_agents:
            bonus_per_agent = total_failed_weight / len(effective_agents)
            for name in effective_agents:
                if name in weights:
                    weights[name] += bonus_per_agent
        
        # 基于贡献质量调整
        for name, contrib in contributions.items():
            if name in weights and contrib.success:
                # 根据置信度微调
                confidence_factor = 0.8 + 0.4 * contrib.confidence
                weights[name] *= confidence_factor
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _collect_candidates(
        self, 
        contributions: Dict[str, AgentContribution]
    ) -> Dict[str, Dict[str, Any]]:
        """收集所有候选材料及其属性"""
        all_candidates = {}
        
        for agent_name, contrib in contributions.items():
            if not contrib.success:
                continue
            
            # 处理 candidates
            for cand in contrib.candidates:
                mat_id = cand.get("material_id")
                if not mat_id:
                    continue
                
                if mat_id not in all_candidates:
                    all_candidates[mat_id] = {
                        "material_id": mat_id,
                        "formula": cand.get("formula"),
                        "sources": [],
                        "source_data": {},
                        "properties": {},
                    }
                
                all_candidates[mat_id]["sources"].append(agent_name)
                all_candidates[mat_id]["source_data"][agent_name] = {
                    "confidence": contrib.confidence,
                    "data": cand,
                }
                
                # 合并属性
                for k, v in cand.items():
                    if k not in ["material_id", "formula", "source"]:
                        all_candidates[mat_id]["properties"][k] = v
            
            # 处理 predictions
            for mat_id, score in contrib.predictions.items():
                if mat_id not in all_candidates:
                    all_candidates[mat_id] = {
                        "material_id": mat_id,
                        "sources": [],
                        "source_data": {},
                        "properties": {},
                    }
                all_candidates[mat_id]["sources"].append(f"{agent_name}_pred")
                all_candidates[mat_id]["properties"]["ml_score"] = score
            
            # 处理 properties
            for mat_id, props in contrib.properties.items():
                if mat_id in all_candidates:
                    all_candidates[mat_id]["properties"].update(props)
        
        return all_candidates
    
    def _calculate_scores(
        self,
        candidates: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
        contributions: Dict[str, AgentContribution]
    ) -> Dict[str, Dict[str, Any]]:
        """计算多维度评分"""
        
        for mat_id, data in candidates.items():
            source_scores = {}
            
            # 基于来源计算得分
            for source in set(data["sources"]):
                base_source = source.replace("_pred", "")
                weight = weights.get(base_source, 0.1)
                
                # 获取该来源的置信度
                source_data = data["source_data"].get(base_source, {})
                confidence = source_data.get("confidence", 0.5)
                
                source_scores[source] = weight * confidence
            
            # 属性加成
            props = data["properties"]
            property_bonus = 0.0
            
            # 形成能评分
            fe = props.get("formation_energy")
            if fe is not None:
                optimal_fe = self.optimal_params["formation_energy"]
                fe_score = math.exp(-abs(fe - optimal_fe) / 0.5)
                property_bonus += 0.1 * fe_score
                props["formation_energy_score"] = fe_score
            
            # d带中心评分
            d_band = props.get("d_band_center")
            if d_band is not None:
                optimal_d = self.optimal_params["d_band_center"]
                d_score = math.exp(-abs(d_band - optimal_d) / 1.0)
                property_bonus += 0.1 * d_score
                props["d_band_score"] = d_score
            
            # 多源奖励
            unique_sources = len(set(s.replace("_pred", "") for s in data["sources"]))
            multi_source_bonus = min(unique_sources * 0.05, self.multi_source_bonus)
            
            # 综合评分
            base_score = sum(source_scores.values())
            final_score = base_score + property_bonus + multi_source_bonus
            
            data["source_scores"] = source_scores
            data["property_bonus"] = property_bonus
            data["multi_source_bonus"] = multi_source_bonus
            data["final_score"] = final_score
        
        return candidates
    
    def _generate_explanations(
        self,
        candidates: Dict[str, Dict[str, Any]],
        contributions: Dict[str, AgentContribution]
    ) -> List[RecommendationExplanation]:
        """生成推荐解释"""
        explanations = []
        
        for mat_id, data in candidates.items():
            reasons = []
            reason_details = {}
            
            sources = set(s.replace("_pred", "") for s in data["sources"])
            props = data["properties"]
            
            # 多源证据
            if len(sources) >= 2:
                reasons.append(RecommendationReason.MULTI_SOURCE)
                reason_details["multi_source"] = f"来自 {len(sources)} 个智能体的证据"
            
            # 理论支持
            if "theory" in sources:
                reasons.append(RecommendationReason.THEORY_SUPPORT)
                fe = props.get("formation_energy")
                if fe is not None:
                    reason_details["theory"] = f"形成能 {fe:.3f} eV/atom"
            
            # ML预测
            if "ml" in sources or "ml_pred" in data["sources"]:
                reasons.append(RecommendationReason.ML_PREDICTION)
                ml_score = props.get("ml_score", 0)
                reason_details["ml"] = f"ML预测得分 {ml_score:.3f}"
            
            # 实验验证
            if "experiment" in sources:
                reasons.append(RecommendationReason.EXPERIMENT_VALIDATED)
                reason_details["experiment"] = "已有实验数据支持"
            
            # 文献提及
            if "literature" in sources:
                reasons.append(RecommendationReason.LITERATURE_MENTION)
                reason_details["literature"] = "文献中有相关报道"
            
            # 低形成能
            fe_score = props.get("formation_energy_score", 0)
            if fe_score > 0.7:
                reasons.append(RecommendationReason.LOW_FORMATION_ENERGY)
            
            # 最优d带
            d_score = props.get("d_band_score", 0)
            if d_score > 0.7:
                reasons.append(RecommendationReason.OPTIMAL_D_BAND)
            
            # 计算置信度
            confidence = min(data["final_score"], 1.0)
            confidence_explanation = self._generate_confidence_explanation(
                len(sources), confidence, reasons
            )
            
            exp = RecommendationExplanation(
                material_id=mat_id,
                final_score=data["final_score"],
                rank=0,  # 稍后设置
                source_scores=data.get("source_scores", {}),
                reasons=reasons,
                reason_details=reason_details,
                properties=props,
                confidence=confidence,
                confidence_explanation=confidence_explanation,
            )
            explanations.append(exp)
        
        return explanations
    
    def _generate_confidence_explanation(
        self,
        source_count: int,
        confidence: float,
        reasons: List[RecommendationReason]
    ) -> str:
        """生成置信度解释"""
        if confidence >= 0.8:
            level = "高"
            desc = "多源证据一致支持"
        elif confidence >= 0.5:
            level = "中"
            desc = "有一定证据支持"
        else:
            level = "低"
            desc = "证据有限，需进一步验证"
        
        return f"{level}置信度 ({confidence:.2f}): {desc}，基于 {source_count} 个来源、{len(reasons)} 个理由"


def create_fusion_report(
    explanations: List[RecommendationExplanation],
    top_n: int = 10
) -> str:
    """生成融合报告"""
    lines = ["# HOR催化剂推荐报告\n"]
    
    for exp in explanations[:top_n]:
        lines.append(f"## 第 {exp.rank} 名: {exp.material_id}")
        lines.append(f"- 综合得分: {exp.final_score:.3f}")
        lines.append(f"- 置信度: {exp.confidence_explanation}")
        lines.append("- 推荐理由:")
        for reason in exp.reasons:
            detail = exp.reason_details.get(reason.value.split("_")[0], "")
            lines.append(f"  - {reason.value}: {detail}")
        lines.append("")
    
    return "\n".join(lines)
