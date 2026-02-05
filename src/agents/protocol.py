"""
IMCs Agent Protocol - 智能体协议接口定义

定义多智能体协同系统的标准通信协议，使各智能体能够：
1. 评估自身能力 (assess_capability)
2. 报告资源状态 (get_resource_status)
3. 贡献专业知识 (contribute)

Version: 1.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ContributionType(Enum):
    """贡献类型"""
    CANDIDATES = "candidates"           # 候选材料列表
    PREDICTIONS = "predictions"         # ML预测结果
    PROPERTIES = "properties"           # 材料属性数据
    INSIGHTS = "insights"               # 文献洞察
    METRICS = "metrics"                 # 实验指标
    KNOWLEDGE = "knowledge"             # 知识图谱/规则


@dataclass
class AgentCapability:
    """
    智能体能力评估结果
    
    描述智能体对特定查询的贡献能力
    """
    can_contribute: bool                    # 是否能贡献
    confidence: float = 0.0                 # 置信度 0-1
    contribution_types: List[ContributionType] = field(default_factory=list)  # 可贡献类型
    requirements: List[str] = field(default_factory=list)   # 需要的前置条件
    estimated_items: int = 0                # 预估能提供的条目数
    estimated_time_seconds: float = 0.0     # 预估耗时(秒)
    reason: str = ""                        # 评估理由
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_contribute": self.can_contribute,
            "confidence": self.confidence,
            "contribution_types": [t.value for t in self.contribution_types],
            "requirements": self.requirements,
            "estimated_items": self.estimated_items,
            "estimated_time_seconds": self.estimated_time_seconds,
            "reason": self.reason,
        }


@dataclass
class ResourceStatus:
    """
    智能体资源状态
    
    描述智能体当前可用的资源和能力边界
    """
    agent_name: str
    is_available: bool = True
    
    # 数据资源
    data_count: int = 0                     # 数据条目数
    data_coverage: Dict[str, int] = field(default_factory=dict)  # 各类数据覆盖
    
    # 模型资源
    model_count: int = 0                    # 可用模型数
    model_types: List[str] = field(default_factory=list)
    
    # 外部连接
    api_available: bool = True              # 外部API是否可用
    api_quota_remaining: Optional[int] = None
    
    # 性能指标
    last_success_time: Optional[str] = None
    avg_response_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "is_available": self.is_available,
            "data_count": self.data_count,
            "data_coverage": self.data_coverage,
            "model_count": self.model_count,
            "model_types": self.model_types,
            "api_available": self.api_available,
            "api_quota_remaining": self.api_quota_remaining,
        }


@dataclass 
class AgentContribution:
    """
    智能体贡献结果
    
    包含智能体为查询提供的专业知识
    """
    agent_name: str
    contribution_type: ContributionType
    success: bool = True
    
    # 贡献内容
    candidates: List[Dict[str, Any]] = field(default_factory=list)  # 候选材料
    predictions: Dict[str, float] = field(default_factory=dict)     # 预测值 {material_id: score}
    properties: Dict[str, Dict] = field(default_factory=dict)       # 属性 {material_id: {prop: value}}
    insights: List[Dict[str, Any]] = field(default_factory=list)    # 文献洞察
    metrics: Dict[str, Dict] = field(default_factory=dict)          # 实验指标
    knowledge: Dict[str, Any] = field(default_factory=dict)         # 知识/规则
    
    # 元信息
    confidence: float = 0.0
    reasoning: str = ""
    sources: List[str] = field(default_factory=list)                # 数据来源
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "contribution_type": self.contribution_type.value,
            "success": self.success,
            "candidates": self.candidates,
            "predictions": self.predictions,
            "properties": self.properties,
            "insights": self.insights,
            "metrics": self.metrics,
            "knowledge": self.knowledge,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "sources": self.sources,
        }


@dataclass
class QueryContext:
    """
    查询上下文
    
    包含用户查询和已收集的各智能体贡献
    """
    user_query: str
    task_type: str = ""
    target_elements: List[str] = field(default_factory=list)
    target_properties: List[str] = field(default_factory=list)
    
    # 已收集的贡献
    contributions: Dict[str, AgentContribution] = field(default_factory=dict)
    
    # 迭代信息
    iteration: int = 1
    previous_candidates: List[str] = field(default_factory=list)
    
    def add_contribution(self, contribution: AgentContribution):
        """添加智能体贡献"""
        self.contributions[contribution.agent_name] = contribution
    
    def get_all_candidates(self) -> List[str]:
        """获取所有智能体推荐的候选材料ID"""
        all_ids = set()
        for contrib in self.contributions.values():
            for cand in contrib.candidates:
                if isinstance(cand, dict) and "material_id" in cand:
                    all_ids.add(cand["material_id"])
        return list(all_ids)


class AgentProtocol(ABC):
    """
    智能体协议抽象基类
    
    所有参与协同的智能体必须实现此接口
    """
    
    @property
    @abstractmethod
    def agent_name(self) -> str:
        """智能体名称"""
        pass
    
    @abstractmethod
    def assess_capability(self, query: str, context: Optional[QueryContext] = None) -> AgentCapability:
        """
        评估对给定查询的贡献能力
        
        Args:
            query: 用户查询
            context: 查询上下文(可选,包含其他智能体的贡献)
            
        Returns:
            AgentCapability: 能力评估结果
        """
        pass
    
    @abstractmethod
    def get_resource_status(self) -> ResourceStatus:
        """
        获取当前资源状态
        
        Returns:
            ResourceStatus: 资源状态报告
        """
        pass
    
    @abstractmethod
    def contribute(self, context: QueryContext) -> AgentContribution:
        """
        为查询贡献专业知识
        
        Args:
            context: 查询上下文
            
        Returns:
            AgentContribution: 贡献结果
        """
        pass
    
    def can_iterate(self) -> bool:
        """
        是否支持迭代优化
        
        Returns:
            bool: 是否可以基于反馈迭代
        """
        return False
    
    def iterate(self, context: QueryContext, feedback: Dict[str, Any]) -> AgentContribution:
        """
        基于反馈进行迭代优化
        
        Args:
            context: 查询上下文
            feedback: 反馈信息(如实验结果)
            
        Returns:
            AgentContribution: 更新后的贡献
        """
        raise NotImplementedError("This agent does not support iteration")
