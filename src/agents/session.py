"""
IMCs Iterative Session - 多轮迭代会话管理

实现实验数据持续反馈和模型迭代优化：
1. 会话状态管理
2. 轮次跟踪
3. 实验反馈集成
4. 候选状态更新
5. 持久化支持

Version: 1.0
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.agents.orchestrator import AgentOrchestrator, RecommendationResult
from src.core.logger import get_logger

logger = get_logger(__name__)


class CandidateStatus(Enum):
    """候选材料状态"""
    PENDING = "pending"           # 待验证
    SELECTED = "selected"         # 已选中进行实验
    VALIDATED = "validated"       # 实验验证成功
    REJECTED = "rejected"         # 实验验证失败
    PROMISING = "promising"       # 有潜力（部分验证）


@dataclass
class ExperimentFeedback:
    """实验反馈数据"""
    material_id: str
    experiment_type: str          # LSV, CV, EIS, etc.
    metrics: Dict[str, float]     # 实验指标 {overpotential: 0.05, mass_activity: 2.5}
    status: CandidateStatus = CandidateStatus.PENDING
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "material_id": self.material_id,
            "experiment_type": self.experiment_type,
            "metrics": self.metrics,
            "status": self.status.value,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentFeedback":
        return cls(
            material_id=data["material_id"],
            experiment_type=data.get("experiment_type", "unknown"),
            metrics=data.get("metrics", {}),
            status=CandidateStatus(data.get("status", "pending")),
            notes=data.get("notes", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class IterationRound:
    """迭代轮次记录"""
    round_number: int
    timestamp: str
    query: str
    result: Optional[RecommendationResult] = None
    feedback: List[ExperimentFeedback] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "timestamp": self.timestamp,
            "query": self.query,
            "result": self.result.to_dict() if self.result else None,
            "feedback": [f.to_dict() for f in self.feedback],
        }


class IterativeSession:
    """
    多轮迭代会话管理器
    
    支持：
    - 多轮推荐迭代
    - 实验反馈集成
    - 候选状态跟踪
    - 会话持久化
    """
    
    def __init__(self, session_id: str = None, storage_dir: str = "data/sessions"):
        """
        初始化迭代会话
        
        Args:
            session_id: 会话ID，如果为空则自动生成
            storage_dir: 会话存储目录
        """
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage_dir = storage_dir
        self.created_at = datetime.now().isoformat()
        
        self.rounds: List[IterationRound] = []
        self.candidates: Dict[str, Dict[str, Any]] = {}  # {material_id: {status, scores, feedback}}
        self.orchestrator = AgentOrchestrator()
        
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"IterativeSession created: {self.session_id}")
    
    @property
    def current_round(self) -> int:
        """当前轮次"""
        return len(self.rounds)
    
    def start_recommendation(self, query: str) -> RecommendationResult:
        """
        开始新一轮推荐
        
        Args:
            query: 用户查询
            
        Returns:
            RecommendationResult: 推荐结果
        """
        round_number = self.current_round + 1
        logger.info(f"Starting round {round_number}: {query}")
        
        # 如果有之前的反馈，使用迭代模式
        if self.rounds and self.rounds[-1].feedback:
            result = self._iterate_with_feedback(query)
        else:
            result = self.orchestrator.orchestrate(query)
        
        # 记录轮次
        round_record = IterationRound(
            round_number=round_number,
            timestamp=datetime.now().isoformat(),
            query=query,
            result=result,
        )
        self.rounds.append(round_record)
        
        # 更新候选状态
        self._update_candidates(result)
        
        # 自动保存
        self.save()
        
        return result
    
    def _iterate_with_feedback(self, query: str) -> RecommendationResult:
        """基于反馈进行迭代推荐"""
        previous_result = self.rounds[-1].result
        feedback_data = {
            "query": query,
            "experiment_results": [f.to_dict() for f in self.rounds[-1].feedback],
            "validated_materials": [
                mid for mid, data in self.candidates.items()
                if data.get("status") == CandidateStatus.VALIDATED.value
            ],
            "rejected_materials": [
                mid for mid, data in self.candidates.items()
                if data.get("status") == CandidateStatus.REJECTED.value
            ],
        }
        
        return self.orchestrator.iterate_with_feedback(previous_result, feedback_data)
    
    def _update_candidates(self, result: RecommendationResult):
        """更新候选材料状态"""
        for cand in result.candidates:
            mat_id = cand.get("material_id")
            if not mat_id:
                continue
                
            if mat_id not in self.candidates:
                self.candidates[mat_id] = {
                    "status": CandidateStatus.PENDING.value,
                    "first_seen_round": self.current_round,
                    "scores": [],
                    "feedback": [],
                }
            
            self.candidates[mat_id]["scores"].append({
                "round": self.current_round,
                "score": cand.get("final_score", 0),
            })
    
    def add_experiment_feedback(
        self,
        material_id: str,
        experiment_type: str,
        metrics: Dict[str, float],
        status: CandidateStatus = CandidateStatus.PENDING,
        notes: str = ""
    ) -> bool:
        """
        添加实验反馈
        
        Args:
            material_id: 材料ID
            experiment_type: 实验类型 (LSV, CV, EIS)
            metrics: 实验指标
            status: 候选状态
            notes: 备注
            
        Returns:
            是否成功
        """
        if not self.rounds:
            logger.warning("No rounds yet, cannot add feedback")
            return False
        
        feedback = ExperimentFeedback(
            material_id=material_id,
            experiment_type=experiment_type,
            metrics=metrics,
            status=status,
            notes=notes,
        )
        
        # 添加到当前轮次
        self.rounds[-1].feedback.append(feedback)
        
        # 更新候选状态
        if material_id in self.candidates:
            self.candidates[material_id]["status"] = status.value
            self.candidates[material_id]["feedback"].append(feedback.to_dict())
        
        logger.info(f"Added experiment feedback for {material_id}: {status.value}")
        
        # 自动保存
        self.save()
        
        return True
    
    def get_candidates_by_status(self, status: CandidateStatus) -> List[str]:
        """获取指定状态的候选材料"""
        return [
            mid for mid, data in self.candidates.items()
            if data.get("status") == status.value
        ]
    
    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话摘要"""
        status_counts = {}
        for data in self.candidates.values():
            status = data.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "total_rounds": self.current_round,
            "total_candidates": len(self.candidates),
            "status_distribution": status_counts,
            "rounds": [
                {
                    "round": r.round_number,
                    "query": r.query,
                    "candidates": len(r.result.candidates) if r.result else 0,
                    "feedback_count": len(r.feedback),
                }
                for r in self.rounds
            ],
        }
    
    def save(self, path: str = None):
        """保存会话到文件"""
        path = path or os.path.join(self.storage_dir, f"{self.session_id}.json")
        
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "rounds": [r.to_dict() for r in self.rounds],
            "candidates": self.candidates,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Session saved to {path}")
    
    @classmethod
    def load(cls, session_id: str, storage_dir: str = "data/sessions") -> "IterativeSession":
        """从文件加载会话"""
        path = os.path.join(storage_dir, f"{session_id}.json")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        session = cls(session_id=data["session_id"], storage_dir=storage_dir)
        session.created_at = data["created_at"]
        session.candidates = data["candidates"]
        
        # 恢复轮次记录（简化，不完全恢复 result）
        for r_data in data["rounds"]:
            round_record = IterationRound(
                round_number=r_data["round_number"],
                timestamp=r_data["timestamp"],
                query=r_data["query"],
                feedback=[ExperimentFeedback.from_dict(f) for f in r_data.get("feedback", [])],
            )
            session.rounds.append(round_record)
        
        logger.info(f"Session loaded from {path}")
        return session
    
    @classmethod
    def list_sessions(cls, storage_dir: str = "data/sessions") -> List[str]:
        """列出所有会话"""
        if not os.path.exists(storage_dir):
            return []
        
        return [
            f.replace(".json", "")
            for f in os.listdir(storage_dir)
            if f.endswith(".json")
        ]


# 便捷函数
def create_session(query: str) -> tuple:
    """
    创建新会话并开始推荐
    
    Returns:
        (session, result)
    """
    session = IterativeSession()
    result = session.start_recommendation(query)
    return session, result


def continue_session(session_id: str, query: str = None) -> tuple:
    """
    继续现有会话
    
    Returns:
        (session, result or None)
    """
    session = IterativeSession.load(session_id)
    
    if query:
        result = session.start_recommendation(query)
        return session, result
    
    return session, None
