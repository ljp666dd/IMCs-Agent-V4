# -*- coding: utf-8 -*-
"""
IMCs Conflict Detector — 多智能体冲突检测与辩论模块

检测各 Agent 推荐结果之间的分歧与矛盾：
1. 排名冲突: Agent A 认为 X 最好, Agent B 认为 X 最差
2. 方向冲突: Theory 认为某材料好, ML 认为差 (正负号不同)
3. 置信度分歧: 高分但高不确定度
4. 缺证冲突: 只有一个 Agent 支持, 无交叉验证

输出结构化辩论日志, 供 Orchestrator 参考。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from src.agents.protocol import AgentContribution
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Conflict:
    """单条冲突记录"""
    conflict_type: str          # rank_conflict | direction_conflict | confidence_gap | solo_evidence
    material_id: str
    formula: str
    agents_involved: List[str]
    severity: str               # HIGH | MEDIUM | LOW
    description: str
    scores: Dict[str, float] = field(default_factory=dict)
    resolution: str = ""        # 建议处理方式


@dataclass
class DebateRecord:
    """辩论日志"""
    query: str
    total_conflicts: int = 0
    conflicts: List[Conflict] = field(default_factory=list)
    consensus_materials: List[str] = field(default_factory=list)  # 全票通过的材料
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_conflicts": self.total_conflicts,
            "conflicts": [
                {
                    "type": c.conflict_type,
                    "material_id": c.material_id,
                    "formula": c.formula,
                    "agents": c.agents_involved,
                    "severity": c.severity,
                    "description": c.description,
                    "scores": c.scores,
                    "resolution": c.resolution,
                }
                for c in self.conflicts
            ],
            "consensus_materials": self.consensus_materials,
            "summary": self.summary,
        }


class ConflictDetector:
    """多智能体冲突检测引擎"""

    def detect(
        self,
        contributions: Dict[str, AgentContribution],
        query: str = "",
    ) -> DebateRecord:
        """
        从各 Agent 的贡献中检测冲突

        Returns:
            DebateRecord: 辩论记录
        """
        debate = DebateRecord(query=query)

        # 构造 material→agent→score 映射
        mat_scores: Dict[str, Dict[str, float]] = {}
        mat_formulas: Dict[str, str] = {}

        for agent_name, contrib in contributions.items():
            if not contrib.success:
                continue
            for cand in contrib.candidates:
                mid = cand.get("material_id", "")
                if not mid:
                    continue
                mat_formulas[mid] = cand.get("formula", mid)

                # 尝试获取分数
                score = (
                    cand.get("predicted_activity")
                    or cand.get("exchange_current_density")
                    or cand.get("score")
                    or contrib.confidence
                )
                if score is not None:
                    mat_scores.setdefault(mid, {})[agent_name] = float(score)

        # 1. 排名冲突检测
        self._detect_rank_conflicts(mat_scores, mat_formulas, contributions, debate)

        # 2. 方向冲突检测 (有正有负)
        self._detect_direction_conflicts(mat_scores, mat_formulas, debate)

        # 3. 置信度分歧检测
        self._detect_confidence_gaps(contributions, mat_formulas, debate)

        # 4. 孤证检测
        self._detect_solo_evidence(mat_scores, mat_formulas, debate)

        # 5. 共识材料
        multi_agents = {
            mid for mid, agents in mat_scores.items()
            if len(agents) >= 2
        }
        conflict_mats = {c.material_id for c in debate.conflicts}
        debate.consensus_materials = list(multi_agents - conflict_mats)

        debate.total_conflicts = len(debate.conflicts)
        debate.summary = self._generate_summary(debate)

        logger.info(
            f"ConflictDetector: {debate.total_conflicts} conflicts, "
            f"{len(debate.consensus_materials)} consensus materials"
        )
        return debate

    def _detect_rank_conflicts(
        self,
        mat_scores: Dict[str, Dict[str, float]],
        mat_formulas: Dict[str, str],
        contributions: Dict[str, AgentContribution],
        debate: DebateRecord,
    ):
        """检测不同 Agent 对同一材料的排名是否存在严重分歧"""
        # 构建每个 Agent 的排名列表
        agent_rankings: Dict[str, List[str]] = {}
        for agent_name, contrib in contributions.items():
            if not contrib.success or not contrib.candidates:
                continue
            ranked = [c.get("material_id", "") for c in contrib.candidates if c.get("material_id")]
            if ranked:
                agent_rankings[agent_name] = ranked

        if len(agent_rankings) < 2:
            return

        # 对于出现在多个 Agent 中的材料, 检测排名差异
        for mid, agents_scores in mat_scores.items():
            if len(agents_scores) < 2:
                continue

            positions = {}
            for agent_name, ranking in agent_rankings.items():
                if mid in ranking:
                    idx = ranking.index(mid)
                    total = len(ranking)
                    # 归一化排名 (0=最好, 1=最差)
                    positions[agent_name] = idx / max(total - 1, 1)

            if len(positions) < 2:
                continue

            # 如果一个 Agent 把它排前 20%, 另一个排后 80%, 则为冲突
            pos_vals = list(positions.values())
            pos_min = min(pos_vals)
            pos_max = max(pos_vals)

            if pos_max - pos_min > 0.6:
                top_agent = min(positions, key=positions.get)
                bottom_agent = max(positions, key=positions.get)
                debate.conflicts.append(Conflict(
                    conflict_type="rank_conflict",
                    material_id=mid,
                    formula=mat_formulas.get(mid, mid),
                    agents_involved=[top_agent, bottom_agent],
                    severity="HIGH" if pos_max - pos_min > 0.8 else "MEDIUM",
                    description=(
                        f"{top_agent} 将 {mat_formulas.get(mid, mid)} 排在前 {pos_min*100:.0f}%, "
                        f"而 {bottom_agent} 将其排在后 {pos_max*100:.0f}%"
                    ),
                    scores=agents_scores,
                    resolution="建议仔细对比两个 Agent 的评估依据, 或请求验证实验",
                ))

    def _detect_direction_conflicts(
        self,
        mat_scores: Dict[str, Dict[str, float]],
        mat_formulas: Dict[str, str],
        debate: DebateRecord,
    ):
        """检测对同一材料, 不同 Agent 给出正/负相反方向的评分"""
        for mid, agents_scores in mat_scores.items():
            if len(agents_scores) < 2:
                continue

            vals = list(agents_scores.values())
            has_positive = any(v > 0 for v in vals)
            has_negative = any(v < 0 for v in vals)

            if has_positive and has_negative:
                pos_agents = [a for a, v in agents_scores.items() if v > 0]
                neg_agents = [a for a, v in agents_scores.items() if v < 0]
                debate.conflicts.append(Conflict(
                    conflict_type="direction_conflict",
                    material_id=mid,
                    formula=mat_formulas.get(mid, mid),
                    agents_involved=pos_agents + neg_agents,
                    severity="HIGH",
                    description=(
                        f"对 {mat_formulas.get(mid, mid)}: "
                        f"{pos_agents} 给出正向评价, "
                        f"{neg_agents} 给出负向评价"
                    ),
                    scores=agents_scores,
                    resolution="方向性矛盾需优先排查, 可能存在数据源或特征不一致",
                ))

    def _detect_confidence_gaps(
        self,
        contributions: Dict[str, AgentContribution],
        mat_formulas: Dict[str, str],
        debate: DebateRecord,
    ):
        """检测高分但高不确定度的候选"""
        for agent_name, contrib in contributions.items():
            if not contrib.success:
                continue
            for cand in contrib.candidates:
                mid = cand.get("material_id", "")
                unc = cand.get("uncertainty", 0)
                score = cand.get("predicted_activity", 0)

                # 高分 + 高不确定度
                if abs(score) > 0 and unc > 0 and unc / max(abs(score), 1e-6) > 0.5:
                    debate.conflicts.append(Conflict(
                        conflict_type="confidence_gap",
                        material_id=mid,
                        formula=mat_formulas.get(mid, cand.get("formula", mid)),
                        agents_involved=[agent_name],
                        severity="MEDIUM",
                        description=(
                            f"{agent_name} 对 {mat_formulas.get(mid, mid)} 给出 "
                            f"score={score:.4f} 但 uncertainty={unc:.4f} "
                            f"(不确定度/分数比 = {unc/max(abs(score),1e-6):.1%})"
                        ),
                        scores={agent_name: score, f"{agent_name}_unc": unc},
                        resolution="建议通过实验验证或增加训练数据降低不确定度",
                    ))

    def _detect_solo_evidence(
        self,
        mat_scores: Dict[str, Dict[str, float]],
        mat_formulas: Dict[str, str],
        debate: DebateRecord,
    ):
        """检测只有单一 Agent 支持的材料 (孤证)"""
        for mid, agents_scores in mat_scores.items():
            if len(agents_scores) == 1:
                sole_agent = list(agents_scores.keys())[0]
                debate.conflicts.append(Conflict(
                    conflict_type="solo_evidence",
                    material_id=mid,
                    formula=mat_formulas.get(mid, mid),
                    agents_involved=[sole_agent],
                    severity="LOW",
                    description=(
                        f"{mat_formulas.get(mid, mid)} 仅由 {sole_agent} 推荐, "
                        f"缺少交叉验证支持"
                    ),
                    scores=agents_scores,
                    resolution="孤证不推荐, 建议补充其他 Agent 的评估",
                ))

    def _generate_summary(self, debate: DebateRecord) -> str:
        """生成辩论摘要"""
        if debate.total_conflicts == 0:
            return "各智能体推荐高度一致, 无显著冲突。"

        high = sum(1 for c in debate.conflicts if c.severity == "HIGH")
        med = sum(1 for c in debate.conflicts if c.severity == "MEDIUM")
        low = sum(1 for c in debate.conflicts if c.severity == "LOW")

        parts = []
        if high:
            parts.append(f"{high} 项严重冲突")
        if med:
            parts.append(f"{med} 项中度分歧")
        if low:
            parts.append(f"{low} 项轻微(孤证)")

        summary = f"检测到 {debate.total_conflicts} 项冲突: {', '.join(parts)}。"
        if debate.consensus_materials:
            summary += f" 共识材料 {len(debate.consensus_materials)} 个。"
        return summary
