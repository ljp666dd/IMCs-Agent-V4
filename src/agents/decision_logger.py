# -*- coding: utf-8 -*-
"""
IMCs Decision Logger — 决策链持久化服务

将每次推荐流程的完整决策链持久化到数据库，包括：
- 用户查询与解析意图
- 各 Agent 的能力评估和贡献
- 冲突检测结果
- 融合排名 & 专家报告
- Active Learning 触发记录

支持事后追溯、审计和迭代优化分析。
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from src.services.db.database import DatabaseService
from src.core.logger import get_logger

logger = get_logger(__name__)


class DecisionLogger:
    """决策链持久化服务"""

    def __init__(self, db: Optional[DatabaseService] = None):
        self.db = db or DatabaseService()
        self._ensure_table()

    def _ensure_table(self):
        """确保 decision_logs 表存在"""
        try:
            with self.db._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS decision_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        step_name TEXT NOT NULL,
                        step_order INTEGER DEFAULT 0,
                        data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_decision_logs_session
                    ON decision_logs(session_id)
                """)
        except Exception as e:
            logger.warning(f"Could not create decision_logs table: {e}")

    def create_session(self, query: str) -> str:
        """创建新的决策会话，返回 session_id"""
        session_id = f"dec-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self._log(session_id, "query", 0, {
            "user_query": query,
            "timestamp": datetime.now().isoformat(),
        })
        return session_id

    def log_intent(self, session_id: str, parsed_intent: Dict):
        """记录用户意图解析"""
        self._log(session_id, "intent_parsed", 1, parsed_intent)

    def log_capabilities(self, session_id: str, capabilities: Dict[str, Any]):
        """记录各 Agent 能力评估"""
        cap_data = {}
        for name, cap in capabilities.items():
            if hasattr(cap, '__dict__'):
                cap_data[name] = {
                    "can_contribute": getattr(cap, "can_contribute", False),
                    "confidence": getattr(cap, "confidence", 0),
                    "estimated_time": getattr(cap, "estimated_time_seconds", 0),
                }
            else:
                cap_data[name] = str(cap)
        self._log(session_id, "capabilities", 2, cap_data)

    def log_contribution(self, session_id: str, agent_name: str, contribution):
        """记录单个 Agent 贡献"""
        contrib_data = {
            "agent_name": agent_name,
            "success": contribution.success,
            "num_candidates": len(contribution.candidates),
            "num_insights": len(contribution.insights),
            "confidence": contribution.confidence,
            "reasoning": contribution.reasoning,
        }
        self._log(session_id, f"contribution_{agent_name}", 3, contrib_data)

    def log_conflicts(self, session_id: str, debate_record):
        """记录冲突检测结果"""
        self._log(session_id, "conflicts", 4, debate_record.to_dict())

    def log_fusion(self, session_id: str, top_candidates: List[Dict], explanations: List = None):
        """记录融合排名结果"""
        fusion_data = {
            "num_candidates": len(top_candidates),
            "top_5": [
                {
                    "material_id": c.get("material_id"),
                    "formula": c.get("formula"),
                    "final_score": c.get("final_score"),
                }
                for c in top_candidates[:5]
            ],
            "num_explanations": len(explanations) if explanations else 0,
        }
        self._log(session_id, "fusion", 5, fusion_data)

    def log_active_learning(self, session_id: str, triggered: bool, requests: List[Dict] = None):
        """记录主动学习触发"""
        al_data = {
            "triggered": triggered,
            "num_requests": len(requests) if requests else 0,
            "intercepted": [
                {
                    "material_id": r.get("material_id"),
                    "formula": r.get("formula"),
                    "reason": r.get("active_learning_reason"),
                }
                for r in (requests or [])
            ],
        }
        self._log(session_id, "active_learning", 6, al_data)

    def log_final(self, session_id: str, result):
        """记录最终推荐结果"""
        final_data = {
            "success": result.success,
            "num_candidates": len(result.candidates),
            "iteration": result.iteration,
            "execution_order": result.execution_order,
            "reasoning_length": len(result.reasoning),
        }
        self._log(session_id, "final_result", 7, final_data)

    def get_session_log(self, session_id: str) -> List[Dict]:
        """获取完整决策链日志"""
        try:
            with self.db._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT step_name, step_order, data, created_at "
                    "FROM decision_logs WHERE session_id = ? ORDER BY step_order, id",
                    (session_id,)
                )
                return [
                    {
                        "step": row[0],
                        "order": row[1],
                        "data": json.loads(row[2]) if row[2] else {},
                        "timestamp": row[3],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to read decision log: {e}")
            return []

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        """列出最近的决策会话"""
        try:
            with self.db._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT DISTINCT session_id, MIN(created_at) as started, "
                    "COUNT(*) as steps FROM decision_logs "
                    "GROUP BY session_id ORDER BY started DESC LIMIT ?",
                    (limit,)
                )
                return [
                    {"session_id": row[0], "started": row[1], "steps": row[2]}
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def _log(self, session_id: str, step_name: str, step_order: int, data: Any):
        """写入单条日志"""
        try:
            data_json = json.dumps(data, ensure_ascii=False, default=str)
            with self.db._get_conn() as conn:
                conn.execute(
                    "INSERT INTO decision_logs (session_id, step_name, step_order, data) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, step_name, step_order, data_json)
                )
            logger.debug(f"Decision log: {session_id}/{step_name}")
        except Exception as e:
            logger.warning(f"Failed to log decision step {step_name}: {e}")
