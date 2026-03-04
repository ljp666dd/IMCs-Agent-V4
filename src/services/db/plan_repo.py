from typing import List, Dict, Any, Optional
import sqlite3
import json
import os
from src.core.logger import get_logger, log_exception

logger = get_logger(__name__)

class PlanRepoMixin:
    # ========== Plans & Steps (v4.0) ==========

    def create_plan(self, plan_id: str, user_id: Optional[int], task_type: str, description: str):
        """Create a new execution plan record (idempotent)."""
        query = "INSERT OR IGNORE INTO plans (id, user_id, task_type, status, description) VALUES (?, ?, ?, ?, ?)"
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (plan_id, user_id, task_type, "pending", description))
            return plan_id

    def update_plan_status(self, plan_id: str, status: str):
        """Update plan status."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE plans SET status = ? WHERE id = ?", (status, plan_id))

    def update_plan_step_status(
        self,
        plan_id: str,
        step_id: str,
        status: str,
        result: Dict = None,
        error: str = None,
    ) -> None:
        """Append a plan step status update (preserve history)."""
        agent = "unknown"
        action = "update"
        params = None
        dependencies = None
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT agent, action, params, dependencies
                FROM plan_steps
                WHERE plan_id = ? AND step_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (plan_id, step_id),
            )
            row = cursor.fetchone()
            if row:
                agent = row["agent"] or agent
                action = row["action"] or action
                if row["params"]:
                    try:
                        params = json.loads(row["params"])
                    except Exception:
                        params = None
                if row["dependencies"]:
                    try:
                        dependencies = json.loads(row["dependencies"])
                    except Exception:
                        dependencies = None

        self.log_plan_step(
            plan_id=plan_id,
            step_id=step_id,
            agent=agent,
            action=action,
            status=status,
            result=result,
            error=error,
            dependencies=dependencies,
            params=params,
        )

    def log_plan_step(self, plan_id: str, step_id: str, agent: str, action: str,
                      status: str, result: Dict = None, error: str = None,
                      dependencies: Optional[List[str]] = None,
                      params: Optional[Dict] = None):
        """Log a step execution (Insert or Update)."""
        # We use INSERT OR REPLACE if ID is step_id? No, step_id is text string 'step_1'.
        # We should use INSERT for history log, but for current state capturing 'latest' is fine.
        # Given we want trace, simple INSERT is best. To avoid clutter, maybe check existence?
        # Actually, `plan_steps` has an auto-increment ID.
        # For simplicity in v4 Pilot, just INSERT log entry.
        # 采用追加式日志, 便于追踪每次状态变化
        
        query = """
        INSERT INTO plan_steps (plan_id, step_id, agent, action, params, dependencies, status, result, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        deps_json = json.dumps(dependencies or []) if dependencies is not None else None
        params_json = json.dumps(params) if params is not None else None
        result_json = None
        if result is not None:
            try:
                result_json = json.dumps(result, default=str)
            except Exception:
                result_json = json.dumps(str(result))
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (plan_id, step_id, agent, action, params_json, deps_json, status, result_json, error))

    def get_plan(self, plan_id: str) -> Optional[Dict]:
        """Get a plan by id."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM plans WHERE id = ?", (plan_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_plans(
        self,
        limit: int = 50,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[Dict]:
        """List plans ordered by created_at desc."""
        where: List[str] = []
        params: List[Any] = []
        if status:
            where.append("status = ?")
            params.append(status)
        if task_type:
            where.append("task_type = ?")
            params.append(task_type)
        query = "SELECT * FROM plans"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit or 50))

        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_plan_last_step_created_at(self, plan_id: str) -> Optional[str]:
        """Return the last plan_steps created_at for a plan (best-effort)."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(created_at) AS last_ts FROM plan_steps WHERE plan_id = ?",
                (plan_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return row[0]

    def list_plan_steps(self, plan_id: str) -> List[Dict]:
        """List plan steps for a given plan id (ordered)."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY created_at ASC", (plan_id,))
            rows = cursor.fetchall()
            steps = []
            for row in rows:
                data = dict(row)
                if data.get("result"):
                    try:
                        data["result"] = json.loads(data["result"])
                    except Exception:
                        pass
                if data.get("params"):
                    try:
                        data["params"] = json.loads(data["params"])
                    except Exception:
                        pass
                if data.get("dependencies"):
                    try:
                        data["dependencies"] = json.loads(data["dependencies"])
                    except Exception:
                        pass
                steps.append(data)
            return steps

    def list_latest_plan_steps(self, plan_id: str) -> List[Dict]:
        """List the latest row per step_id for a given plan id.

        plan_steps is append-only, so callers often need the "current state"
        per step_id. This avoids re-implementing reduce-to-latest logic across
        API/UI code paths.
        """
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ps.*
                FROM plan_steps ps
                JOIN (
                    SELECT step_id, MAX(id) AS max_id
                    FROM plan_steps
                    WHERE plan_id = ?
                    GROUP BY step_id
                ) latest
                ON ps.step_id = latest.step_id AND ps.id = latest.max_id
                WHERE ps.plan_id = ?
                """,
                (plan_id, plan_id),
            )
            rows = cursor.fetchall()
            steps = []
            for row in rows:
                data = dict(row)
                if data.get("result"):
                    try:
                        data["result"] = json.loads(data["result"])
                    except Exception:
                        pass
                if data.get("params"):
                    try:
                        data["params"] = json.loads(data["params"])
                    except Exception:
                        pass
                if data.get("dependencies"):
                    try:
                        data["dependencies"] = json.loads(data["dependencies"])
                    except Exception:
                        pass
                steps.append(data)

            def _step_rank(step_id: str) -> int:
                if isinstance(step_id, str) and step_id.startswith("step_"):
                    try:
                        return int(step_id.split("_", 1)[1])
                    except Exception:
                        return 10**9
                return 10**9

            steps.sort(key=lambda s: _step_rank(s.get("step_id")))
            return steps

