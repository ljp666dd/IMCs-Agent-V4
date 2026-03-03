import sqlite3
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from src.core.logger import get_logger, log_exception
from src.config.config import config

logger = get_logger(__name__)

class DatabaseService:
    """
    Service for SQLite database interactions.
    Manages Materials, Experiments, and Models.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DB_PATH
        self._init_db()
        
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("PRAGMA foreign_keys=ON;")
        except Exception:
            pass
        return conn

    def _repo_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    def _resolve_cif_path(self, cif_path: str) -> Optional[str]:
        if not cif_path:
            return None
        path = cif_path
        if not os.path.isabs(path):
            path = os.path.abspath(os.path.join(self._repo_root(), path))
        data_root = os.path.abspath(os.path.join(self._repo_root(), config.DATA_DIR))
        try:
            if os.path.commonpath([path, data_root]) != data_root:
                return None
        except Exception:
            return None
        return path if os.path.exists(path) else None

    def _read_cif_content(self, cif_path: Optional[str]) -> Optional[str]:
        safe_path = self._resolve_cif_path(cif_path or "")
        if not safe_path:
            return None
        try:
            with open(safe_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    def _filter_material_records(self, records: List[Dict], allowed_elements: Optional[List[str]] = None) -> List[Dict]:
        """Filter material records by allowed elements (formula subset)."""
        if not allowed_elements:
            return records
        allowed = set(allowed_elements)
        try:
            from pymatgen.core import Composition
        except Exception:
            return records
        filtered = []
        for rec in records:
            formula = rec.get("formula") or rec.get("formula_pretty")
            if not formula:
                continue
            try:
                elements = {el.symbol for el in Composition(formula).elements}
            except Exception:
                continue
            if elements.issubset(allowed):
                filtered.append(rec)
        if not filtered and records:
            logger.warning("No materials matched allowed elements; returning unfiltered records.")
            return records
        return filtered

    def _allowed_material_ids(self, allowed_elements: Optional[List[str]] = None) -> Optional[set]:
        """Return allowed material_id set based on formula filtering."""
        if not allowed_elements:
            return None
        allowed = set(allowed_elements)
        try:
            from pymatgen.core import Composition
        except Exception:
            return None
        material_ids = set()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT material_id, formula FROM materials")
            for material_id, formula in cursor.fetchall():
                if not material_id or not formula:
                    continue
                try:
                    elements = {el.symbol for el in Composition(formula).elements}
                except Exception:
                    continue
                if elements.issubset(allowed):
                    material_ids.add(material_id)
        return material_ids
        
    @log_exception(logger)
    def _init_db(self):
        """Initialize database with schema."""
        # Load schema from file
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        if not os.path.exists(schema_path):
            logger.error(f"Schema not found at {schema_path}")
            return
            
        with open(schema_path, 'r') as f:
            schema = f.read()
            
        with self._get_conn() as conn:
            conn.executescript(schema)
        logger.info(f"Database initialized at {self.db_path}")

    # ========== Users (v4.0) ==========

    def create_user(self, username: str, password_hash: str) -> Optional[int]:
        """Create a new user."""
        query = "INSERT INTO users (username, password_hash) VALUES (?, ?)"
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (username, password_hash))
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None # Duplicate username

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========== Chat Sessions & Messages (v4.0) ==========

    def create_chat_session(self, title: str, user_id: Optional[int] = None) -> int:
        """Create a chat session and return its id."""
        query = "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)"
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (user_id, title))
            return cursor.lastrowid

    def list_chat_sessions(self, limit: int = 50, user_id: Optional[int] = None) -> List[Dict]:
        """List chat sessions ordered by last update."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if user_id is None:
                cursor.execute(
                    "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                    (user_id, limit),
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_chat_session(self, session_id: int) -> Optional[Dict]:
        """Get a chat session by id."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_chat_session_title(self, session_id: int, title: str):
        """Update chat session title."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, session_id),
            )

    def touch_chat_session(self, session_id: int):
        """Touch chat session updated_at."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )

    def add_chat_message(self, session_id: int, role: str, content: str, artifacts: Dict = None) -> int:
        """Add a chat message to a session."""
        artifacts_json = json.dumps(artifacts) if artifacts else None
        query = """
        INSERT INTO chat_messages (session_id, role, content, artifacts)
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (session_id, role, content, artifacts_json))
            msg_id = cursor.lastrowid
            cursor.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
            return msg_id

    def list_chat_messages(self, session_id: int) -> List[Dict]:
        """List chat messages for a session."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            )
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                data = dict(row)
                if data.get("artifacts"):
                    try:
                        data["artifacts"] = json.loads(data["artifacts"])
                    except Exception:
                        pass
                messages.append(data)
            return messages

    def delete_chat_session(self, session_id: int):
        """Delete a chat session and its messages."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

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

    # ========== Materials ==========
    
    @log_exception(logger)
    def save_material(self, material_id: str, formula: str, 
                      energy: float = None, cif_path: str = None) -> int:
        """Save theoretical material data."""
        query = """
        INSERT INTO materials (material_id, formula, formation_energy, cif_path)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(material_id) DO UPDATE SET
            formula = COALESCE(excluded.formula, materials.formula),
            formation_energy = COALESCE(excluded.formation_energy, materials.formation_energy),
            cif_path = COALESCE(excluded.cif_path, materials.cif_path)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (material_id, formula, energy, cif_path))
            return cursor.lastrowid

    def _normalize_material_id(self, material_id: Optional[str]) -> Optional[str]:
        mid = (str(material_id) if material_id is not None else "").strip()
        return mid or None

    def _guess_stub_formula(self, material_id: str, formula: Optional[str] = None) -> str:
        if formula is not None:
            val = str(formula).strip()
            if val:
                return val

        mid = (str(material_id) if material_id is not None else "").strip()
        if not mid:
            return "UNKNOWN"
        if mid.startswith("fake:"):
            rest = mid.split("fake:", 1)[1].strip()
            return rest or "UNKNOWN"
        if mid.startswith("mp-"):
            return "UNKNOWN"
        for ch in (":", "/", "\\", " ", "-"):
            if ch in mid:
                return "UNKNOWN"
        return mid

    def ensure_material_stub(self, material_id: str, formula: Optional[str] = None) -> Optional[str]:
        """
        Ensure a materials row exists for the given material_id without overwriting
        any existing formula/fields.

        This is used to prevent FK errors when external systems (robot/importers)
        report metrics for material_ids that are not yet present in the local DB.
        """
        mid = self._normalize_material_id(material_id)
        if not mid:
            return None
        stub_formula = self._guess_stub_formula(mid, formula=formula)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO materials (material_id, formula, formation_energy, cif_path)
                VALUES (?, ?, NULL, NULL)
                ON CONFLICT(material_id) DO NOTHING
                """,
                (mid, stub_formula),
            )
        return mid
            
    def list_materials(self, limit: int = 100, allowed_elements: Optional[List[str]] = None, require_cif: bool = False) -> List[Dict]:
        """List stored materials."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM materials"
            if require_cif:
                query += " WHERE cif_path IS NOT NULL"
            query += " ORDER BY created_at DESC LIMIT ?"
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
            return self._filter_material_records(records, allowed_elements)

    def list_materials_since(self, created_at: str, limit: int = 100, allowed_elements: Optional[List[str]] = None) -> List[Dict]:
        """List materials created since a timestamp."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM materials WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (created_at, limit),
            )
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
            return self._filter_material_records(records, allowed_elements)

    def list_materials_by_ids(
        self,
        material_ids: List[str],
        allowed_elements: Optional[List[str]] = None,
    ) -> List[Dict]:
        """List material records by material_ids (preserve input order)."""
        if not material_ids:
            return []

        ordered: List[str] = []
        seen = set()
        for mid in material_ids:
            mid = (str(mid) if mid is not None else "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            ordered.append(mid)

        if not ordered:
            return []

        placeholders = ",".join(["?"] * len(ordered))
        query = f"SELECT * FROM materials WHERE material_id IN ({placeholders})"
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, tuple(ordered))
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
            records = self._filter_material_records(records, allowed_elements)
            by_id = {r.get("material_id"): r for r in records if r.get("material_id")}
            return [by_id[mid] for mid in ordered if mid in by_id]
            
    def get_material_by_id(self, material_id: str) -> Optional[Dict]:
        """Get material details including CIF content."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM materials WHERE material_id = ?", (material_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            data = dict(row)
            # Read CIF content if path exists
            data["cif_content"] = self._read_cif_content(data.get("cif_path"))
                
            return data

    def get_material_with_evidence(self, material_id: str, include_cif: bool = False) -> Optional[Dict]:
        """Get material details with evidence (optionally include CIF content)."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM materials WHERE material_id = ?", (material_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            if include_cif:
                data["cif_content"] = self._read_cif_content(data.get("cif_path"))
            
            # Auto-parse DOS data
            if data.get("dos_data") and isinstance(data["dos_data"], str):
                try:
                    data["dos_data"] = json.loads(data["dos_data"])
                except Exception:
                    pass

            data["evidence"] = self.get_evidence_for_material(material_id)
            # 附加吸附能记录(若存在)
            data["adsorption_energies"] = self.list_adsorption_energies(material_id)
            data["activity_metrics"] = self.list_activity_metrics(material_id)
            return data

    # ========== Evidence & Coverage ==========

    def get_evidence_stats(self, allowed_elements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return system-wide evidence coverage stats."""
        stats: Dict[str, Any] = {}
        allowed_ids = self._allowed_material_ids(allowed_elements)
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if allowed_ids is None:
                cursor.execute("SELECT COUNT(*) AS total FROM materials")
                stats["total_materials"] = cursor.fetchone()["total"] or 0

                cursor.execute("SELECT COUNT(*) AS total FROM materials WHERE formation_energy IS NOT NULL")
                stats["formation_energy_count"] = cursor.fetchone()["total"] or 0

                cursor.execute("SELECT COUNT(*) AS total FROM materials WHERE dos_data IS NOT NULL")
                stats["dos_count"] = cursor.fetchone()["total"] or 0

                cursor.execute("SELECT source_type, COUNT(DISTINCT material_id) AS cnt FROM evidence GROUP BY source_type")
                stats["evidence_by_source"] = {row["source_type"]: row["cnt"] for row in cursor.fetchall()}

                cursor.execute("SELECT COUNT(*) AS total FROM adsorption_energies")
                stats["adsorption_rows"] = cursor.fetchone()["total"] or 0
                cursor.execute("SELECT COUNT(DISTINCT material_id) AS total FROM adsorption_energies WHERE material_id IS NOT NULL")
                stats["adsorption_materials"] = cursor.fetchone()["total"] or 0

                cursor.execute("SELECT COUNT(*) AS total FROM activity_metrics")
                stats["activity_rows"] = cursor.fetchone()["total"] or 0
                cursor.execute("SELECT COUNT(DISTINCT material_id) AS total FROM activity_metrics WHERE material_id IS NOT NULL")
                stats["activity_materials"] = cursor.fetchone()["total"] or 0
            else:
                cursor.execute("SELECT material_id, formation_energy, dos_data FROM materials")
                total = 0
                fe_count = 0
                dos_count = 0
                for row in cursor.fetchall():
                    mid = row["material_id"]
                    if mid not in allowed_ids:
                        continue
                    total += 1
                    if row["formation_energy"] is not None:
                        fe_count += 1
                    if row["dos_data"] is not None:
                        dos_count += 1
                stats["total_materials"] = total
                stats["formation_energy_count"] = fe_count
                stats["dos_count"] = dos_count

                cursor.execute("SELECT material_id, source_type FROM evidence")
                ev_map: Dict[str, set] = {}
                for row in cursor.fetchall():
                    mid = row["material_id"]
                    if mid not in allowed_ids:
                        continue
                    stype = row["source_type"] or "unknown"
                    ev_map.setdefault(stype, set()).add(mid)
                stats["evidence_by_source"] = {k: len(v) for k, v in ev_map.items()}

                cursor.execute("SELECT material_id FROM adsorption_energies WHERE material_id IS NOT NULL")
                ads_rows = 0
                ads_ids = set()
                for (mid,) in cursor.fetchall():
                    if mid in allowed_ids:
                        ads_rows += 1
                        ads_ids.add(mid)
                stats["adsorption_rows"] = ads_rows
                stats["adsorption_materials"] = len(ads_ids)

                cursor.execute("SELECT material_id FROM activity_metrics WHERE material_id IS NOT NULL")
                act_rows = 0
                act_ids = set()
                for (mid,) in cursor.fetchall():
                    if mid in allowed_ids:
                        act_rows += 1
                        act_ids.add(mid)
                stats["activity_rows"] = act_rows
                stats["activity_materials"] = len(act_ids)

            cursor.execute("SELECT COUNT(*) AS total FROM models")
            stats["model_count"] = cursor.fetchone()["total"] or 0

        return stats

    def get_data_integrity_stats(self) -> Dict[str, Any]:
        """
        Lightweight data integrity / orphan-row checks.

        Rationale:
        - In SQLite, foreign key enforcement is connection-scoped and must be
          explicitly enabled. We also want visibility into any legacy orphan
          rows that were inserted before enforcement was turned on.
        """
        stats: Dict[str, Any] = {}
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys;")
                row = cursor.fetchone()
                stats["foreign_keys_enabled"] = bool(row and int(row[0]) == 1)
            except Exception:
                stats["foreign_keys_enabled"] = None

            # Evidence (material_id is NOT NULL)
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM evidence e
                LEFT JOIN materials m ON e.material_id = m.material_id
                WHERE m.material_id IS NULL
                """
            )
            stats["evidence_orphan_rows"] = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT COUNT(DISTINCT e.material_id)
                FROM evidence e
                LEFT JOIN materials m ON e.material_id = m.material_id
                WHERE m.material_id IS NULL
                """
            )
            stats["evidence_orphan_material_ids"] = int(cursor.fetchone()[0] or 0)

            # Activity metrics (material_id is nullable)
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM activity_metrics a
                LEFT JOIN materials m ON a.material_id = m.material_id
                WHERE a.material_id IS NOT NULL AND m.material_id IS NULL
                """
            )
            stats["activity_metric_orphan_rows"] = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT COUNT(DISTINCT a.material_id)
                FROM activity_metrics a
                LEFT JOIN materials m ON a.material_id = m.material_id
                WHERE a.material_id IS NOT NULL AND m.material_id IS NULL
                """
            )
            stats["activity_metric_orphan_material_ids"] = int(cursor.fetchone()[0] or 0)

            # Adsorption energies (material_id is nullable)
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM adsorption_energies ae
                LEFT JOIN materials m ON ae.material_id = m.material_id
                WHERE ae.material_id IS NOT NULL AND m.material_id IS NULL
                """
            )
            stats["adsorption_orphan_rows"] = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT COUNT(DISTINCT ae.material_id)
                FROM adsorption_energies ae
                LEFT JOIN materials m ON ae.material_id = m.material_id
                WHERE ae.material_id IS NOT NULL AND m.material_id IS NULL
                """
            )
            stats["adsorption_orphan_material_ids"] = int(cursor.fetchone()[0] or 0)

        return stats

    def get_material_by_formula(self, formula: str) -> Optional[Dict]:
        """Get material by formula (Exact Match)."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM materials WHERE formula = ?", (formula,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========== M3: Evidence Chain ==========

    def save_evidence(self, material_id: str, source_type: str, source_id: str, score: float = 1.0, metadata: Dict = None) -> int:
        """
        Save evidence linking a material to a source (Lit, Exp, ML).
        """
        query = """
        INSERT INTO evidence (material_id, source_type, source_id, score, metadata)
        VALUES (?, ?, ?, ?, ?)
        """
        meta_json = json.dumps(metadata) if metadata else None
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (material_id, source_type, source_id, score, meta_json))
            evid_id = cursor.lastrowid

        # Best-effort sync into Knowledge Core
        try:
            from src.services.knowledge import KnowledgeService
            ks = KnowledgeService(self.db_path)
            ks.upsert_material_evidence(
                material_id=material_id,
                source_type=source_type,
                source_id=source_id,
                score=score,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Knowledge sync failed: {e}")

        return evid_id

    def get_evidence_for_material(self, material_id: str) -> List[Dict]:
        """Get all evidence linked to a material."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evidence WHERE material_id = ? ORDER BY created_at DESC", (material_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_evidence_counts(self, material_ids: List[str]) -> Dict[str, Dict[str, int]]:
        """Get evidence counts by source_type for each material_id."""
        if not material_ids:
            return {}
        placeholders = ",".join(["?"] * len(material_ids))
        counts: Dict[str, Dict[str, int]] = {mid: {} for mid in material_ids}
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT material_id, source_type, COUNT(*) AS cnt
                FROM evidence
                WHERE material_id IN ({placeholders})
                GROUP BY material_id, source_type
                """,
                material_ids,
            )
            for row in cursor.fetchall():
                mid = row["material_id"]
                stype = row["source_type"] or "unknown"
                counts.setdefault(mid, {})[stype] = row["cnt"] or 0
        return counts

    def get_material_feature_flags(self, material_ids: List[str]) -> Dict[str, Dict[str, bool]]:
        """Return material-level feature flags for evidence gap analysis."""
        if not material_ids:
            return {}
        placeholders = ",".join(["?"] * len(material_ids))
        flags: Dict[str, Dict[str, bool]] = {
            mid: {"formation_energy": False, "dos_data": False} for mid in material_ids
        }
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT material_id, formation_energy, dos_data
                FROM materials
                WHERE material_id IN ({placeholders})
                """,
                material_ids,
            )
            for row in cursor.fetchall():
                mid = row["material_id"]
                if not mid:
                    continue
                flags[mid] = {
                    "formation_energy": row["formation_energy"] is not None,
                    "dos_data": row["dos_data"] is not None,
                }
        return flags

    # ========== Dataset Snapshots (Reproducibility) ==========

    def create_dataset_snapshot(self, plan_id: Optional[str], name: str,
                                description: str = "", metadata: Dict = None) -> Optional[int]:
        """Create a dataset snapshot entry."""
        meta_json = json.dumps(metadata) if metadata else None
        query = """
        INSERT INTO dataset_snapshots (plan_id, name, description, metadata)
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (plan_id, name, description, meta_json))
            return cursor.lastrowid

    def add_snapshot_item(self, snapshot_id: int, item_type: str, item_id: str,
                          metadata: Dict = None) -> Optional[int]:
        """Add an item to a dataset snapshot."""
        meta_json = json.dumps(metadata) if metadata else None
        query = """
        INSERT INTO dataset_snapshot_items (snapshot_id, item_type, item_id, metadata)
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (snapshot_id, item_type, item_id, meta_json))
            return cursor.lastrowid

    def list_snapshot_items(self, snapshot_id: int) -> List[Dict]:
        """List items of a dataset snapshot."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM dataset_snapshot_items WHERE snapshot_id = ? ORDER BY created_at DESC",
                (snapshot_id,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_snapshot(self, snapshot_id: int) -> Optional[Dict]:
        """Get snapshot metadata by id."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM dataset_snapshots WHERE id = ?", (snapshot_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_snapshot_by_plan(self, plan_id: str) -> Optional[Dict]:
        """Get latest snapshot for a plan."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM dataset_snapshots WHERE plan_id = ? ORDER BY created_at DESC LIMIT 1",
                (plan_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========== Experiments ==========
    
    @log_exception(logger)
    def save_experiment(self, name: str, exp_type: str, raw_path: str, results: Dict, material_id: str = None) -> int:
        """Save experiment results."""
        query = """
        INSERT INTO experiments (name, type, raw_data_path, results, material_id)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (name, exp_type, raw_path, json.dumps(results), material_id))
            return cursor.lastrowid
            
    def fetch_training_set(self, target_col: str = "formation_energy",
                           allowed_elements: Optional[List[str]] = None) -> List[Dict]:
        """Fetch data for training."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Check if target_col exists in table
            cursor.execute("PRAGMA table_info(materials)")
            columns = [row["name"] for row in cursor.fetchall()]
            
            if target_col in columns:
                query = f"SELECT material_id, formula, cif_path, dos_data, {target_col} FROM materials WHERE {target_col} IS NOT NULL"
                cursor.execute(query)
                rows = cursor.fetchall()
            else:
                # Fallback: Fetch dos_data and extract in Python
                query = "SELECT material_id, formula, cif_path, dos_data FROM materials WHERE dos_data IS NOT NULL"
                cursor.execute(query)
                rows = cursor.fetchall()
                
            records = []
            for row in rows:
                rec = dict(row)
                if target_col not in rec:
                     # Try extract from dos_data
                     try:
                         dos = json.loads(rec.get("dos_data", "{}"))
                         val = dos.get(target_col)
                         if val is not None:
                             rec[target_col] = val
                         else:
                             continue # Skip if target not found
                     except Exception:
                         continue
                records.append(rec)
                
            return self._filter_material_records(records, allowed_elements)

    def fetch_activity_training_set(self, metric_name: str,
                                    allowed_elements: Optional[List[str]] = None) -> List[Dict]:
        """Fetch activity metrics joined with materials for ML training."""
        if not metric_name:
            return []
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT m.material_id, m.formula, m.cif_path, m.dos_data,
                       a.metric_value, a.created_at
                FROM activity_metrics a
                JOIN materials m ON a.material_id = m.material_id
                WHERE a.metric_name = ? AND a.metric_value IS NOT NULL
                ORDER BY a.created_at DESC
                """,
                (metric_name,),
            )
            rows = cursor.fetchall()
            if not rows:
                return []
            # Keep only latest per material_id
            seen = set()
            records = []
            for row in rows:
                mid = row["material_id"]
                if not mid or mid in seen:
                    continue
                rec = dict(row)
                rec["target"] = rec.pop("metric_value")
                records.append(rec)
                seen.add(mid)
            return self._filter_material_records(records, allowed_elements)

    # ========== Adsorption Energies (Catalysis-Hub) ==========

    def save_adsorption_energy(self, material_id: Optional[str], surface_composition: str,
                               facet: str, adsorbate: str,
                               reaction_energy: Optional[float],
                               activation_energy: Optional[float],
                               source: str = "Catalysis-Hub",
                               metadata: Dict = None) -> int:
        """Save adsorption energy record (proxy for activity)."""
        material_id = self._normalize_material_id(material_id)
        if material_id:
            # Ensure FK target exists (do not override existing formula).
            try:
                self.ensure_material_stub(material_id)
            except Exception:
                pass
        query = """
        INSERT INTO adsorption_energies
        (material_id, surface_composition, facet, adsorbate, reaction_energy, activation_energy, source, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        meta_json = json.dumps(metadata) if metadata else None
        record_id = None
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                material_id,
                surface_composition,
                facet,
                adsorbate,
                reaction_energy,
                activation_energy,
                source,
                meta_json
            ))
            record_id = cursor.lastrowid
        # Link as evidence when material_id is known (new connection to avoid locks)
        if material_id:
            try:
                self.save_evidence(
                    material_id=material_id,
                    source_type="adsorption_energy",
                    source_id=str(record_id),
                    score=0.8,
                    metadata={
                        "surface_composition": surface_composition,
                        "facet": facet,
                        "adsorbate": adsorbate,
                        "reaction_energy": reaction_energy,
                        "activation_energy": activation_energy,
                        "source": source,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to link adsorption evidence: {e}")

        return record_id

    def list_adsorption_energies(self, material_id: str) -> List[Dict]:
        """List adsorption energy records for a material."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM adsorption_energies WHERE material_id = ? ORDER BY created_at DESC",
                (material_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


    # ========== Activity Metrics (HOR/HER Indicators) ==========

    def save_activity_metric(self, material_id: Optional[str], metric_name: str,
                             metric_value: Optional[float], unit: Optional[str] = None,
                             conditions: Dict = None, source: str = "experiment",
                             source_id: Optional[str] = None, metadata: Dict = None) -> int:
        """Save activity metric record and link as evidence when possible."""
        material_id = self._normalize_material_id(material_id)
        if material_id:
            # Ensure FK target exists (do not override existing formula).
            try:
                self.ensure_material_stub(material_id, formula=(metadata or {}).get("formula") if isinstance(metadata, dict) else None)
            except Exception:
                # If we can't ensure, fall back to NULL to avoid rejecting the metric row.
                material_id = None
        query = """
        INSERT INTO activity_metrics
        (material_id, metric_name, metric_value, unit, conditions, source, source_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cond_json = json.dumps(conditions) if conditions else None
        meta_json = json.dumps(metadata) if metadata else None
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (material_id, metric_name, metric_value, unit, cond_json, source, source_id, meta_json)
            )
            record_id = cursor.lastrowid

        if material_id:
            try:
                self.save_evidence(
                    material_id=material_id,
                    source_type="activity_metric",
                    source_id=str(record_id),
                    score=0.9,
                    metadata={
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "unit": unit,
                        "conditions": conditions,
                        "source": source,
                        "source_id": source_id,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to link activity metric evidence: {e}")

        return record_id

    def list_activity_metrics(self, material_id: str) -> List[Dict]:
        """List activity metrics for a material."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM activity_metrics WHERE material_id = ? ORDER BY created_at DESC",
                (material_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    def update_material_dos(self, material_id: str, dos_data: Dict) -> None:
        """Update DOS descriptors for a material (JSON)."""
        if dos_data is None:
            return
        dos_json = json.dumps(dos_data)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE materials SET dos_data = ? WHERE material_id = ?",
                (dos_json, material_id)
            )

    # ========== Models ==========
    
    @log_exception(logger)
    def save_model(self, name: str, model_type: str, target: str, 
                   metrics: Dict, filepath: str,
                   hyperparameters: Dict = None,
                   feature_cols: List[str] = None,
                   training_size: int = 0) -> int:
        """Save trained model run (M4 Enhanced)."""
        query = """
        INSERT INTO models (name, type, target, metrics, filepath, hyperparameters, feature_cols, training_size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        hp_json = json.dumps(hyperparameters) if hyperparameters else None
        feat_json = json.dumps(feature_cols) if feature_cols else None
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (name, model_type, target, json.dumps(metrics), filepath, hp_json, feat_json, training_size))
            return cursor.lastrowid

    def list_models(self) -> List[Dict]:
        """List all models."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # ========== Robot/Middleware Tasks ==========

    def create_robot_task(self, task_type: str, payload: Dict = None, external_id: Optional[str] = None) -> int:
        """Create a robot/middleware task."""
        payload_json = json.dumps(payload) if payload else None
        query = """
        INSERT INTO robot_tasks (task_type, payload, status, external_id)
        VALUES (?, ?, 'queued', ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (task_type, payload_json, external_id))
            return cursor.lastrowid

    def update_robot_task(self, task_id: int, status: Optional[str] = None,
                          result: Dict = None, external_id: Optional[str] = None) -> None:
        """Update a robot task status/result."""
        updates = []
        params: List[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if result is not None:
            updates.append("result = ?")
            params.append(json.dumps(result))
        if external_id is not None:
            updates.append("external_id = ?")
            params.append(external_id)
        if not updates:
            return
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(task_id)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE robot_tasks SET {', '.join(updates)} WHERE id = ?", params)

    def get_robot_task(self, task_id: int) -> Optional[Dict]:
        """Get a robot task by id."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM robot_tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("payload"):
                try:
                    data["payload"] = json.loads(data["payload"])
                except Exception:
                    pass
            if data.get("result"):
                try:
                    data["result"] = json.loads(data["result"])
                except Exception:
                    pass
            return data

    def get_robot_task_by_external(self, external_id: str) -> Optional[Dict]:
        """Get a robot task by external_id."""
        if not external_id:
            return None
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM robot_tasks WHERE external_id = ?", (external_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("payload"):
                try:
                    data["payload"] = json.loads(data["payload"])
                except Exception:
                    pass
            if data.get("result"):
                try:
                    data["result"] = json.loads(data["result"])
                except Exception:
                    pass
            return data

    def list_robot_tasks(self, limit: int = 50) -> List[Dict]:
        """List recent robot tasks."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM robot_tasks ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            items = []
            for row in rows:
                data = dict(row)
                if data.get("payload"):
                    try:
                        data["payload"] = json.loads(data["payload"])
                    except Exception:
                        pass
                if data.get("result"):
                    try:
                        data["result"] = json.loads(data["result"])
                    except Exception:
                        pass
                items.append(data)
            return items

    # ========== Robot Task Events (Idempotent callbacks) ==========

    def _stable_json(self, obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return json.dumps({"repr": repr(obj)}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def hash_payload(self, obj: Any) -> str:
        text = self._stable_json(obj)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_robot_task_event_by_callback_id(self, task_id: int, callback_id: str) -> Optional[Dict]:
        if not callback_id:
            return None
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT *
                FROM robot_task_events
                WHERE task_id = ? AND callback_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (task_id, callback_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("payload"):
                try:
                    data["payload"] = json.loads(data["payload"])
                except Exception:
                    pass
            return data

    def log_robot_task_event_idempotent(
        self,
        task_id: int,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        external_id: Optional[str] = None,
        callback_id: Optional[str] = None,
        payload_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log a robot callback event with idempotency (by callback_id or payload_hash)."""
        payload_hash = payload_hash or self.hash_payload(payload or {})
        payload_json = self._stable_json(payload or {})

        if callback_id:
            existing = self.get_robot_task_event_by_callback_id(task_id, callback_id)
            if existing:
                if existing.get("payload_hash") != payload_hash:
                    return {
                        "inserted": False,
                        "event_id": existing.get("id"),
                        "duplicate": False,
                        "conflict": True,
                    }
                return {
                    "inserted": False,
                    "event_id": existing.get("id"),
                    "duplicate": True,
                    "conflict": False,
                }

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO robot_task_events
                (task_id, external_id, callback_id, status, payload_hash, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_id, external_id, callback_id, status, payload_hash, payload_json),
            )
            inserted = cursor.rowcount == 1

            event_id = None
            if callback_id:
                cursor.execute(
                    "SELECT id, payload_hash FROM robot_task_events WHERE task_id = ? AND callback_id = ? LIMIT 1",
                    (task_id, callback_id),
                )
                row = cursor.fetchone()
                if row:
                    event_id = row[0]
                    existing_hash = row[1]
                    if existing_hash != payload_hash:
                        return {"inserted": False, "event_id": event_id, "duplicate": False, "conflict": True}
                    return {"inserted": inserted, "event_id": event_id, "duplicate": not inserted, "conflict": False}

            cursor.execute(
                "SELECT id FROM robot_task_events WHERE task_id = ? AND payload_hash = ? LIMIT 1",
                (task_id, payload_hash),
            )
            row = cursor.fetchone()
            event_id = row[0] if row else None
            return {"inserted": inserted, "event_id": event_id, "duplicate": not inserted, "conflict": False}

    def list_robot_task_events(self, task_id: int, limit: int = 50, status: Optional[str] = None) -> List[Dict]:
        """List robot task events (newest first)."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    """
                    SELECT *
                    FROM robot_task_events
                    WHERE task_id = ? AND status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (task_id, status, int(limit or 50)),
                )
            else:
                cursor.execute(
                    """
                    SELECT *
                    FROM robot_task_events
                    WHERE task_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (task_id, int(limit or 50)),
                )
            rows = cursor.fetchall()
            items: List[Dict[str, Any]] = []
            for row in rows:
                data = dict(row)
                if data.get("payload"):
                    try:
                        data["payload"] = json.loads(data["payload"])
                    except Exception:
                        pass
                items.append(data)
            return items
