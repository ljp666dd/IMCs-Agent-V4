import sqlite3
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from src.core.logger import get_logger, log_exception
from src.config.config import config

logger = get_logger(__name__)

from .chat_repo import ChatRepoMixin
from .plan_repo import PlanRepoMixin
from .material_repo import MaterialRepoMixin

class DatabaseService(ChatRepoMixin, PlanRepoMixin, MaterialRepoMixin):
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

    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall database system statistics (M4 feature)."""
        stats = {
            "total_materials": 0,
            "total_models": 0,
            "total_tasks": 0,
            "total_chat_sessions": 0
        }
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM materials")
                stats["total_materials"] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM models")
                stats["total_models"] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM robot_tasks")
                stats["total_tasks"] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM chat_sessions")
                stats["total_chat_sessions"] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                pass
        return stats
