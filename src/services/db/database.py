import sqlite3
import os
import json
from typing import List, Dict, Any, Optional
from src.core.logger import get_logger, log_exception

logger = get_logger(__name__)

class DatabaseService:
    """
    Service for SQLite database interactions.
    Manages Materials, Experiments, and Models.
    """
    
    def __init__(self, db_path: str = "data/imcs.db"):
        self.db_path = db_path
        self._init_db()
        
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        return conn

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
            
    def list_materials(self, limit: int = 100, allowed_elements: Optional[List[str]] = None) -> List[Dict]:
        """List stored materials."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM materials ORDER BY created_at DESC LIMIT ?", (limit,))
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
            cif_path = data.get("cif_path")
            if cif_path and os.path.exists(cif_path):
                with open(cif_path, "r") as f:
                    data["cif_content"] = f.read()
            else:
                data["cif_content"] = None
                
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
                cif_path = data.get("cif_path")
                if cif_path and os.path.exists(cif_path):
                    with open(cif_path, "r") as f:
                        data["cif_content"] = f.read()
                else:
                    data["cif_content"] = None
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

            if not allowed_ids:
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
            # Fetch only rows where target is not null
            query = f"SELECT material_id, formula, cif_path, dos_data, {target_col} FROM materials WHERE {target_col} IS NOT NULL"
            cursor.execute(query)
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
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
