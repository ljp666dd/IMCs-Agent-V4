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
        return sqlite3.connect(self.db_path)
        
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
                      dependencies: Optional[List[str]] = None):
        """Log a step execution (Insert or Update)."""
        # We use INSERT OR REPLACE if ID is step_id? No, step_id is text string 'step_1'.
        # We should use INSERT for history log, but for current state capturing 'latest' is fine.
        # Given we want trace, simple INSERT is best. To avoid clutter, maybe check existence?
        # Actually, `plan_steps` has an auto-increment ID.
        # For simplicity in v4 Pilot, just INSERT log entry.
        
        query = """
        INSERT INTO plan_steps (plan_id, step_id, agent, action, dependencies, status, result, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        deps_json = json.dumps(dependencies or []) if dependencies is not None else None
        result_json = None
        if result is not None:
            try:
                result_json = json.dumps(result, default=str)
            except Exception:
                result_json = json.dumps(str(result))
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (plan_id, step_id, agent, action, deps_json, status, result_json, error))

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
        INSERT OR REPLACE INTO materials (material_id, formula, formation_energy, cif_path) 
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (material_id, formula, energy, cif_path))
            return cursor.lastrowid
            
    def list_materials(self, limit: int = 100) -> List[Dict]:
        """List stored materials."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM materials ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def list_materials_since(self, created_at: str, limit: int = 100) -> List[Dict]:
        """List materials created since a timestamp."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM materials WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (created_at, limit),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
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
            return data

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
            return cursor.lastrowid

    def get_evidence_for_material(self, material_id: str) -> List[Dict]:
        """Get all evidence linked to a material."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evidence WHERE material_id = ? ORDER BY created_at DESC", (material_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

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
            
    def fetch_training_set(self, target_col: str = "formation_energy") -> List[Dict]:
        """Fetch data for training."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Fetch only rows where target is not null
            query = f"SELECT material_id, formula, cif_path, {target_col} FROM materials WHERE {target_col} IS NOT NULL"
            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

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
