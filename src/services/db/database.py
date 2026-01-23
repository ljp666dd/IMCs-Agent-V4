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
            
    # ========== Experiments ==========
    
    @log_exception(logger)
    def save_experiment(self, name: str, exp_type: str, raw_path: str, results: Dict) -> int:
        """Save experiment results."""
        query = """
        INSERT INTO experiments (name, type, raw_data_path, results)
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (name, exp_type, raw_path, json.dumps(results)))
            return cursor.lastrowid
            
    # ========== Models ==========
    
    @log_exception(logger)
    def save_model(self, name: str, model_type: str, target: str, 
                   metrics: Dict, filepath: str) -> int:
        """Save trained model run."""
        query = """
        INSERT INTO models (name, type, target, metrics, filepath)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (name, model_type, target, json.dumps(metrics), filepath))
            return cursor.lastrowid

    def list_models(self) -> List[Dict]:
        """List all models."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
