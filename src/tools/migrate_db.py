import sqlite3
import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.config.config import config

def migrate():
    """
    Apply v4.0 Schema changes to existing database.
    Safe to run multiple times.
    """
    db_path = config.DB_PATH
    print(f"Migrating database at: {db_path}")
    
    if not os.path.exists(db_path):
        print("Database not found. Initializing fresh...")
        # DatabaseService will init on first run, so nothing to do here
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create New Tables (Users, Sessions)
    # The schema.sql has these, but execute script might skip if DB exists.
    # We force create them here.
    
    tables = [
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            api_key TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",
        """CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );""",
        """CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            artifacts TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
        );""",
        """CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY, -- Using UUID string for easier backend handling
            user_id INTEGER,
            task_type TEXT NOT NULL,
            status TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );""",
        """CREATE TABLE IF NOT EXISTS plan_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            agent TEXT NOT NULL,
            action TEXT NOT NULL,
            params TEXT,
            status TEXT NOT NULL,
            result TEXT, -- JSON
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(plan_id) REFERENCES plans(id)
        );""",
        """CREATE TABLE IF NOT EXISTS adsorption_energies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id TEXT,
            surface_composition TEXT,
            facet TEXT,
            adsorbate TEXT,
            reaction_energy REAL,
            activation_energy REAL,
            source TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(material_id) REFERENCES materials(material_id)
        );""",
        """CREATE TABLE IF NOT EXISTS activity_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id TEXT,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            unit TEXT,
            conditions TEXT,
            source TEXT,
            source_id TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(material_id) REFERENCES materials(material_id)
        );""",
        """CREATE TABLE IF NOT EXISTS knowledge_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            canonical_id TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(entity_type, name)
        );""",
        """CREATE TABLE IF NOT EXISTS knowledge_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_id TEXT,
            title TEXT,
            url TEXT,
            year INTEGER,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",
        """CREATE TABLE IF NOT EXISTS knowledge_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER NOT NULL,
            predicate TEXT NOT NULL,
            object_id INTEGER NOT NULL,
            confidence REAL,
            evidence_source_id INTEGER,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(subject_id) REFERENCES knowledge_entities(id),
            FOREIGN KEY(object_id) REFERENCES knowledge_entities(id),
            FOREIGN KEY(evidence_source_id) REFERENCES knowledge_sources(id)
        );""",
        """CREATE TABLE IF NOT EXISTS knowledge_relation_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relation_id INTEGER NOT NULL,
            source_id INTEGER NOT NULL,
            score REAL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(relation_id) REFERENCES knowledge_relations(id),
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(id)
        );""",
        """CREATE TABLE IF NOT EXISTS dataset_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT,
            name TEXT,
            description TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(plan_id) REFERENCES plans(id)
        );""",
        """CREATE TABLE IF NOT EXISTS dataset_snapshot_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            item_type TEXT NOT NULL,
            item_id TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(snapshot_id) REFERENCES dataset_snapshots(id)
        );"""
        ,
        """CREATE TABLE IF NOT EXISTS robot_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT NOT NULL,
            payload TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            result TEXT,
            external_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );"""
        ,
        """CREATE TABLE IF NOT EXISTS robot_task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            external_id TEXT,
            callback_id TEXT,
            status TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(task_id) REFERENCES robot_tasks(id),
            UNIQUE(task_id, callback_id),
            UNIQUE(task_id, payload_hash)
        );"""
    ]
    
    for sql in tables:
        try:
            cursor.execute(sql)
            print("Table created/verified.")
        except Exception as e:
            print(f"Error checking table: {e}")

    # 2. Add Columns to Existing Tables (Materials, Experiments)
    # SQLite ALTER TABLE ADD COLUMN is tricky with IF NOT EXISTS logic in one go.
    # We check columns first.
    
    def add_column_if_missing(table, col_def):
        col_name = col_def.split()[0]
        cursor.execute(f"PRAGMA table_info({table})")
        existing_cols = [row[1] for row in cursor.fetchall()]
        
        if col_name not in existing_cols:
            print(f"Adding column {col_name} to {table}...")
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
            except Exception as e:
                print(f"Failed to add column: {e}")
        else:
            print(f"Column {col_name} exists in {table}.")

    add_column_if_missing("materials", "user_id INTEGER REFERENCES users(id)")
    add_column_if_missing("experiments", "user_id INTEGER REFERENCES users(id)")
    add_column_if_missing("models", "user_id INTEGER REFERENCES users(id)")
    add_column_if_missing("plan_steps", "dependencies TEXT")
    add_column_if_missing("plan_steps", "params TEXT")

    conn.commit()
    conn.close()
    print("Migration complete. v4.0 Schema active.")

if __name__ == "__main__":
    migrate()
