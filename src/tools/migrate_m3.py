import sqlite3
import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.config.config import config

def migrate_m3():
    """
    Apply M3 Schema changes (Evidence Chain).
    """
    db_path = config.DB_PATH
    print(f"Migrating M3 (Evidence) at: {db_path}")
    
    if not os.path.exists(db_path):
        print("Database not found. Please run migrate_db.py (M1/M2) first.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Evidence Table
    # A unified table to link material_id to various evidence sources
    sql_evidence = """
    CREATE TABLE IF NOT EXISTS evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_id TEXT NOT NULL,      -- FK to materials.material_id (or internal ID)
        source_type TEXT NOT NULL,      -- 'experiment', 'literature', 'theory', 'ml_prediction'
        source_id TEXT,                 -- ID in the source table (e.g. experiment id, paper doi)
        score REAL,                     -- Confidence score / Quality score
        metadata TEXT,                  -- JSON: { "method": "DFT+U", "citation": "..." }
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(material_id) REFERENCES materials(material_id)
    );
    """
    try:
        cursor.execute(sql_evidence)
        print("Table 'evidence' created.")
    except Exception as e:
        print(f"Error creating evidence table: {e}")

    # 2. Add Linkage Columns to tables
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

    # Link Experiments -> Material (Theory/ID)
    add_column_if_missing("experiments", "material_id TEXT") 
    
    # Link Models -> Training Data Snapshot (for reproducibility)
    add_column_if_missing("models", "dataset_hash TEXT")

    conn.commit()
    conn.close()
    print("M3 Migration complete.")

if __name__ == "__main__":
    migrate_m3()
