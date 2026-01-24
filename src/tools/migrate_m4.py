import sqlite3
import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.config.config import config

def migrate_m4():
    """
    Apply M4 Schema changes (ML Reliability).
    """
    db_path = config.DB_PATH
    print(f"Migrating M4 (ML Registry) at: {db_path}")
    
    if not os.path.exists(db_path):
        print("Database not found. Please run migrate_db.py first.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Add columns to models table
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

    # Enhanced Registry Metadata
    add_column_if_missing("models", "hyperparameters TEXT") 
    add_column_if_missing("models", "feature_cols TEXT") # JSON list of feature names used
    add_column_if_missing("models", "training_size INTEGER")

    conn.commit()
    conn.close()
    print("M4 Migration complete.")

if __name__ == "__main__":
    migrate_m4()
