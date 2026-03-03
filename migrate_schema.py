import os
import json
import sqlite3
import argparse

DB_PATH = 'data/imcs.db'

def alter_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Check if columns exist
    cursor.execute("PRAGMA table_info(materials)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'space_group' not in columns:
        print("Adding space_group to materials...")
        cursor.execute("ALTER TABLE materials ADD COLUMN space_group TEXT")
    
    if 'is_ordered' not in columns:
        print("Adding is_ordered to materials...")
        cursor.execute("ALTER TABLE materials ADD COLUMN is_ordered BOOLEAN DEFAULT 0")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    alter_schema()
