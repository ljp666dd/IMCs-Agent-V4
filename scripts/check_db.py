import sys
import os
import sqlite3

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from services.db.database import DatabaseService

def check_db():
    print(f"Checking database from CWD: {os.getcwd()}")
    
    db_path = "data/imcs.db"
    if not os.path.exists(db_path):
        print(f"❌ Database file NOT found at: {os.path.abspath(db_path)}")
        return
        
    print(f"Database found at: {os.path.abspath(db_path)}")
    
    db = DatabaseService()
    try:
        materials = db.list_materials()
        print(f"Found {len(materials)} materials in DB:")
        for m in materials:
            print(f" - {m['material_id']}: {m['formula']} (Energy: {m['formation_energy']})")
            
    except Exception as e:
        print(f"❌ Query failed: {e}")

if __name__ == "__main__":
    check_db()
