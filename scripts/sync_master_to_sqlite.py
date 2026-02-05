import sqlite3
import json
import os

MASTER_PATH = 'data/theory/master_theory_db.json'
DB_PATH = 'data/imcs.db'

def sync_to_db():
    print(f"Loading {MASTER_PATH}...")
    with open(MASTER_PATH, 'r', encoding='utf-8') as f:
        master_data = json.load(f)
        
    print(f"Syncing {len(master_data)} records to {DB_PATH}...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    updated_count = 0
    
    for rec in master_data:
        mid = rec['material_id']
        
        # We want to store the WHOLE record (minus material_id/formula if redundant, but keeping it simple) 
        # as the 'dos_data' blob, or specifically the features part.
        # The 'dos_data' column in DB is intended for the features JSON.
        
        # Clean record for DB storage:
        # 1. Remove top-level keys that are already columns (formula, formation_energy) to save space? 
        #    Actually, keeping them in the JSON blob doesn't hurt and makes it self-contained.
        #    But `dos_data` semantic is "electronic structure". 
        #    However, user wants "All info". 
        #    Let's store the entire merged dictionary into `dos_data` column, 
        #    so `get_material_details` returns it all.
        
        # Ensure file paths are relative or consistent
        if rec.get('cif_file_path'):
            # fix potential absolute paths or mixed slashes
            pass 
            
        json_str = json.dumps(rec)
        
        cursor.execute("UPDATE materials SET dos_data = ? WHERE material_id = ?", (json_str, mid))
        updated_count += 1
        
    conn.commit()
    conn.close()
    print(f"Successfully updated {updated_count} records in SQLite.")

if __name__ == "__main__":
    sync_to_db()
