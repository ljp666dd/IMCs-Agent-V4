import json
import os
import sqlite3

MASTER_PATH = 'data/theory/master_theory_db.json'
DB_PATH = 'data/imcs.db'
DOS_DIR = 'data/theory/dos_raw'

def fix_dos_links():
    print(f"Loading {MASTER_PATH}...")
    with open(MASTER_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print("Fixing DOS file paths...")
    updated = 0
    for rec in data:
        mid = rec['material_id']
        expected_name = f"{mid}_dos.json.gz"
        expected_path = os.path.join(DOS_DIR, expected_name)
        
        # Check if actually exists
        if os.path.exists(expected_path):
            rec['dos_file_path'] = expected_path.replace('\\', '/')
            updated += 1
        else:
            # Try without suffix?
            alt_path = os.path.join(DOS_DIR, f"{mid}.json.gz")
            if os.path.exists(alt_path):
                rec['dos_file_path'] = alt_path.replace('\\', '/')
                updated += 1
            else:
                rec['dos_file_path'] = None # Explicitly mark missing
                
    print(f"Linked DOS files for {updated}/{len(data)} records.")
    
    # Save
    with open(MASTER_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    # Sync DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for rec in data:
         mid = rec['material_id']
         json_str = json.dumps(rec)
         cursor.execute("UPDATE materials SET dos_data = ? WHERE material_id = ?", (json_str, mid))
    conn.commit()
    conn.close()
    print("DB Synced.")

if __name__ == "__main__":
    fix_dos_links()
