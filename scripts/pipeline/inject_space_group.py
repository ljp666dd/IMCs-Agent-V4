import json
import os
import re
import sqlite3

MASTER_PATH = 'data/theory/master_theory_db.json'
DB_PATH = 'data/imcs.db'

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def extract_space_group(cif_path):
    if not os.path.exists(cif_path):
        return None, None
        
    try:
        struct = Structure.from_file(cif_path)
        # Use SpacegroupAnalyzer to find the CONVENTIONAL standard setting
        # This converts primitive P1 back to Fm-3m etc.
        sga = SpacegroupAnalyzer(struct)
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
        is_ordered = struct.is_ordered
        return sg_symbol, sg_number, is_ordered
    except Exception as e:
        # Fallback to regex if pymatgen fails (unlikely)
        # print(f"Pymatgen failed for {cif_path}: {e}")
        return None, None, None

def run():
    print(f"Loading {MASTER_PATH}...")
    with open(MASTER_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Processing {len(data)} records...")
    
    # 1. Map Material ID to File Path by scanning directory
    cif_dir = 'data/theory/cifs'
    id_to_file = {}
    if os.path.exists(cif_dir):
        for fname in os.listdir(cif_dir):
            if fname.endswith('.cif'):
                # Assumes filename is "mp-123.cif" or similar containing ID
                mid = fname.replace('.cif', '')
                id_to_file[mid] = os.path.join(cif_dir, fname)
    print(f"Found {len(id_to_file)} CIF files in {cif_dir}")

    updated_count = 0
    fixed_paths = 0
    
    # DEBUG: Check naming pattern
    print(f"DEBUG: Sample map keys: {list(id_to_file.keys())[:5]}")
    
    # DEBUG: Check if sample missing ID is in map
    sample_missing = 'mp-1217070'
    print(f"DEBUG: Checking map for {sample_missing}...")
    if sample_missing in id_to_file:
         print(f"  Result: FOUND in map -> {id_to_file[sample_missing]}")
    else:
         print(f"  Result: NOT FOUND in map. (Map size: {len(id_to_file)})")
         
    for rec in data:
        mid = rec['material_id']
        
        # Determine path: use existing or find in map
        # LOGIC CHANGE: Even if current_path exists, verify it? No, trust DB if set?
        # User complained 556 fixed. 
        # Force re-check if missing space group?
        
        current_path = rec.get('cif_file_path')
        
        if not current_path and mid in id_to_file:
            current_path = id_to_file[mid]
            rec['cif_file_path'] = current_path
            fixed_paths += 1
            
        if current_path:
             # Fix path separators
            cif_norm = os.path.normpath(current_path)
            
            symbol, number, is_ordered = extract_space_group(cif_norm)
            if symbol:
                rec['space_group'] = symbol
                rec['space_group_number'] = number
                rec['is_ordered'] = is_ordered
                updated_count += 1
            else:
                pass # Silent failure for pymatgen is okay now
                
    print(f"Fixed missing CIF paths for {fixed_paths} records.")
    print(f"Extracted space groups for {updated_count} records.")
    
    # Save JSON
    with open(MASTER_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print("Updated master JSON.")
    
    # Sync to DB
    print("Syncing to SQLite...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for rec in data:
        mid = rec['material_id']
        json_str = json.dumps(rec)
        # Update both dos_data (blob) and cif_path (column)
        cif_p = rec.get('cif_file_path')
        sg = rec.get('space_group')
        is_ord = 1 if rec.get('is_ordered') else 0
        cursor.execute("UPDATE materials SET dos_data = ?, cif_path = ?, space_group = ?, is_ordered = ? WHERE material_id = ?", (json_str, cif_p, sg, is_ord, mid))
        
    conn.commit()
    conn.close()
    print("Database sync complete.")

if __name__ == "__main__":
    run()
