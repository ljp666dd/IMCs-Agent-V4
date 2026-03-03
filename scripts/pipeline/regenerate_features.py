import sqlite3
import json
import os
import numpy as np
from collections import defaultdict

# Config
DB_PATH = 'data/imcs.db'
LEGACY_JSON_PATH = 'data/theory/dos_data_extended.json'
OUTPUT_PATH = 'data/theory/master_theory_db.json'

def regenerate():
    print("Loading Legacy Feature Cache...")
    feature_cache = {}
    if os.path.exists(LEGACY_JSON_PATH):
        try:
            with open(LEGACY_JSON_PATH, 'r', encoding='utf-8') as f:
                raw_list = json.load(f)
                # Convert list to dict keyed by material_id
                for item in raw_list:
                    if 'material_id' in item:
                        feature_cache[item['material_id']] = item
            print(f"  Loaded {len(feature_cache)} legacy records.")
        except Exception as e:
            print(f"  Warning: Failed to load legacy JSON: {e}")
    
    print("\nfetching Adsorption Data from DB...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Fetch Adsorption (H/OH) -> Map to material_id -> {adsorbate: min_energy}
    ads_map = defaultdict(dict)
    
    # H Adsorption
    cursor.execute("SELECT material_id, reaction_energy FROM adsorption_energies WHERE adsorbate LIKE 'H%' AND material_id IS NOT NULL")
    for row in cursor.fetchall():
        mid = row['material_id']
        e = row['reaction_energy']
        # Keep minimum energy (strongest binding / most stable) if multiple sites
        if 'H' not in ads_map[mid] or e < ads_map[mid]['H']:
            ads_map[mid]['H'] = e
            
    # OH Adsorption
    cursor.execute("SELECT material_id, reaction_energy FROM adsorption_energies WHERE adsorbate LIKE 'OH%' AND material_id IS NOT NULL")
    for row in cursor.fetchall():
        mid = row['material_id']
        e = row['reaction_energy']
        if 'OH' not in ads_map[mid] or e < ads_map[mid]['OH']:
            ads_map[mid]['OH'] = e
            
    print(f"  Mapped Adsorption Data for {len(ads_map)} materials.")
    
    print("\nIterating Main Database (Materials)...")
    cursor.execute("SELECT material_id, formula, formation_energy, dos_data, cif_path FROM materials WHERE dos_data IS NOT NULL")
    rows = cursor.fetchall()
    print(f"  Found {len(rows)} materials with DOS data.")
    
    master_db = []
    
    for row in rows:
        mid = row['material_id']
        formula = row['formula']
        fe = row['formation_energy']
        cif_path = row['cif_path']
        
        # 1. Base Feature Dict
        record = {
            "material_id": mid,
            "formula": formula,
            "formation_energy": fe,
            "cif_file_path": cif_path, # Raw File Reference
            # "dos_file_path": None # Will calculate below
        }
        
        # 2. Merge Extended Features
        if mid in feature_cache:
            # Copy all keys from legacy cache
            legacy = feature_cache[mid]
            for k, v in legacy.items():
                if k not in record: # Don't overwrite basic info
                    record[k] = v
        else:
            # Missing in legacy! (The 37 new ones)
            # Try to extract minimal features from dos_data string in DB
            try:
                dos_json = json.loads(row['dos_data'])
                # Extract basic features if available in the DB JSON object
                # (The DB JSON is compact descriptors, not raw DOS)
                if 'd_band_center' in dos_json:
                    record['d_band_center'] = dos_json['d_band_center']
                if 'd_band_width' in dos_json:
                    record['d_band_width'] = dos_json['d_band_width']
                if 'd_band_filling' in dos_json:
                    record['d_band_filling'] = dos_json['d_band_filling']
                
                # If DB JSON has file path, use it
                if 'dos_file_path' in dos_json:
                    record['dos_file_path'] = dos_json['dos_file_path']
            except:
                pass
                
        # 3. Merge Adsorption
        if mid in ads_map:
            record['adsorption'] = ads_map[mid]
        else:
            record['adsorption'] = {}
            
        master_db.append(record)
        
    conn.close()
    
    print(f"\nGenerated Master DB with {len(master_db)} records.")
    
    # Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(master_db, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    regenerate()
