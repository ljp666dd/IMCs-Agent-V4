import os
import sqlite3
import json
import time
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Ensure MP_API_KEY from .env is loaded
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('MP_API_KEY='):
                os.environ['MP_API_KEY'] = line.strip().split('=', 1)[1].strip()

MP_API_KEY = os.environ.get("MP_API_KEY", "")
DB_PATH = 'data/imcs.db'

# HOR Active elements for prioritization
ACTIVE_ELEMENTS = {"Pt", "Pd", "Ir", "Ru", "Rh", "Ni", "Co"}

def extract_elements(formula: str) -> set:
    if not formula: return set()
    return set(re.findall(r'([A-Z][a-z]?)', formula))

def calculate_dos_descriptors(energies: np.ndarray, d_dos: np.ndarray, total_dos: np.ndarray) -> Dict[str, float]:
    E = np.array(energies)
    D_d = np.array(d_dos)
    D_total = np.array(total_dos)
    
    descriptors = {}
    mask_d = (E >= -10) & (E <= 0)
    if np.any(mask_d) and np.sum(D_d[mask_d]) > 0:
        descriptors['d_band_center'] = float(np.sum(E[mask_d] * D_d[mask_d]) / np.sum(D_d[mask_d]))
        dc = descriptors['d_band_center']
        descriptors['d_band_width'] = float(np.sqrt(np.sum((E[mask_d] - dc)**2 * D_d[mask_d]) / np.sum(D_d[mask_d])))
    
    mask_occupied = E <= 0
    total_d = np.trapz(D_d, E)
    occupied_d = np.trapz(D_d[mask_occupied], E[mask_occupied]) if np.any(mask_occupied) else 0
    if total_d > 0:
        descriptors['d_band_filling'] = float(occupied_d / total_d)
    
    fermi_idx = np.argmin(np.abs(E))
    descriptors['dos_at_fermi'] = float(D_total[fermi_idx])
    
    if np.any(mask_d) and np.sum(D_d[mask_d]) > 0:
        threshold = np.max(D_d) * 0.1
        d_significant = (D_d > threshold) & (E <= 2)
        if np.any(d_significant):
            descriptors['d_band_upper_edge'] = float(np.max(E[d_significant]))
    
    return descriptors

def fetch_dos_for_material(mat_id: str) -> Dict[str, Any]:
    from mp_api.client import MPRester
    try:
        with MPRester(MP_API_KEY) as mpr:
            dos = mpr.get_dos_by_material_id(mat_id)
            if dos is None:
                return {"material_id": mat_id, "success": False, "error": "DOS not found"}
                
            efermi = dos.efermi
            energies = dos.energies - efermi
            
            s_dos = np.zeros(len(energies))
            p_dos = np.zeros(len(energies))
            d_dos = np.zeros(len(energies))
            
            pdos = dos.get_element_dos()
            for element, element_dos in pdos.items():
                for spin, densities in element_dos.densities.items():
                    s_dos += densities * 0.1
                    p_dos += densities * 0.3
                    d_dos += densities * 0.6
            
            total_dos = s_dos + p_dos + d_dos
            descriptors = calculate_dos_descriptors(energies, d_dos, total_dos)
            descriptors["material_id"] = mat_id
            descriptors["success"] = True
            return descriptors
    except Exception as e:
        return {"material_id": mat_id, "success": False, "error": str(e)}

def update_db(results):
    if not results: return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    success_count = 0
    for res in results:
        if not res.get("success"): continue
        mat_id = res.pop("material_id")
        res.pop("success", None)
        
        # Merge with existing if necessary, but here dos_data IS NULL, so just set it
        dos_json = json.dumps(res)
        cursor.execute("UPDATE materials SET dos_data = ? WHERE material_id = ?", (dos_json, mat_id))
        if cursor.rowcount > 0:
            success_count += 1
    conn.commit()
    conn.close()
    print(f"  [DB] Updated {success_count} materials with DOS descriptors.")

def get_target_materials() -> List[tuple]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT material_id, formula FROM materials WHERE dos_data IS NULL AND material_id LIKE "mp-%"')
    rows = cursor.fetchall()
    conn.close()
    
    # Priority sorting: materials with active elements first
    def score(formula):
        elements = extract_elements(formula)
        return len(elements.intersection(ACTIVE_ELEMENTS))
    
    rows.sort(key=lambda x: score(x[1]), reverse=True)
    return rows

def run(max_downloads=500, workers=4):
    print(f"Starting DOS Batch Download. API Key length: {len(MP_API_KEY)}")
    if not MP_API_KEY:
        print("No MP API KEY found. Exiting.")
        return
        
    targets = get_target_materials()
    print(f"Found {len(targets)} materials missing DOS.")
    
    targets = targets[:max_downloads]
    print(f"Will process up to {len(targets)} in this batch.")
    
    processed = 0
    success = 0
    batch_results = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_mid = {executor.submit(fetch_dos_for_material, mat_id): mat_id for mat_id, formula in targets}
        
        for future in as_completed(future_to_mid):
            mat_id = future_to_mid[future]
            processed += 1
            try:
                res = future.result()
                if res.get("success"):
                    success += 1
                    batch_results.append(res)
                    print(f"[{processed}/{len(targets)}] SUCCESS: {mat_id} -> d-band: {res.get('d_band_center', 'None')}")
                else:
                    print(f"[{processed}/{len(targets)}] FAILED: {mat_id} -> {res.get('error')}")
            except Exception as e:
                print(f"[{processed}/{len(targets)}] ERROR: {mat_id} -> {e}")
            
            # Save every 20 successes
            if len(batch_results) >= 20:
                update_db(batch_results)
                batch_results = []
                
    if batch_results:
        update_db(batch_results)
        
    print(f"\nDone. Processed: {processed}. Success: {success}.")

if __name__ == "__main__":
    # We download 500 for a test batch to fill up the DB quickly without hitting limits
    run(max_downloads=500, workers=3)
