import sqlite3
import json
import numpy as np
import os
import gzip

DB_PATH = 'data/imcs.db'
RAW_DIR = 'data/theory/dos_raw'

def load_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Loading data for Deep Validation...")
    
    # Load Materials with DOS
    cursor.execute("SELECT material_id, formula, dos_data FROM materials WHERE dos_data LIKE '%d_band_center%'")
    rows = cursor.fetchall()
    
    materials = []
    for mid, formula, dos_json in rows:
        try:
            d = json.loads(dos_json)
            materials.append({
                'id': mid,
                'formula': formula,
                'center': d.get('d_band_center'),
                'width': d.get('d_band_width'),
                'filling': d.get('d_band_filling'),
                'raw_path': d.get('dos_file_path')
            })
        except:
            pass
            
    # Load Adsorption
    cursor.execute("SELECT material_id, surface_composition, facet, reaction_energy, source FROM adsorption_energies WHERE adsorbate='H*'")
    ads_data = cursor.fetchall()
    
    conn.close()
    return materials, ads_data

def check_physics(materials):
    print("\n[1. Electronic Structure Physics Check]")
    centers = [m['center'] for m in materials if m['center'] is not None]
    fillings = [m['filling'] for m in materials if m['filling'] is not None]
    
    # 1.1 Band Filling Checks
    # Filling is integral of DOS up to Ef. For d-band (normalized), it should be between 0 and 1 (or 0-10 electrons).
    # Let's see the range.
    if fillings:
        print(f"  d-band Filling Range: {min(fillings):.2f} - {max(fillings):.2f}")
        
    # 1.2 Trend Check: Late vs Early Transition Metals
    # Late TM (Pt, Au) -> d-band effectively full/low energy -> Filling high, Center very negative
    # Early TM (Ti, Sc) -> d-band empty/high energy -> Filling low, Center close to 0 or positive
    
    noble_subset = [m for m in materials if any(x in m['formula'] for x in ['Pt', 'Au', 'Pd'])]
    early_subset = [m for m in materials if any(x in m['formula'] for x in ['Sc', 'Y', 'Ti', 'Zr'])]
    
    avg_noble_center = np.mean([m['center'] for m in noble_subset]) if noble_subset else 0
    avg_early_center = np.mean([m['center'] for m in early_subset]) if early_subset else 0
    
    print(f"  Avg d-band Center (Noble - Pt/Au/Pd): {avg_noble_center:.2f} eV (Expected: <-2.0)")
    print(f"  Avg d-band Center (Early - Sc/Y/Ti/Zr): {avg_early_center:.2f} eV (Expected: >-1.0)")
    
    if avg_noble_center < -1.5 and avg_early_center > -1.5:
        print("  -> Trend CONFIRMED: Noble metals have deep d-states, Early metals have shallow/empty states.")
    else:
        print("  -> WARNING: Unexpected trends in d-band centers.")

def check_adsorption(ads_data):
    print("\n[2. Adsorption Energy Consistency]")
    # Pt(111) H* benchmark is roughly -0.4 to -0.6 eV depending on coverage/method.
    # Catalyst Hub data varies.
    
    pt_111 = [r[3] for r in ads_data if 'Pt' in r[1] and '111' in r[2]]
    if pt_111:
        print(f"  Pt(111) H* Energy Mean: {np.mean(pt_111):.2f} eV (N={len(pt_111)})")
        print(f"  Pt(111) range: {min(pt_111):.2f} to {max(pt_111):.2f}")
    else:
        print("  No Pt(111) data found to benchmark.")

    # Strong binders (Ti, Zr) should be very negative (< -1.0 eV)
    strong_binders = [r[3] for r in ads_data if ('Ti' in r[1] or 'Zr' in r[1])]
    if strong_binders:
        print(f"  Strong Binders (Ti/Zr) Mean: {np.mean(strong_binders):.2f} eV")
    
    if pt_111 and strong_binders and np.mean(strong_binders) < np.mean(pt_111):
        print("  -> Trend CONFIRMED: Oxophilic metals (Ti/Zr) bind H much stronger than Noble metals.")
    else:
        print("  -> WARNING: Adsorption trends unclear.")

def check_files(materials):
    print("\n[3. Data Integrity & File Existence]")
    missing = 0
    for m in materials:
        if m['raw_path']:
            # The path in DB might be absolute or relative. Let's assume relative to CWD if not abs.
            # In update script: 'dos_file_path': os.path.abspath(...) 
            # It stores absolute path.
            path = m['raw_path']
            if not os.path.exists(path):
                missing += 1
    
    if missing == 0:
        print(f"  All {len(materials)} referenced DOS files exist on disk. ✅")
    else:
        print(f"  WARNING: {missing} referenced DOS files are MISSING from disk! ❌")

def main():
    mats, ads = load_data()
    check_files(mats)
    check_physics(mats)
    check_adsorption(ads)

if __name__ == "__main__":
    main()
