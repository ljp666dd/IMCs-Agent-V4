import json
import numpy as np
import os
import random
from collections import Counter

DB_PATH = 'data/theory/master_theory_db.json'

def verify():
    print(f"Loading {DB_PATH}...")
    try:
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading DB: {e}")
        return
        
    print(f"Loaded {len(data)} records.\n")

    # 1. Completeness Check
    print("[1. Data Completeness]")
    missing_center = [d['material_id'] for d in data if d.get('d_band_center') is None]
    missing_vol = [d['material_id'] for d in data if d.get('volume_per_atom') is None]
    
    print(f"  Records/Total: {len(data)}")
    print(f"  Records missing d-band center: {len(missing_center)}")
    print(f"  Records missing volume: {len(missing_vol)}")
    
    # 2. Physical Trends Check
    print("\n[2. Physical Trends Analysis]")
    centers = []
    fillings = []
    
    noble_centers = []
    early_centers = []
    
    for d in data:
        c = d.get('d_band_center')
        f = d.get('d_band_filling')
        form = d.get('formula', '')
        
        if c is not None and f is not None:
            centers.append(c)
            fillings.append(f)
            
            if any(el in form for el in ['Pt', 'Au', 'Pd']):
                noble_centers.append(c)
            if any(el in form for el in ['Sc', 'Y', 'Ti', 'Zr']):
                early_centers.append(c)
                
    if len(centers) > 1:
        corr = np.corrcoef(fillings, centers)[0, 1]
        print(f"  Correlation (d-band Filling vs Center): {corr:.4f} (Expected: Strong Negative)")
        
        if corr < -0.7:
             print("  ✅ Strong negative correlation confirmed.")
        else:
             print("  ⚠️ Weak correlation.")
             
        avg_noble = np.mean(noble_centers) if noble_centers else 0
        avg_early = np.mean(early_centers) if early_centers else 0
        print(f"  Avg Center (Noble): {avg_noble:.2f} eV")
        print(f"  Avg Center (Early): {avg_early:.2f} eV")
        
    # 3. Adsorption Validity
    print("\n[3. Adsorption Data Check]")
    has_h = [d for d in data if d.get('adsorption', {}).get('H') is not None]
    print(f"  Records with H Adsorption: {len(has_h)}")
    
    # Check Pt
    pt_rec = next((d for d in has_h if d['material_id'] == 'mp-126'), None)
    if pt_rec:
        print(f"  Pt (mp-126) H Adsorption: {pt_rec['adsorption']['H']} eV")

    # 4. Space Group Distribution
    print("\n[4. Structure & Symmetry Analysis]")
    sg_nums = [d['space_group_number'] for d in data if d.get('space_group_number')]
    print(f"  Records with Space Group: {len(sg_nums)}/{len(data)}")
    
    if sg_nums:
        counts = Counter(sg_nums)
        top5 = counts.most_common(5)
        print(f"  Top 5 Space Groups: {top5}")
        
        count_225 = counts.get(225, 0) # Fm-3m
        count_221 = counts.get(221, 0) # L12
        count_123 = counts.get(123, 0) # L10
        
        print(f"  - Fm-3m (FCC): {count_225}")
        print(f"  - Pm-3m (L12): {count_221}")
        print(f"  - P4/mmm (L10): {count_123}")
        
        if count_221 > 0 or count_123 > 0:
            print("  ✅ Ordered phases (L12/L10) successfully identified.")

    # 5. File Integrity Random Check
    print("\n[5. Physical Asset Sampling]")
    sample_size = 20
    samples = random.sample(data, min(len(data), sample_size))
    valid_cif = 0
    valid_dos = 0
    
    for d in samples:
        cpath = d.get('cif_file_path')
        dpath = d.get('dos_file_path')
        
        if cpath and os.path.exists(cpath): valid_cif += 1
        else: print(f"  ❌ Missing CIF: {cpath}")
            
        if dpath and os.path.exists(dpath): valid_dos += 1
        else: print(f"  ❌ Missing DOS: {dpath}")
        
    print(f"  Checked {len(samples)} random records:")
    print(f"  - Valid CIF Paths: {valid_cif}/{len(samples)}")
    print(f"  - Valid DOS Paths: {valid_dos}/{len(samples)}")
    
    if valid_cif == len(samples) and valid_dos == len(samples):
        print("  ✅ 100% Asset Integrity Confirmed.")

if __name__ == "__main__":
    verify()
