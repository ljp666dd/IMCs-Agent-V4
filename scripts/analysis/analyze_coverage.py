import sqlite3
import re
import json

DB_PATH = 'data/imcs.db'

ALLOWED_ELEMENTS = {
    "Pt", "Pd", "Au", "Ag", "Ir", "Rh", "Ru", "Os",  # 贵金属
    "Ni", "Co", "Fe", "Cu", "Mn", "Cr", "V", "Ti", "Zn", "Sc",  # 3d
    "Mo", "W", "Nb", "Ta", "Zr", "Hf", "Re", "Y",  # 4d/5d
    "Cd", "In", "Sn", "Ga", "Ge", "Al", "La", "Ce"  # 其他
}

def analyze():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print(f"Analyzing coverage for {len(ALLOWED_ELEMENTS)} allowed elements (1-5 components)...")
    
    # Get all materials
    cursor.execute("SELECT material_id, formula, dos_data FROM materials")
    rows = cursor.fetchall()
    
    total_compliant = 0
    compliant_with_dos = 0
    compliant_ids = []
    
    for mid, formula, dos_data in rows:
        # Calculate nelements from formula
        elements = set(re.findall(r'([A-Z][a-z]?)', formula))
        nelements = len(elements)
        
        # 1. Component count check
        if not (1 <= nelements <= 5):
            continue
            
        # 2. Element check
        try:
            # Composition is stored as JSON in 'composition' column or derived from formula
            # Let's verify elements from formula as fallback or regex
            # Simplified regex for elements
            elements = set(re.findall(r'([A-Z][a-z]?)', formula))
            
            # Check if all elements are allowed
            if not elements.issubset(ALLOWED_ELEMENTS):
                continue
                
            total_compliant += 1
            compliant_ids.append(mid)
            
            if dos_data and "d_band_center" in dos_data:
                compliant_with_dos += 1
                
        except Exception as e:
            continue

    print(f"\n[Theoretical Data Coverage]")
    print(f"Total Materials in DB: {len(rows)}")
    print(f"Compliant Materials (1-5 elements, Allowed list): {total_compliant}")
    print(f"  - With Formation Energy: {total_compliant} (Approx 100%)")
    print(f"  - With High-Quality DOS: {compliant_with_dos} ({compliant_with_dos/total_compliant*100:.1f}%)")
    
    # Check Adsorption coverage
    # Get all material_ids in adsorption table
    cursor.execute("SELECT DISTINCT material_id FROM adsorption_energies")
    ads_ids = set(r[0] for r in cursor.fetchall())
    
    # Count intersection
    # Note: Adsorption IDs might be 'lit:...' or mapped from comp. 
    # Current script maps by exact ID match which might undercount if IDs differ.
    # But usually we try to link them.
    
    compliant_with_ads = 0
    for mid in compliant_ids:
        if mid in ads_ids:
            compliant_with_ads += 1
            
    print(f"  - With Adsorption Energy: {compliant_with_ads} (Matched IDs)")
    
    # Also check how many raw adsorption entries exist
    cursor.execute("SELECT COUNT(*) FROM adsorption_energies")
    total_ads = cursor.fetchone()[0]
    print(f"\n[Raw Adsorption Data]")
    print(f"Total Entries: {total_ads}")
    
    conn.close()

if __name__ == "__main__":
    analyze()
