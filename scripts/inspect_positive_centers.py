import sqlite3
import json

def inspect():
    conn = sqlite3.connect('data/imcs.db')
    cursor = conn.cursor()
    
    print("Inspecting Positive d-band Centers:")
    cursor.execute("SELECT material_id, formula, dos_data FROM materials WHERE dos_data LIKE '%d_band_center%'")
    
    count_noble = 0
    count_early = 0
    noble_metals = ["Pt", "Pd", "Au", "Ag", "Ir", "Rh", "Ru", "Os"]
    
    for row in cursor.fetchall():
        try:
            data = json.loads(row[2])
            center = data.get('d_band_center')
            if center and center > 0:
                formula = row[1]
                print(f"  {row[0]:12} {formula:15} : {center:.2f} eV")
                
                # Check composition
                is_noble = any(el in formula for el in noble_metals)
                if is_noble:
                    count_noble += 1
                else:
                    count_early += 1
        except:
            pass
            
    print(f"\nSummary of Positive Centers:")
    print(f"  Noble Metal containing: {count_noble}")
    print(f"  Others (Early TM?): {count_early}")
    
    conn.close()

if __name__ == "__main__":
    inspect()
