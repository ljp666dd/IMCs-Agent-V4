import json

DB_PATH = 'data/theory/master_theory_db.json'

def validate():
    try:
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Total Records: {len(data)}")
        
        # Check Pt (mp-126) for completeness
        pt = next((d for d in data if d['material_id'] == 'mp-126'), None)
        if pt:
            print(f"\n[Validation: mp-126 (Pt)]")
            print(f"  Formula: {pt.get('formula')}")
            print(f"  d-band Center: {pt.get('d_band_center')}")
            print(f"  H Adsorption: {pt.get('adsorption', {}).get('H')}")
            print(f"  CIF Path: {pt.get('cif_file_path')}")
            # print(f"  DOS Path: {pt.get('dos_file_path')}") # might be None if generated
            
        # Check for 100% features
        missing_center = len([d for d in data if 'd_band_center' not in d])
        print(f"\nRecords missing d_band_center: {missing_center}")
        
    except Exception as e:
        print(f"Validation Error: {e}")

if __name__ == "__main__":
    validate()
