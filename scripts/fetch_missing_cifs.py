import json
import os
import time
from dotenv import load_dotenv
from mp_api.client import MPRester

# Config
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")
MASTER_PATH = 'data/theory/master_theory_db.json'
CIF_DIR = 'data/theory/cifs'

def fetch_missing():
    if not API_KEY:
        print("Error: MP_API_KEY not found in .env")
        return

    print(f"Loading {MASTER_PATH}...")
    with open(MASTER_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Identify missing
    missing_ids = []
    for rec in data:
        mid = rec['material_id']
        path = rec.get('cif_file_path')
        
        # logical check: path missing OR file missing
        needs_fetch = False
        if not path:
             needs_fetch = True
        elif not os.path.exists(path):
             needs_fetch = True
             
        # Also check if file exists in dir even if path not set (already covered by previous fix script, but safety first)
        if needs_fetch:
            expected_file = os.path.join(CIF_DIR, f"{mid}.cif")
            if os.path.exists(expected_file):
                needs_fetch = False
                
        if needs_fetch:
            missing_ids.append(mid)
            
    print(f"Found {len(missing_ids)} missing CIFs.")
    if not missing_ids:
        print("Nothing to fetch.")
        return

    # Batch process
    batch_size = 100
    total_fetched = 0
    
    print(f"Fetching in batches of {batch_size}...")
    
    with MPRester(API_KEY) as mpr:
        for i in range(0, len(missing_ids), batch_size):
            batch = missing_ids[i:i+batch_size]
            print(f"  Batch {i//batch_size + 1}: Fetching {len(batch)} items...")
            
            try:
                docs = mpr.materials.summary.search(
                    material_ids=batch,
                    fields=["material_id", "structure"]
                )
                
                for doc in docs:
                    mid = str(doc.material_id)
                    struct = doc.structure
                    filepath = os.path.join(CIF_DIR, f"{mid}.cif")
                    struct.to(filename=filepath)
                    total_fetched += 1
                    
                print(f"    Saved {len(docs)} CIFs.")
                
            except Exception as e:
                print(f"    Error batch: {e}")
                
            time.sleep(1) # Be nice to API

    print(f"Done. Successfully fetched {total_fetched} new CIF files.")

if __name__ == "__main__":
    fetch_missing()
