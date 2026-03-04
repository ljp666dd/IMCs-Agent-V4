import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from src.services.db.database import DatabaseService

def main():
    db = DatabaseService("data/imcs.db")
    
    mats = db.list_materials(limit=5000)
    print(f"Total materials fetched limit 5000: {len(mats)}")
    
    with_cif = [m for m in mats if m.get("cif_path")]
    print(f"Materials with cif_path: {len(with_cif)}")
    
    if with_cif:
        print(f"Sample cif_path: {with_cif[0]['cif_path']}")
        import json
        dos_str = with_cif[0].get("dos_data")
        dos_obj = json.loads(dos_str) if isinstance(dos_str, str) else dos_str
        print(f"Sample has dos_data? {bool(dos_obj)}")

if __name__ == "__main__":
    main()
