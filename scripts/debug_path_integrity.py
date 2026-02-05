import json
import os
import glob

MASTER_PATH = 'data/theory/master_theory_db.json'
CIF_DIR = 'data/theory/cifs'
DOS_DIR = 'data/theory/dos_raw'

def diagnose():
    print(f"Loading {MASTER_PATH}...")
    with open(MASTER_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total Materials in DB: {len(data)}")

    # 1. Inventory of Files on Disk
    disk_cifs = {os.path.basename(f) for f in glob.glob(os.path.join(CIF_DIR, "*.cif"))}
    disk_dos = {os.path.basename(f) for f in glob.glob(os.path.join(DOS_DIR, "*.json.gz"))}
    
    print(f"\n[Disk Inventory]")
    print(f"  CIF Files found: {len(disk_cifs)}")
    print(f"  DOS Files found: {len(disk_dos)}")

    # 2. Check DB Links
    missing_cif_link = 0
    broken_cif_link = 0
    missing_dos_link = 0
    broken_dos_link = 0
    
    missing_cif_ids = []
    
    for rec in data:
        mid = rec['material_id']
        
        # Check CIF
        cif_path = rec.get('cif_file_path')
        if not cif_path:
            missing_cif_link += 1
            missing_cif_ids.append(mid)
        elif not os.path.exists(cif_path):
            broken_cif_link += 1
            
        # Check DOS
        dos_path = rec.get('dos_file_path')
        if not dos_path:
            missing_dos_link += 1
        elif not os.path.exists(dos_path):
            broken_dos_link += 1

    print(f"\n[DB Link Health]")
    print(f"  Materials missing CIF path link: {missing_cif_link}")
    print(f"  Materials with BROKEN CIF path: {broken_cif_link}")
    print(f"  Materials missing DOS path link: {missing_dos_link}")
    print(f"  Materials with BROKEN DOS path: {broken_dos_link}")
    
    # 3. Analyze Mismatches
    # Why are links missing if files exist?
    print(f"\n[Mismatch Analysis]")
    print(f"  Sample Missing CIF IDs: {missing_cif_ids[:5]}")
    
    # Check if these missing IDs exist on disk with slightly different names
    found_recoverable = 0
    for mid in missing_cif_ids:
        # Check matching
        # Strategy 1: exact match "mid.cif" -> this was tried
        # Strategy 2: "mid_computed.cif" ?
        # Strategy 3: case sensitivity?
        expected = f"{mid}.cif"
        if expected in disk_cifs:
            # It exists on disk! Why wasn't it linked?
            # Maybe the previous script logic was flawed or map build failed?
            pass
        else:
            # Maybe file name is different?
            pass

    # Print first few disk filenames to check pattern
    print(f"  Sample Disk CIF Names: {list(disk_cifs)[:5]}")

if __name__ == "__main__":
    diagnose()
