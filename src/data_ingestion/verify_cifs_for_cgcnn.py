import os
import random
import warnings
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from tqdm import tqdm

# Suppress PyMatgen warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CIF_DIR = os.path.join(BASE_DIR, "data", "theory", "cifs")

def verify_cifs():
    cif_files = [f for f in os.listdir(CIF_DIR) if f.endswith(".cif")]
    
    if not cif_files:
        print("No CIF files found to verify.")
        return

    print(f"Verifying {len(cif_files)} CIF files for CGCNN compatibility...")
    
    valid_count = 0
    corrupted = []
    
    # We'll check all of them to be sure
    for cif_file in tqdm(cif_files):
        cif_path = os.path.join(CIF_DIR, cif_file)
        try:
            # 1. Parsing Test
            # CGCNN typically uses Structure.from_file or CifParser
            parser = CifParser(cif_path)
            structure = parser.get_structures()[0]
            
            # 2. Validity Check
            if not structure.is_valid():
                corrupted.append((cif_file, "Invalid Structure"))
                continue
                
            # 3. Neighbor Graph Test (Simulating CGCNN input generation)
            # CGCNN needs to find neighbors within a radius (usually 8 Angstrom)
            # If this fails, the cell might be too small or singular
            neighbors = structure.get_all_neighbors(r=8.0)
            
            if not neighbors:
                corrupted.append((cif_file, "No Neighbors Found (Cell too small?)"))
                continue
                
            valid_count += 1
            
        except Exception as e:
            corrupted.append((cif_file, str(e)))

    print("-" * 30)
    print(f"Verification Complete.")
    print(f"Total Valid: {valid_count} / {len(cif_files)}")
    print(f"Success Rate: {valid_count / len(cif_files) * 100:.2f}%")
    
    if corrupted:
        print(f"\nFound {len(corrupted)} corrupted or incompatible files:")
        for name, reason in corrupted[:5]: # Show top 5 errors
            print(f"  - {name}: {reason}")
    else:
        print("\nAll files are compatible with CGCNN inputs.")

if __name__ == "__main__":
    verify_cifs()
