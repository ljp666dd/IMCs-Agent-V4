import os
import json
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# User provided API Key
API_KEY = "3ilURTMlr6NpX206DwWWFE3ftG00RCuW"

# Target Elements
ELEMENTS = [
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Y", "Zr", "Nb", "Mo", 
    "Ru", "Pd", "Cd", "In", "Sn", "La", "Ce", "Pr", "Nd", "Ta", "W", "Pt"
]

# Paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "theory")
CIF_DIR = os.path.join(DATA_DIR, "cifs")

def fetch_data():
    if not os.path.exists(CIF_DIR):
        os.makedirs(CIF_DIR)

    print(f"Connecting to Materials Project with API Key: {API_KEY[:4]}...")
    
    with MPRester(API_KEY) as mpr:
        print("Searching for Ordered Alloys (L10, L11, L12, B2, B11, B35) with 2-5 elements...")
        
        # Strategy: Iterate by single element to cover Binaries, Ternaries, Quaternaries, Quinaries
        # This avoids the combinatorial explosion of generating all 5-element combinations.
        
        # Common non-metals to exclude to reduce query size speed up processing
        EXCLUDED = ["O", "C", "N", "H", "F", "Cl", "S", "P", "Si", "Br", "I"]
        
        structure_counts = {}
        processed_ids = set() # To avoid duplicates if a material matches multiple element queries
        
        allowed_set = set(ELEMENTS)
        
        from tqdm import tqdm
        for el in tqdm(ELEMENTS, desc="Scanning elements"):
            try:
                # Fetch materials containing 'el', with 2-5 elements, excluding common anions
                docs = mpr.summary.search(
                    elements=[el],
                    num_elements=[2, 3, 4, 5],
                    exclude_elements=EXCLUDED, 
                    energy_above_hull=(0, 0.05),
                    fields=["material_id", "formula_pretty", "composition", "energy_above_hull", "formation_energy_per_atom", "structure", "symmetry"]
                )
                
                for doc in docs:
                    mat_id = str(doc.material_id)
                    if mat_id in processed_ids:
                        continue
                        
                    # 1. Strict Element Check
                    # Ensure ALL elements in this material are in our target list
                    # (The query only ensured 'el' is present and 'EXCLUDED' are not, but others like 'Li' might be there)
                    mat_elements = set(str(e) for e in doc.composition.elements)
                    if not mat_elements.issubset(allowed_set):
                        continue
                        
                    processed_ids.add(mat_id)
                    
                    # 2. Strict Structure Filter
                    sg_number = doc.symmetry.number if doc.symmetry else 0
                    structure_type = None
                    
                    # Mapping based on User Request
                    if sg_number == 123: 
                        structure_type = "L1_0"
                    elif sg_number == 166:
                        structure_type = "L1_1"
                    elif sg_number == 129:
                        structure_type = "B11 (Ga5Pt5-type)"
                    elif sg_number == 191:
                        structure_type = "B35 (CoSn-type)"
                    elif sg_number == 221:
                        # Differentiate L1_2 vs B2 based on composition stoichiometry
                        # B2 is typically 1:1 (AB). L12 is 1:3 (AB3).
                        amounts = sorted(list(doc.composition.get_el_amt_dict().values()))
                        total = sum(amounts)
                        ratios = [a/total for a in amounts]
                        
                        # Check for ~0.5/0.5 for B2
                        if any(0.45 < r < 0.55 for r in ratios) and len(ratios) == 2:
                             structure_type = "B2 (CsCl)"
                        # Check for ~0.25/0.75 for L12
                        elif any(0.20 < r < 0.30 for r in ratios):
                             structure_type = "L1_2"
                        else:
                             # If we can't be sure, strictly speaking we might skip, 
                             # or tag as "Ordered (SG 221)"
                             structure_type = "Ordered-SG221 (L1_2/B2)"
                    
                    # If not one of the above, SKIP (Exclude A1, A2, A3 etc)
                    if not structure_type:
                        continue
                        
                    # Save
                    structure_counts[structure_type] = structure_counts.get(structure_type, 0) + 1
                    
                    try:
                        cif_writer = CifWriter(doc.structure)
                        cif_path = os.path.join(CIF_DIR, f"{mat_id}.cif")
                        cif_writer.write_file(cif_path)
                    except Exception as e:
                        print(f"Error saving CIF {mat_id}: {e}")
                        continue

            except Exception as e:
                # print(f"Query error for {el}: {e}")
                continue
                
        print(f"Scan complete.")
        print("Found structures:")
        for k, v in structure_counts.items():
            print(f"  {k}: {v}")
            
    # No JSON summary needed for this step as we rely on the CIFs directory for training
    print(f"Total Unique Ordered Materials Downloaded: {len(processed_ids)}")

if __name__ == "__main__":
    fetch_data()
