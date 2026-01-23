"""
Fetch Formation Energy for All Materials
Re-fetches formation energy per atom from Materials Project for all 1086 CIF materials.
"""

import os
import json
from mp_api.client import MPRester
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# API Key
API_KEY = "3ilURTMlr6NpX206DwWWFE3ftG00RCuW"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "theory")
CIF_DIR = os.path.join(DATA_DIR, "cifs")
OUTPUT_FILE = os.path.join(DATA_DIR, "formation_energy_full.json")


def get_material_ids():
    """Get list of material IDs from existing CIF files."""
    cif_files = [f for f in os.listdir(CIF_DIR) if f.endswith(".cif")]
    mat_ids = [f.replace(".cif", "") for f in cif_files]
    return mat_ids


def fetch_formation_energy():
    """Fetch formation energy for all materials."""
    
    mat_ids = get_material_ids()
    print(f"Found {len(mat_ids)} materials in CIF directory")
    
    results = []
    failed_ids = []
    
    print("\nFetching formation energy from Materials Project...")
    
    with MPRester(API_KEY) as mpr:
        # Fetch in batches for efficiency
        batch_size = 100
        
        for i in tqdm(range(0, len(mat_ids), batch_size), desc="Fetching batches"):
            batch_ids = mat_ids[i:i + batch_size]
            
            try:
                # Query summary data for batch
                docs = mpr.summary.search(
                    material_ids=batch_ids,
                    fields=["material_id", "formula_pretty", "composition", 
                           "formation_energy_per_atom", "energy_above_hull",
                           "symmetry"]
                )
                
                for doc in docs:
                    mat_id = str(doc.material_id)
                    
                    # Get space group info
                    sg_number = doc.symmetry.number if doc.symmetry else None
                    sg_symbol = doc.symmetry.symbol if doc.symmetry else None
                    
                    results.append({
                        "material_id": mat_id,
                        "formula": doc.formula_pretty,
                        "formation_energy": doc.formation_energy_per_atom,
                        "energy_above_hull": doc.energy_above_hull,
                        "space_group_number": sg_number,
                        "space_group_symbol": sg_symbol
                    })
                    
            except Exception as e:
                print(f"\nBatch error: {e}")
                failed_ids.extend(batch_ids)
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create lookup map
    fetched_ids = set(r["material_id"] for r in results)
    missing = [m for m in mat_ids if m not in fetched_ids]
    
    # Summary
    print("\n" + "=" * 60)
    print("Formation Energy Fetch Complete!")
    print("=" * 60)
    print(f"Successfully fetched: {len(results)} materials")
    print(f"Missing: {len(missing)} materials")
    print(f"Output saved to: {OUTPUT_FILE}")
    
    if results:
        fe_values = [r["formation_energy"] for r in results if r["formation_energy"] is not None]
        print(f"\nFormation Energy Statistics:")
        print(f"  Mean: {sum(fe_values)/len(fe_values):.3f} eV/atom")
        print(f"  Range: [{min(fe_values):.3f}, {max(fe_values):.3f}] eV/atom")
    
    if missing:
        missing_file = os.path.join(DATA_DIR, "formation_energy_missing.txt")
        with open(missing_file, 'w') as f:
            f.write("\n".join(missing))
        print(f"\nMissing IDs saved to: {missing_file}")


if __name__ == "__main__":
    fetch_formation_energy()
