"""
Fetch Full DOS Data from Materials Project
Downloads complete DOS curves (energy + density arrays) for all 798 materials.
Run in terminal: python src/data_ingestion/fetch_full_dos.py
"""

import os
import json
from mp_api.client import MPRester
from tqdm import tqdm
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# API Key (同 fetch_mp_data.py)
API_KEY = "3ilURTMlr6NpX206DwWWFE3ftG00RCuW"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "theory")
CIF_DIR = os.path.join(DATA_DIR, "cifs")
OUTPUT_FILE = os.path.join(DATA_DIR, "full_dos_data.json")

def get_material_ids():
    """Get list of material IDs from existing CIF files."""
    cif_files = [f for f in os.listdir(CIF_DIR) if f.endswith(".cif")]
    mat_ids = [f.replace(".cif", "") for f in cif_files]
    return mat_ids

def fetch_full_dos():
    """Fetch complete DOS data from Materials Project API."""
    
    mat_ids = get_material_ids()
    print(f"Found {len(mat_ids)} materials in CIF directory")
    
    # Load existing data if resuming
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            existing_data = json.load(f)
        processed_ids = set(existing_data.keys())
        print(f"Resuming: {len(processed_ids)} already processed")
    else:
        existing_data = {}
        processed_ids = set()
    
    dos_data = existing_data.copy()
    failed_ids = []
    
    print(f"\nFetching DOS data from Materials Project...")
    print("This may take 30-60 minutes depending on network speed.\n")
    
    with MPRester(API_KEY) as mpr:
        for mat_id in tqdm(mat_ids, desc="Fetching DOS"):
            if mat_id in processed_ids:
                continue
            
            try:
                # Use the correct method to get DOS
                complete_dos = mpr.get_dos_by_material_id(mat_id)
                
                if complete_dos is None:
                    failed_ids.append(mat_id)
                    continue
                
                # Extract data from CompleteDos object
                energies = complete_dos.energies.tolist()
                efermi = float(complete_dos.efermi)
                
                # Get total DOS densities (summing over spins if needed)
                densities_dict = {}
                for spin, dens in complete_dos.densities.items():
                    spin_key = str(spin)
                    densities_dict[spin_key] = dens.tolist()
                
                dos_data[mat_id] = {
                    "material_id": mat_id,
                    "energies": energies,
                    "densities": densities_dict,
                    "efermi": efermi,
                    "num_points": len(energies)
                }
                    
            except Exception as e:
                failed_ids.append(mat_id)
                # Print error occasionally for debugging
                if len(failed_ids) <= 5:
                    print(f"\nError for {mat_id}: {type(e).__name__}: {str(e)[:100]}")
                continue
            
            # Save periodically (every 50 materials)
            if len(dos_data) % 50 == 0 and len(dos_data) > len(processed_ids):
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(dos_data, f)
                print(f"  [Checkpoint saved: {len(dos_data)} materials]")
    
    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dos_data, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOS Fetch Complete!")
    print("=" * 60)
    print(f"Successfully fetched: {len(dos_data)} materials")
    print(f"Failed: {len(failed_ids)} materials")
    print(f"Data saved to: {OUTPUT_FILE}")
    
    if dos_data:
        sample_id = list(dos_data.keys())[0]
        sample = dos_data[sample_id]
        print(f"\nSample DOS ({sample_id}):")
        print(f"  Energy points: {sample['num_points']}")
        print(f"  Energy range: {min(sample['energies']):.2f} to {max(sample['energies']):.2f} eV")
        print(f"  Fermi energy: {sample['efermi']:.2f} eV")
    
    if failed_ids:
        failed_file = os.path.join(DATA_DIR, "failed_dos_fetch.txt")
        with open(failed_file, 'w') as f:
            f.write("\n".join(failed_ids))
        print(f"\nFailed IDs saved to: {failed_file}")

if __name__ == "__main__":
    fetch_full_dos()
