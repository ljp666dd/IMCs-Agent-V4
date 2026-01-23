import os
import json
import numpy as np
import sys
from mp_api.client import MPRester
from tqdm import tqdm
import warnings

# Add project root to path to import utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.dos_processing import get_dos_fingerprint, calculate_electronic_descriptors

# Suppress warnings
warnings.filterwarnings("ignore")

# API Key
API_KEY = "3ilURTMlr6NpX206DwWWFE3ftG00RCuW"

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data", "theory")
JSON_SUMMARY_FILE = os.path.join(DATA_DIR, "mp_data_summary.json")
OUTPUT_DOS_FILE = os.path.join(DATA_DIR, "dos_features.json")

def fetch_and_process_dos():
    # 1. Load the list of materials we already found
    if not os.path.exists(JSON_SUMMARY_FILE):
        print("Summary JSON not found. Please run fetch_mp_data.py first.")
        return
        
    with open(JSON_SUMMARY_FILE, "r") as f:
        materials_list = json.load(f)
        
    # Check for existing progress
    existing_data = []
    processed_ids = set()
    if os.path.exists(OUTPUT_DOS_FILE):
        try:
            with open(OUTPUT_DOS_FILE, "r") as f:
                existing_data = json.load(f)
            processed_ids = {item["material_id"] for item in existing_data}
            print(f"Found existing data with {len(existing_data)} entries. Resuming...")
        except:
            print("Existing DOS file corrupted or empty. Starting fresh.")
    
    # Filter list
    to_process = [m for m in materials_list if m["material_id"] not in processed_ids]
    
    print(f"Total candidates: {len(materials_list)}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining to fetch: {len(to_process)}")
    
    if not to_process:
        print("All materials processed!")
        return

    results = existing_data
    failed_count = 0
    save_interval = 20
    
    with MPRester(API_KEY) as mpr:
        for i, item in enumerate(tqdm(to_process, desc="Fetching remaining DOS")):
            mat_id = item["material_id"]
            formula = item["formula"]
            
            try:
                # Fetch DOS
                dos = mpr.get_dos_by_material_id(mat_id)
                
                if dos is None:
                    # print(f"No DOS found for {formula} ({mat_id})")
                    failed_count += 1
                    continue
                    
                # 2. Process - Fingerprint (400 bins)
                fingerprint = get_dos_fingerprint(dos, n_bins=400, e_min=-5.0, e_max=5.0)
                
                # 3. Process - 11 Descriptors
                target_element = None
                valid_tms = ["Pt", "Pd", "Ru", "Ir", "Rh", "Au", "Ag", "Cu", "Co", "Ni", "Fe", "Mn", "Cr", "V", "Yi", "Ti"]
                
                for tm in valid_tms:
                    if tm in formula:
                        target_element = tm
                        break
                
                descriptors = calculate_electronic_descriptors(dos, element_symbol=target_element)
                
                # 4. Save
                entry = {
                    "material_id": mat_id,
                    "formula": formula,
                    "dos_fingerprint": fingerprint.tolist(), 
                    "descriptors": descriptors,
                    "target_element_used": target_element
                }
                
                results.append(entry)
                
                # Periodic Save
                if (i + 1) % save_interval == 0:
                    with open(OUTPUT_DOS_FILE, "w") as f:
                        json.dump(results, f, indent=4)

            except Exception as e:
                # print(f"Error fetching/processing DOS for {mat_id}: {e}")
                failed_count += 1
                continue

    # Final Save
    with open(OUTPUT_DOS_FILE, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Successfully processed {len(results)} total materials.")
    print(f"Failed in this run: {failed_count}")
    print(f"Output saved to {OUTPUT_DOS_FILE}")

if __name__ == "__main__":
    fetch_and_process_dos()
