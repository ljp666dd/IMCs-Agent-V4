import json
import os
import sys

FILE_PATH = r"C:\Users\Administrator\Desktop\课题2-有序合金-HOR\IMCs\IMCs\data\theory\mp_data_summary.json"

def analyze():
    print(f"Analyzing {FILE_PATH}...")
    
    if not os.path.exists(FILE_PATH):
        print("Error: File not found.")
        return

    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    if isinstance(data, dict):
        print(f"Structure: Dict with {len(data)} keys.")
        sample_key = list(data.keys())[0]
        sample_val = data[sample_key]
        print(f"Sample Key: {sample_key}")
        print(f"Sample Value Type: {type(sample_val)}")
        if isinstance(sample_val, dict):
            print(f"Sample Value Keys: {list(sample_val.keys())}")
            # Check for d-band center
            if 'd_band_center' in sample_val:
                print(f"Sample d_band_center: {sample_val['d_band_center']}")
    elif isinstance(data, list):
        print(f"Structure: List with {len(data)} items.")
        if len(data) > 0:
            sample = data[0]
            print(f"Sample Item Keys: {list(sample.keys())}")
            if 'material_id' in sample:
                print(f"Sample ID: {sample['material_id']}")
    else:
        print(f"Unknown structure: {type(data)}")

    # Specific check for extended features
    count_dos = 0
    count_center = 0
    keys = data.keys() if isinstance(data, dict) else range(len(data))
    
    # Iterate to count
    if isinstance(data, dict):
        iterable = data.values()
    else:
        iterable = data
        
    for item in iterable:
        if isinstance(item, dict):
            if 'd_band_center' in item:
                count_center += 1
            if 'dos_data' in item or 'densities' in item: # heuristic for raw DOS
                count_dos += 1
                
    print(f"\nStatistics:")
    print(f"  Total Entries included: {len(data)}")
    print(f"  Entries with d_band_center: {count_center}")
    print(f"  Entries with Raw/Detailed DOS keys: {count_dos}")

if __name__ == "__main__":
    analyze()
