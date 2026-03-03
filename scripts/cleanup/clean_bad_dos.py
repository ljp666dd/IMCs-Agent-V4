import sqlite3
import json

DB_PATH = 'data/imcs.db'

# IDs from the bad test run (Step 1196 output)
BAD_IDS = [
    "mp-126", "mp-1194", "mp-1206750", "mp-1670", "mp-2260", 
    "mp-2678", "mp-894", "mp-949", "mp-945"
]

def clean():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print(f"Cleaning {len(BAD_IDS)} bad DOS records...")
    
    for mid in BAD_IDS:
        # Check current value
        cursor.execute("SELECT dos_data FROM materials WHERE material_id=?", (mid,))
        row = cursor.fetchone()
        if row and row[0]:
            data = json.loads(row[0])
            center = data.get('d_band_center')
            print(f"  {mid}: current center = {center:.2f}")
            
            # Remove d_band features but keep others? 
            # Actually previous script updated `dos_data` with features.
            # If `dos_data` was NULL before, we should set it to NULL.
            # But we might have other data?
            # The script assumes `dos_data` stores the features.
            # Let's just set dos_data to NULL to force re-fetch.
            
            cursor.execute("UPDATE materials SET dos_data = NULL WHERE material_id=?", (mid,))
        else:
            print(f"  {mid}: already NULL or not found")
            
    conn.commit()
    print("Cleaned.")
    conn.close()

if __name__ == "__main__":
    clean()
