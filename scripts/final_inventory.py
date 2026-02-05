import sqlite3
import os
import json

DB_PATH = 'data/imcs.db'
DOS_RAW_DIR = 'data/theory/dos_raw'
CIF_DIR = 'data/theory/cifs'

def inventory():
    stats = {}
    
    # 1. File System
    stats['dos_files'] = len([f for f in os.listdir(DOS_RAW_DIR) if f.endswith('.json.gz')]) if os.path.exists(DOS_RAW_DIR) else 0
    stats['cif_files'] = len([f for f in os.listdir(CIF_DIR) if f.endswith('.cif')]) if os.path.exists(CIF_DIR) else 0
    
    # 2. Database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Materials
    cursor.execute("SELECT COUNT(*) FROM materials")
    stats['total_materials'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM materials WHERE formation_energy IS NOT NULL")
    stats['with_formation_energy'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM materials WHERE dos_data IS NOT NULL")
    stats['with_dos_features'] = cursor.fetchone()[0]
    
    # Adsorption
    cursor.execute("SELECT COUNT(*) FROM adsorption_energies WHERE adsorbate LIKE 'H%'")
    stats['h_adsorption'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM adsorption_energies WHERE adsorbate LIKE 'OH%'")
    stats['oh_adsorption'] = cursor.fetchone()[0]
    
    # Unique Materials with Adsorption
    cursor.execute("SELECT COUNT(DISTINCT material_id) FROM adsorption_energies")
    stats['materials_with_ads'] = cursor.fetchone()[0]

    # Models
    cursor.execute("SELECT COUNT(*) FROM models")
    stats['models'] = cursor.fetchone()[0]
    
    conn.close()
    
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    inventory()
