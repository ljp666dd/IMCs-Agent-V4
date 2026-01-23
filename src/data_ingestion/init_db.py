import sqlite3
import os

# Database file path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "her_catalysts.db")

def init_db():
    print(f"Initializing database at: {DB_PATH}")
    
    # Connect to SQLite (creates file if not exists)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Experiments Table
    # Stores real-world experimental results
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        formula TEXT NOT NULL,              -- e.g., Pt3Co
        structure_type TEXT,                -- e.g., L1_2, Disordered
        synthesis_method TEXT,              -- e.g., Solvothermal
        temperature_c REAL,                 -- Synthesis Temp
        annealing_time_h REAL,              -- Annealing Time (hrs) [NEW]
        precursor_ratio TEXT,               -- Precursor Ratio (e.g., "1:3") [NEW]
        
        -- Performance Metrics (HER in 1M KOH)
        overpotential_10ma REAL,            -- mV @ 10 mA/cm2
        tafel_slope REAL,                   -- mV/dec
        stability_decay_mv REAL,            -- Voltage change after 500 cycles (mV)
        
        -- Metadata
        batch_id TEXT,
        researcher TEXT,
        comments TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 2. Theory Table
    # Stores calculated properties from MP or local VASP
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS theory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_id TEXT UNIQUE,            -- e.g., mp-1234
        formula TEXT NOT NULL,
        space_group TEXT,                   -- e.g., Pm-3m
        
        -- Calculated Descriptors
        formation_energy_per_atom REAL,     -- eV/atom
        energy_above_hull REAL,             -- eV/atom
        delta_g_h REAL,                     -- Hydrogen Adsorption Free Energy (eV)
        d_band_center REAL,                 -- eV
        
        cif_path TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database tables created successfully.")

if __name__ == "__main__":
    init_db()
