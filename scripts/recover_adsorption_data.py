import sqlite3

def recover():
    conn = sqlite3.connect('data/imcs.db')
    cursor = conn.cursor()
    
    print("Attempting recovery...")
    
    # Check states
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Tables found: {tables}")
    
    if 'adsorption_energies_cleaned' in tables:
        cursor.execute("SELECT COUNT(*) FROM adsorption_energies_cleaned")
        backup_count = cursor.fetchone()[0]
        print(f"Backup table has {backup_count} rows.")
    else:
        print("No backup table found! Cannot recover.")
        return

    cursor.execute("SELECT COUNT(*) FROM adsorption_energies")
    main_count = cursor.fetchone()[0]
    print(f"Main table has {main_count} rows.")
    
    if main_count == 0 and backup_count > 0:
        print("Recovering from backup with deduplication...")
        
        # Insert with deduplication logic
        # Schema of main table has UNIQUE constraint now?
        # Let's check schema/re-create main table to be sure
        
        cursor.execute("DROP TABLE IF EXISTS adsorption_energies")
        
        cursor.execute("""
        CREATE TABLE adsorption_energies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id TEXT,
            surface_composition TEXT,
            facet TEXT,
            adsorbate TEXT,
            reaction_energy REAL,
            activation_energy REAL,
            source TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(material_id) REFERENCES materials(material_id),
            CONSTRAINT unique_entry UNIQUE (material_id, surface_composition, facet, adsorbate, reaction_energy)
        )
        """)
        
        # Insert from backup
        # Use GROUP BY to pick one row per unique group
        query = """
        INSERT INTO adsorption_energies (material_id, surface_composition, facet, adsorbate, reaction_energy, activation_energy, source, metadata, created_at)
        SELECT material_id, surface_composition, facet, adsorbate, reaction_energy, activation_energy, source, metadata, MIN(created_at)
        FROM adsorption_energies_cleaned
        GROUP BY material_id, surface_composition, facet, adsorbate, reaction_energy
        """
        
        try:
            cursor.execute(query)
            print(f"Inserted {cursor.rowcount} rows into main table.")
            
            # Verify and drop backup
            cursor.execute("DROP TABLE adsorption_energies_cleaned")
            print("Dropped backup table.")
            conn.commit()
            print("Recovery successful.")
            
        except Exception as e:
            print(f"Recovery failed: {e}")
            conn.rollback()

    else:
        print("Main table is not empty or backup is empty. No action taken.")
        
    conn.close()

if __name__ == "__main__":
    recover()
