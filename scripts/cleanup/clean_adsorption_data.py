import sqlite3

def clean_duplicates():
    conn = sqlite3.connect('data/imcs.db')
    cursor = conn.cursor()
    
    print("Beginning duplication cleanup...")
    
    # 1. Count before
    cursor.execute("SELECT COUNT(*) FROM adsorption_energies")
    before = cursor.fetchone()[0]
    print(f"Total rows before: {before}")
    
    # 2. Create Temp Table with DISTINCT data
    # We include all columns to ensure we don't lose unique metadata if any
    cursor.execute("""
    CREATE TABLE adsorption_energies_cleaned AS
    SELECT material_id, surface_composition, facet, adsorbate, reaction_energy, activation_energy, source, metadata, MIN(created_at) as created_at
    FROM adsorption_energies
    GROUP BY material_id, surface_composition, facet, adsorbate, reaction_energy
    """)
    
    cursor.execute("SELECT COUNT(*) FROM adsorption_energies_cleaned")
    after = cursor.fetchone()[0]
    print(f"Total rows distinct: {after}")
    
    if after < before:
        print(f"Removing {before - after} duplicate rows...")
        
        # 3. Swap Tables
        cursor.execute("DROP TABLE adsorption_energies")
        
        # Re-create table with proper schema and constraints including ID
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
        
        # Move data back
        cursor.execute("""
        INSERT INTO adsorption_energies (material_id, surface_composition, facet, adsorbate, reaction_energy, activation_energy, source, metadata, created_at)
        SELECT material_id, surface_composition, facet, adsorbate, reaction_energy, activation_energy, source, metadata, created_at
        FROM adsorption_energies_cleaned
        """)
        
        cursor.execute("DROP TABLE adsorption_energies_cleaned")
        
        conn.commit()
        print("Cleanup successful. Unique constraint added.")
    else:
        print("No duplicates found? (Unexpected)")
        cursor.execute("DROP TABLE adsorption_energies_cleaned")
        
    conn.close()

if __name__ == "__main__":
    clean_duplicates()
