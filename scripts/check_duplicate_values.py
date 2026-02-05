import sqlite3

def check_values():
    conn = sqlite3.connect('data/imcs.db')
    cursor = conn.cursor()
    
    # Check variance in duplicates for one example
    print("Checking variance for 'lit:Ni1' H* adsorption:")
    cursor.execute("""
        SELECT reaction_energy, COUNT(*) 
        FROM adsorption_energies 
        WHERE material_id='lit:Ni1' AND adsorbate='H*'
        GROUP BY reaction_energy
    """)
    rows = cursor.fetchall()
    for r in rows:
        print(f"  Energy {r[0]:.4f} eV: {r[1]} copies")
        
    conn.close()

if __name__ == "__main__":
    check_values()
