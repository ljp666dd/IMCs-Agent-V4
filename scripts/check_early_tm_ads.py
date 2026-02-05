import sqlite3

def check():
    conn = sqlite3.connect('data/imcs.db')
    cursor = conn.cursor()
    
    # Check for metals that typically bind H strongly
    strong_binders = ['Ti', 'Zr', 'Hf', 'Nb', 'Ta', 'Sc', 'Y']
    
    print("Checking H* adsorption coverage for Early TMs:")
    for metal in strong_binders:
        cursor.execute(f"SELECT COUNT(*) FROM adsorption_energies WHERE (surface_composition LIKE '%{metal}%') AND adsorbate='H*'")
        count = cursor.fetchone()[0]
        print(f"  {metal}: {count} entries")
        
        if count > 0:
            cursor.execute(f"SELECT reaction_energy FROM adsorption_energies WHERE (surface_composition LIKE '%{metal}%') AND adsorbate='H*' LIMIT 3")
            sample = [f"{x[0]:.2f}" for x in cursor.fetchall()]
            print(f"    Sample E_ads: {sample} eV")

    conn.close()

if __name__ == "__main__":
    check()
