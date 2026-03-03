import sqlite3
DB_PATH = 'data/imcs.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM adsorption_energies WHERE adsorbate = 'OH'")
print(f"OH: {cursor.fetchone()[0]}")
cursor.execute("SELECT COUNT(*) FROM adsorption_energies WHERE adsorbate = 'H'")
print(f"H: {cursor.fetchone()[0]}")
conn.close()
