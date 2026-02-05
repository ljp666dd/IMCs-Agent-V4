import sqlite3
import re

DB_PATH = 'data/imcs.db'

ALLOWED_ELEMENTS = [
    "Pt", "Pd", "Au", "Ag", "Ir", "Rh", "Ru", "Os",
    "Ni", "Co", "Fe", "Cu", "Mn", "Cr", "V", "Ti", "Zn", "Sc",
    "Mo", "W", "Nb", "Ta", "Zr", "Hf", "Re", "Y",
    "Cd", "In", "Sn", "Ga", "Ge", "Al", "La", "Ce"
]

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. 检查总数和 NULL dos
cursor.execute('SELECT COUNT(*) FROM materials')
total = cursor.fetchone()[0]
print(f"Total materials: {total}")

cursor.execute('SELECT COUNT(*) FROM materials WHERE dos_data IS NULL')
null_dos = cursor.fetchone()[0]
print(f"Materials with dos_data IS NULL: {null_dos}")

# 2. 检查 filtering 逻辑
cursor.execute("SELECT material_id, formula FROM materials WHERE dos_data IS NULL LIMIT 20")
rows = cursor.fetchall()
print(f"Checking {len(rows)} samples...")

allowed_set = set(ALLOWED_ELEMENTS)

for mid, formula in rows:
    elements = set(re.findall(r'([A-Z][a-z]?)', formula))
    is_allowed = elements.issubset(allowed_set)
    print(f"  {mid} ({formula}): Elements={elements}, Allowed={is_allowed}")
    if not is_allowed:
        print(f"    Difference: {elements - allowed_set}")

conn.close()
