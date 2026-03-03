import sqlite3

def check_schema():
    conn = sqlite3.connect('data/imcs.db')
    cursor = conn.cursor()
    
    # 1. Get Table Schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='adsorption_energies'")
    schema = cursor.fetchone()
    if schema:
        print("Schema:")
        print(schema[0])
    
    # 2. Check Duplicate Counts
    print("\nChecking for duplicates (group by material_id, surface_composition, facet, adsorbate):")
    query = """
    SELECT material_id, surface_composition, facet, adsorbate, COUNT(*) 
    FROM adsorption_energies 
    GROUP BY material_id, surface_composition, facet, adsorbate 
    HAVING COUNT(*) > 1
    ORDER BY COUNT(*) DESC
    LIMIT 10
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    if rows:
        print(f"Found {len(rows)} duplicate groups (showing top 10):")
        for row in rows:
            print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]} : {row[4]} entries")
    else:
        print("No exact duplicates found by grouping key.")

    # 3. Check simple duplication by just material_id + adsorbate (since we saw ID 'Pt')
    print("\nChecking duplicates by (material_id, adsorbate) only:")
    query_simple = """
    SELECT material_id, adsorbate, COUNT(*)
    FROM adsorption_energies
    GROUP BY material_id, adsorbate
    HAVING COUNT(*) > 1
    ORDER BY COUNT(*) DESC
    LIMIT 10
    """
    cursor.execute(query_simple)
    rows = cursor.fetchall()
    for row in rows:
         print(f"  {row[0]} | {row[1]} : {row[2]} entries")

    conn.close()

if __name__ == "__main__":
    check_schema()
