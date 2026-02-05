"""
从 Catalysis-Hub 获取 H 和 OH 吸附能数据
"""

import sqlite3
import requests
import json
from typing import List, Dict

DB_PATH = 'data/imcs.db'

def fetch_catalysis_hub(adsorbate: str, limit: int = 1000) -> List[Dict]:
    print(f"\nFetching {adsorbate} adsorption data from Catalysis-Hub (limit={limit})...")
    
    url = "https://api.catalysis-hub.org/graphql"
    
    # Query for reactions involving the adsorbate
    # We look for reactions where the adsorbate is a product (desorption) or reactant (adsorption)
    # Ideally we want adsorption energy.
    # CatHub often lists "H star" or simply searches by chemical composition.
    # Let's try searching for reactions with the specific adsorbate.
    
    query = """
    {
      reactions(first: %d, reactants: "%s") {
        edges {
          node {
            Equation
            reactionEnergy
            activationEnergy
            surfaceComposition
            facet
            chemicalComposition
          }
        }
      }
    }
    """ % (limit, adsorbate)
    
    results = []
    
    try:
        response = requests.post(url, json={"query": query}, timeout=60)
        if response.status_code == 200:
            data = response.json()
            edges = data.get("data", {}).get("reactions", {}).get("edges", [])
            
            for edge in edges:
                node = edge.get("node", {})
                results.append({
                    "equation": node.get("Equation", ""),
                    "reaction_energy": node.get("reactionEnergy"),
                    "activation_energy": node.get("activationEnergy"),
                    "surface": node.get("surfaceComposition", ""),
                    "facet": node.get("facet", ""),
                    "composition": node.get("chemicalComposition", "")
                })
            
            print(f"  Found {len(results)} records.")
        else:
            print(f"  Error: {response.status_code}")
            print(response.text[:200])
            
    except Exception as e:
        print(f"  Failed: {e}")
    
    return results

def save_to_db(adsorbate: str, data: List[Dict]):
    if not data:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    for item in data:
        if item.get('reaction_energy') is not None:
            # Simple mapping: use composition as material_id hint??
            # No, 'material_id' in DB expects 'mp-...' or 'lit:...'.
            # CatHub doesn't provide MP IDs directly usually.
            # We store the surface composition and map later.
            
            cursor.execute('''
                INSERT OR REPLACE INTO adsorption_energies 
                (material_id, surface_composition, facet, adsorbate, reaction_energy, activation_energy, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.get('composition', 'unknown'), # Use composition as temp ID
                item.get('surface', ''),
                item.get('facet', ''),
                adsorbate,
                item.get('reaction_energy'),
                item.get('activation_energy'),
                'Catalysis-Hub'
            ))
            count += 1
            
    conn.commit()
    print(f"  Saved {count} records for {adsorbate}.")
    conn.close()

def main():
    # Fetch H
    h_data = fetch_catalysis_hub("H", limit=1000)
    save_to_db("H", h_data)
    
    # Fetch OH
    # Try "OH"
    oh_data = fetch_catalysis_hub("OH", limit=1000)
    save_to_db("OH", oh_data)
    
    # Verify
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT adsorbate, COUNT(*) FROM adsorption_energies GROUP BY adsorbate")
    print("\nCurrent Database Statistics:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")
    conn.close()

if __name__ == "__main__":
    main()
