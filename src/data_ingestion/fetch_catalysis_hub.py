"""
Catalysis-Hub Adsorption Energy Fetcher
Fetches H and OH adsorption energies from Catalysis-Hub GraphQL API.
Matches with our existing ordered alloy CIF data by composition.
"""

import os
import json
import requests
from collections import defaultdict

# Catalysis-Hub GraphQL endpoint
API_ENDPOINT = "https://api.catalysis-hub.org/graphql"

# Target elements (same as our CIF data)
TARGET_ELEMENTS = {
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Y", "Zr", "Nb", "Mo", 
    "Ru", "Pd", "Cd", "In", "Sn", "La", "Ce", "Pr", "Nd", "Ta", "W", "Pt"
}

# GraphQL query for reactions (will filter H/OH in post-processing)
QUERY_TEMPLATE = """
query {{
  reactions(first: {limit}, after: "{cursor}") {{
    totalCount
    pageInfo {{
      hasNextPage
      endCursor
    }}
    edges {{
      node {{
        id
        reactionEnergy
        activationEnergy
        surfaceComposition
        chemicalComposition
        facet
        sites
        reactants
        products
        Equation
        pubId
      }}
    }}
  }}
}}
"""

def fetch_all_reactions(limit=1000, max_pages=50):
    """Fetch all reaction data from Catalysis-Hub."""
    all_data = []
    cursor = ""
    page = 0
    
    print(f"Fetching reaction data from Catalysis-Hub...")
    
    while page < max_pages:
        query = QUERY_TEMPLATE.format(limit=limit, cursor=cursor)
        
        try:
            response = requests.post(
                API_ENDPOINT,
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                print(f"GraphQL Error: {data['errors']}")
                break
                
            reactions = data.get("data", {}).get("reactions", {})
            edges = reactions.get("edges", [])
            
            if not edges:
                print("No more data found.")
                break
                
            for edge in edges:
                node = edge.get("node", {})
                all_data.append({
                    "id": node.get("id"),
                    "reaction_energy": node.get("reactionEnergy"),
                    "activation_energy": node.get("activationEnergy"),
                    "surface_composition": node.get("surfaceComposition"),
                    "chemical_composition": node.get("chemicalComposition"),
                    "facet": node.get("facet"),
                    "sites": node.get("sites"),
                    "reactants": node.get("reactants"),
                    "products": node.get("products"),
                    "equation": node.get("Equation"),
                    "pub_id": node.get("pubId")
                })
            
            page_info = reactions.get("pageInfo", {})
            if not page_info.get("hasNextPage", False):
                print("Reached last page.")
                break
                
            cursor = page_info.get("endCursor", "")
            page += 1
            print(f"  Page {page}: {len(all_data)} records so far...")
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
    
    print(f"Total records fetched: {len(all_data)}")
    return all_data

def filter_by_adsorbate(data, adsorbate="H"):
    """Filter reactions by adsorbate (H or OH) based on equation/reactants."""
    filtered = []
    
    for record in data:
        equation = record.get("equation", "") or ""
        reactants = record.get("reactants", "") or ""
        products = record.get("products", "") or ""
        
        # Look for adsorption reactions: X* -> X (desorption) or X -> X* (adsorption)
        # For H: look for "Hstar" or "H*" in equation
        # For OH: look for "OHstar" or "OH*"
        
        search_terms = []
        if adsorbate == "H":
            search_terms = ["Hstar", "H*", "H(star)", "Hydrogen", "->H", "H->"]
        elif adsorbate == "OH":
            search_terms = ["OHstar", "OH*", "OH(star)", "->OH", "OH->"]
        
        combined_text = f"{equation} {reactants} {products}".lower()
        
        if any(term.lower() in combined_text for term in search_terms):
            record["adsorbate"] = adsorbate
            filtered.append(record)
    
    return filtered

def filter_by_target_elements(data, target_elements):
    """Filter data to only include surfaces with target elements."""
    filtered = []
    
    for record in data:
        composition = record.get("surface_composition", "")
        if not composition:
            continue
            
        # Parse composition (e.g., "Pt3Co" -> {Pt, Co})
        # Simple heuristic: extract capital letters followed by lowercase
        import re
        elements = set(re.findall(r'[A-Z][a-z]?', composition))
        
        # Check if all elements are in target set
        if elements and elements.issubset(target_elements):
            filtered.append(record)
    
    return filtered

def load_existing_cif_data():
    """Load existing CIF metadata to find matching compositions."""
    summary_path = os.path.join("data", "theory", "mp_data_summary.json")
    
    if not os.path.exists(summary_path):
        print(f"CIF summary not found at {summary_path}")
        return {}
    
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    # Create composition -> material_id mapping
    formula_map = {}
    for item in data:
        formula = item.get("formula", "")
        mat_id = item.get("material_id", "")
        if formula and mat_id:
            formula_map[formula] = mat_id
    
    return formula_map

def match_with_cif_data(adsorption_data, formula_map):
    """Match adsorption data with our CIF data by composition."""
    matched = []
    unmatched_compositions = set()
    
    for record in adsorption_data:
        composition = record.get("surface_composition", "")
        
        # Try direct match first
        if composition in formula_map:
            record["matched_material_id"] = formula_map[composition]
            matched.append(record)
        else:
            # Try normalized match (sort elements alphabetically)
            import re
            from pymatgen.core.composition import Composition
            try:
                comp = Composition(composition)
                normalized = comp.reduced_formula
                if normalized in formula_map:
                    record["matched_material_id"] = formula_map[normalized]
                    matched.append(record)
                else:
                    unmatched_compositions.add(composition)
            except:
                unmatched_compositions.add(composition)
    
    return matched, unmatched_compositions

def main():
    # 1. Fetch all reaction data
    all_reactions = fetch_all_reactions(limit=500, max_pages=30)
    
    if not all_reactions:
        print("No data fetched. Exiting.")
        return
    
    # 2. Filter by adsorbate type
    h_data = filter_by_adsorbate(all_reactions, adsorbate="H")
    oh_data = filter_by_adsorbate(all_reactions, adsorbate="OH")
    
    print(f"\nFiltered H adsorption: {len(h_data)} records")
    print(f"Filtered OH adsorption: {len(oh_data)} records")
    
    # 3. Filter by target elements
    h_filtered = filter_by_target_elements(h_data, TARGET_ELEMENTS)
    oh_filtered = filter_by_target_elements(oh_data, TARGET_ELEMENTS)
    
    print(f"Filtered H (target elements): {len(h_filtered)} records")
    print(f"Filtered OH (target elements): {len(oh_filtered)} records")
    
    # 4. Load existing CIF data
    formula_map = load_existing_cif_data()
    print(f"Loaded {len(formula_map)} CIF compositions for matching")
    
    # 5. Match with CIF data
    h_matched, h_unmatched = match_with_cif_data(h_filtered, formula_map)
    oh_matched, oh_unmatched = match_with_cif_data(oh_filtered, formula_map)
    
    print(f"\nMatched H records: {len(h_matched)}")
    print(f"Matched OH records: {len(oh_matched)}")
    
    if h_unmatched:
        print(f"Sample unmatched H compositions: {list(h_unmatched)[:5]}")
    if oh_unmatched:
        print(f"Sample unmatched OH compositions: {list(oh_unmatched)[:5]}")
    
    # 6. Combine and save
    output_dir = os.path.join("data", "adsorption")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all fetched data
    with open(os.path.join(output_dir, "h_adsorption_all.json"), 'w') as f:
        json.dump(h_filtered, f, indent=2)
    
    with open(os.path.join(output_dir, "oh_adsorption_all.json"), 'w') as f:
        json.dump(oh_filtered, f, indent=2)
    
    # Save matched data
    with open(os.path.join(output_dir, "h_adsorption_matched.json"), 'w') as f:
        json.dump(h_matched, f, indent=2)
    
    with open(os.path.join(output_dir, "oh_adsorption_matched.json"), 'w') as f:
        json.dump(oh_matched, f, indent=2)
    
    # Summary
    summary = {
        "total_reactions_fetched": len(all_reactions),
        "h_filtered_by_adsorbate": len(h_data),
        "h_filtered_by_elements": len(h_filtered),
        "h_matched": len(h_matched),
        "oh_filtered_by_adsorbate": len(oh_data),
        "oh_filtered_by_elements": len(oh_filtered),
        "oh_matched": len(oh_matched),
        "unique_matched_compositions": len(set(
            [r["surface_composition"] for r in h_matched] + 
            [r["surface_composition"] for r in oh_matched]
        ))
    }
    
    with open(os.path.join(output_dir, "fetch_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Fetch Summary:")
    print(json.dumps(summary, indent=2))
    print("=" * 50)
    print(f"Data saved to {output_dir}/")

if __name__ == "__main__":
    main()
