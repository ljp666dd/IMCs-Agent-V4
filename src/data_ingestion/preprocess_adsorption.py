"""
Preprocess Adsorption Energy Data
Aggregates H adsorption data to create material_id -> average ΔG_H mapping.
Filters for direct H adsorption reactions only.
"""

import os
import json
from collections import defaultdict

def preprocess_adsorption_data():
    # Paths
    input_file = "data/adsorption/h_adsorption_matched.json"
    output_file = "data/adsorption/h_adsorption_aggregated.json"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} H adsorption records")
    
    # Filter for direct H adsorption reactions
    # Look for: "0.5H2(g) + * -> H*" or "H2(g) + 2* -> 2H*"
    direct_h_reactions = []
    for record in data:
        equation = record.get("equation", "")
        
        # Check for direct H adsorption (not part of complex multi-species reactions)
        if "H*" in equation or "Hstar" in equation:
            # Prefer simple reactions
            if "0.5H2(g)" in equation and "+ * ->" in equation:
                # Single H adsorption: 0.5H2(g) + * -> H*
                record["reaction_type"] = "single_H"
                direct_h_reactions.append(record)
            elif "H2(g) + 2*" in equation and "2H*" in equation:
                # Double H adsorption: H2(g) + 2* -> 2H*
                # Energy per H = reaction_energy / 2
                record["reaction_type"] = "double_H"
                record["energy_per_h"] = record["reaction_energy"] / 2.0
                direct_h_reactions.append(record)
    
    print(f"Found {len(direct_h_reactions)} direct H adsorption reactions")
    
    # Aggregate by material_id
    mat_id_to_energies = defaultdict(list)
    for record in direct_h_reactions:
        mat_id = record.get("matched_material_id")
        if not mat_id:
            continue
        
        if record["reaction_type"] == "single_H":
            energy = record["reaction_energy"]
        else:
            energy = record.get("energy_per_h", record["reaction_energy"])
        
        mat_id_to_energies[mat_id].append({
            "energy": energy,
            "facet": record.get("facet"),
            "sites": record.get("sites"),
            "surface_composition": record.get("surface_composition")
        })
    
    print(f"Unique materials with H adsorption: {len(mat_id_to_energies)}")
    
    # Aggregate: take average across all sites/facets for each material
    aggregated = {}
    for mat_id, records in mat_id_to_energies.items():
        energies = [r["energy"] for r in records]
        avg_energy = sum(energies) / len(energies)
        
        aggregated[mat_id] = {
            "material_id": mat_id,
            "surface_composition": records[0]["surface_composition"],
            "delta_g_h": avg_energy,
            "num_records": len(records),
            "energy_std": (sum((e - avg_energy)**2 for e in energies) / len(energies))**0.5 if len(energies) > 1 else 0,
            "facets": list(set(r["facet"] for r in records if r["facet"]))
        }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nSaved aggregated data to {output_file}")
    
    # Statistics
    all_energies = [v["delta_g_h"] for v in aggregated.values()]
    print(f"\nStatistics:")
    print(f"  Materials: {len(aggregated)}")
    print(f"  ΔG_H Range: {min(all_energies):.3f} to {max(all_energies):.3f} eV")
    print(f"  ΔG_H Mean: {sum(all_energies)/len(all_energies):.3f} eV")
    
    # Show sample
    print("\nSample entries:")
    for i, (mat_id, entry) in enumerate(list(aggregated.items())[:5]):
        print(f"  {mat_id}: {entry['surface_composition']} -> ΔG_H = {entry['delta_g_h']:.4f} eV")
    
    return aggregated

if __name__ == "__main__":
    preprocess_adsorption_data()
