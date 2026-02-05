import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

from src.agents.core.theory_agent import TheoryDataAgent

def verify():
    print("Initializing TheoryDataAgent...")
    agent = TheoryDataAgent()
    
    mid = 'mp-126' # Pt (Known to have ads data)
    print(f"\nFetching details for {mid}...")
    
    details = agent.get_material_details(mid)
    
    if not details:
        print("❌ Material not found!")
        return
        
    print(f"✅ Material Found: {details.get('formula')}")
    
    # 1. Check Formation Energy
    fe = details.get('formation_energy')
    print(f"   Formation Energy: {fe} eV/atom")
    
    # 2. Check DOS Data
    dos_data = details.get('dos_data')
    if dos_data:
        print(f"   DOS Data Found (Type: {type(dos_data)})")
        if isinstance(dos_data, str):
            print("   ⚠️ DOS Data is STRING (not parsed). Parsing now...")
            try:
                dos_data = json.loads(dos_data)
                print("   ✅ Parsed successfully.")
            except:
                print("   ❌ Parsing failed.")
        
        center = dos_data.get('d_band_center')
        print(f"   d-band Center: {center} eV")
    else:
        print("   ❌ No DOS Data found.")
        
    # 3. Check Adsorption Energies
    ads = details.get('adsorption_energies')
    if ads:
        print(f"   Adsorption Energies Found: {len(ads)} records")
        for r in ads[:3]:
            print(f"     - {r['adsorbate']} on {r['surface_composition']} ({r['facet']}): {r['reaction_energy']} eV")
    else:
        print("   ❌ No Adsorption Energy linked.")

if __name__ == "__main__":
    verify()
