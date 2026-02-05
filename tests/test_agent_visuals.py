import sys
import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend

sys.path.append(os.getcwd())
from src.agents.core.theory_agent import TheoryDataAgent

def test_upgrades():
    print("Initializing Agent...")
    agent = TheoryDataAgent()
    
    # 1. Get Materials
    print("\n[1. Fetching Candidates]")
    # Get materials that actually have DOS data
    all_mats = agent.db.list_materials(limit=2000)
    mats = [m for m in all_mats if m.get('dos_file_path')]
    
    if len(mats) < 2:
        # Fallback to hardcoded known ones if DB query fails to find them easily
        print("Warning: Auto-discovery low. Using mp-126 (Pt) and mp-2 (Pd) if available.")
        mats = [agent.get_material_details('mp-126'), agent.get_material_details('mp-2')]
        mats = [m for m in mats if m]
    
    if len(mats) < 2:
        print("Error: Not enough materials for comparison.")
        return
        
    id1 = mats[0]['material_id']
    id2 = mats[1]['material_id']
    print(f"Candidates: {id1} ({mats[0]['formula']}), {id2} ({mats[1]['formula']})")
    
    # 2. Test Compare
    print("\n[2. Testing Comparison Table]")
    table = agent.compare_materials([id1, id2])
    print("--- Table Output ---")
    print(table)
    print("--------------------")
    if "| Formula" in table and id1 in table:
        print("✅ Comparison Table Generated Successfully.")
    else:
        print("❌ Comparison Table Failed.")
        
    # 3. Test Plotting
    print(f"\n[3. Testing DOS Plot for {id1}]")
    output = agent.plot_dos(id1)
    if output and os.path.exists(output):
        print(f"✅ Plot Generated: {output}")
    else:
        print("❌ Plot Generation Failed.")

if __name__ == "__main__":
    test_upgrades()
