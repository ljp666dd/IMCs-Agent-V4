"""
Fetch Orbital-Projected DOS (PDOS) from Materials Project
Downloads complete DOS with s, p, d orbital decomposition for DOSnet compatibility.
Run in terminal: python src/data_ingestion/fetch_orbital_dos.py
"""

import os
import json
from mp_api.client import MPRester
from pymatgen.electronic_structure.core import OrbitalType
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# API Key
API_KEY = "3ilURTMlr6NpX206DwWWFE3ftG00RCuW"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "theory")
CIF_DIR = os.path.join(DATA_DIR, "cifs")
OUTPUT_FILE = os.path.join(DATA_DIR, "orbital_pdos.json")


def get_material_ids():
    """Get list of material IDs from existing CIF files."""
    cif_files = [f for f in os.listdir(CIF_DIR) if f.endswith(".cif")]
    mat_ids = [f.replace(".cif", "") for f in cif_files]
    return mat_ids


def extract_orbital_dos(complete_dos, energy_range=(-15, 10), n_points=2000):
    """
    Extract s, p, d orbital DOS from CompleteDos object.
    Interpolates to a standard energy grid relative to Fermi level.
    
    Args:
        complete_dos: pymatgen CompleteDos object
        energy_range: (min, max) relative to Fermi level
        n_points: Number of energy points in output grid
        
    Returns:
        dict with standardized orbital DOS
    """
    efermi = complete_dos.efermi
    
    # Create standardized energy grid (relative to Fermi level)
    standard_energies = np.linspace(energy_range[0], energy_range[1], n_points)
    
    # Get element-orbital decomposed DOS
    try:
        spd_dos = complete_dos.get_spd_dos()
    except Exception as e:
        return None
    
    # Initialize orbital DOS arrays
    s_dos = np.zeros(n_points)
    p_dos = np.zeros(n_points)
    d_dos = np.zeros(n_points)
    
    # Extract each orbital type
    for orbital_type, dos_obj in spd_dos.items():
        # Get original energies and densities (relative to Fermi)
        original_energies = dos_obj.energies - efermi
        
        # Sum over spins if spin-polarized
        if hasattr(dos_obj, 'densities'):
            total_dens = None
            for spin, dens in dos_obj.densities.items():
                if total_dens is None:
                    total_dens = np.array(dens)
                else:
                    total_dens = total_dens + np.array(dens)
        else:
            total_dens = np.array(dos_obj.get_densities())
        
        # Interpolate to standard grid
        interpolated = np.interp(standard_energies, original_energies, total_dens, left=0, right=0)
        
        # Assign to orbital type
        if orbital_type == OrbitalType.s:
            s_dos = interpolated
        elif orbital_type == OrbitalType.p:
            p_dos = interpolated
        elif orbital_type == OrbitalType.d:
            d_dos = interpolated
    
    # Total DOS
    total_dos = s_dos + p_dos + d_dos
    
    return {
        "energies": standard_energies.tolist(),
        "s_dos": s_dos.tolist(),
        "p_dos": p_dos.tolist(),
        "d_dos": d_dos.tolist(),
        "total_dos": total_dos.tolist(),
        "efermi": float(efermi),
        "n_points": n_points,
        "energy_range": list(energy_range)
    }


def fetch_orbital_dos():
    """Fetch orbital-projected DOS for all materials."""
    
    mat_ids = get_material_ids()
    print(f"Found {len(mat_ids)} materials in CIF directory")
    
    # Load existing data if resuming
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            existing_data = json.load(f)
        processed_ids = set(existing_data.keys())
        print(f"Resuming: {len(processed_ids)} already processed")
    else:
        existing_data = {}
        processed_ids = set()
    
    dos_data = existing_data.copy()
    failed_ids = []
    
    print(f"\nFetching orbital-projected DOS from Materials Project...")
    print("Energy range: -15 to +10 eV (relative to Fermi)")
    print("Grid points: 2000")
    print("This may take 1-2 hours...\n")
    
    with MPRester(API_KEY) as mpr:
        for mat_id in tqdm(mat_ids, desc="Fetching PDOS"):
            if mat_id in processed_ids:
                continue
            
            try:
                # Get CompleteDos object
                complete_dos = mpr.get_dos_by_material_id(mat_id)
                
                if complete_dos is None:
                    failed_ids.append(mat_id)
                    continue
                
                # Extract orbital DOS
                orbital_data = extract_orbital_dos(complete_dos)
                
                if orbital_data is None:
                    failed_ids.append(mat_id)
                    continue
                
                orbital_data["material_id"] = mat_id
                dos_data[mat_id] = orbital_data
                
            except Exception as e:
                failed_ids.append(mat_id)
                if len(failed_ids) <= 5:
                    print(f"\nError for {mat_id}: {type(e).__name__}: {str(e)[:80]}")
                continue
            
            # Save checkpoint every 50 materials
            if len(dos_data) % 50 == 0 and len(dos_data) > len(processed_ids):
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(dos_data, f)
                print(f"  [Checkpoint: {len(dos_data)} materials saved]")
    
    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dos_data, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Orbital DOS Fetch Complete!")
    print("=" * 60)
    print(f"Successfully fetched: {len(dos_data)} materials")
    print(f"Failed: {len(failed_ids)} materials")
    print(f"Output saved to: {OUTPUT_FILE}")
    
    if dos_data:
        sample_id = list(dos_data.keys())[0]
        sample = dos_data[sample_id]
        print(f"\nSample ({sample_id}):")
        print(f"  Energy points: {sample['n_points']}")
        print(f"  Energy range: {sample['energy_range']} eV")
        print(f"  Orbitals: s, p, d + total")
    
    if failed_ids:
        failed_file = os.path.join(DATA_DIR, "orbital_dos_failed.txt")
        with open(failed_file, 'w') as f:
            f.write("\n".join(failed_ids))
        print(f"\nFailed IDs saved to: {failed_file}")


if __name__ == "__main__":
    fetch_orbital_dos()
