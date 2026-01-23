"""
Extract DOS Descriptors from Full DOS Data
Calculates 11 DOS descriptors from the complete DOS curves and combines with formation energy.
"""

import os
import json
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "theory")
FULL_DOS_FILE = os.path.join(DATA_DIR, "full_dos_data.json")
FORMATION_FILE = os.path.join(DATA_DIR, "mp_data_summary.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "dos_descriptors_full.json")


def compute_dos_descriptors(energies, densities, efermi):
    """
    Compute 11 DOS descriptors from full DOS data.
    
    Args:
        energies: List of energy values (eV)
        densities: Dict of {spin: density_array} or total density array
        efermi: Fermi energy (eV)
        
    Returns:
        dict: 11 descriptors
    """
    energies = np.array(energies)
    
    # Sum spin channels if present
    if isinstance(densities, dict):
        total_dos = None
        for spin, dens in densities.items():
            dens_arr = np.array(dens)
            if total_dos is None:
                total_dos = dens_arr
            else:
                total_dos = total_dos + dens_arr
    else:
        total_dos = np.array(densities)
    
    # Shift energies relative to Fermi level
    e_shifted = energies - efermi
    
    # 1. DOS at Fermi level (DOS_EF)
    ef_idx = np.argmin(np.abs(e_shifted))
    DOS_EF = float(total_dos[ef_idx])
    
    # 2-4. d-band descriptors (approximated from total DOS in range -10 to 5 eV)
    d_band_mask = (e_shifted >= -10) & (e_shifted <= 5)
    e_dband = e_shifted[d_band_mask]
    dos_dband = total_dos[d_band_mask]
    
    if len(dos_dband) > 0 and np.sum(dos_dband) > 0:
        # d-band center (first moment)
        norm = trapezoid(dos_dband, e_dband)
        if norm > 0:
            d_band_center = trapezoid(e_dband * dos_dband, e_dband) / norm
        else:
            d_band_center = 0.0
        
        # d-band width (second moment, standard deviation)
        if norm > 0:
            variance = trapezoid((e_dband - d_band_center)**2 * dos_dband, e_dband) / norm
            d_band_width = np.sqrt(max(0, variance))
        else:
            d_band_width = 0.0
        
        # d-band filling (fraction below Fermi level)
        occ_mask = e_dband <= 0
        if np.sum(dos_dband) > 0:
            d_band_filling = trapezoid(dos_dband[occ_mask], e_dband[occ_mask]) / norm if np.any(occ_mask) else 0.0
        else:
            d_band_filling = 0.0
    else:
        d_band_center = 0.0
        d_band_width = 0.0
        d_band_filling = 0.0
    
    # 5. epsilon_d minus EF (same as d_band_center for EF-shifted values)
    epsilon_d_minus_EF = d_band_center
    
    # 6. DOS window around Fermi level (-0.3 to 0.3 eV)
    window_mask = (e_shifted >= -0.3) & (e_shifted <= 0.3)
    if np.any(window_mask):
        DOS_window = float(np.mean(total_dos[window_mask]))
    else:
        DOS_window = DOS_EF
    
    # 7. Unoccupied d-states (0 to 0.5 eV above Fermi)
    unocc_mask = (e_shifted >= 0) & (e_shifted <= 0.5)
    if np.any(unocc_mask):
        unoccupied_d_states = float(trapezoid(total_dos[unocc_mask], e_shifted[unocc_mask]))
    else:
        unoccupied_d_states = 0.0
    
    # 8. Valence DOS slope (derivative at Fermi level)
    if ef_idx > 0 and ef_idx < len(total_dos) - 1:
        de = e_shifted[ef_idx + 1] - e_shifted[ef_idx - 1]
        if abs(de) > 1e-6:
            valence_DOS_slope = float((total_dos[ef_idx + 1] - total_dos[ef_idx - 1]) / de)
        else:
            valence_DOS_slope = 0.0
    else:
        valence_DOS_slope = 0.0
    
    # 9. Number of DOS peaks (in valence band)
    valence_mask = (e_shifted >= -10) & (e_shifted <= 0)
    dos_valence = total_dos[valence_mask]
    if len(dos_valence) > 10:
        # Find peaks with minimum prominence
        peaks, properties = find_peaks(dos_valence, prominence=0.1 * np.max(dos_valence))
        num_DOS_peaks = len(peaks)
    else:
        num_DOS_peaks = 0
    
    # 10. First peak position (relative to Fermi)
    if len(dos_valence) > 10 and num_DOS_peaks > 0:
        e_valence = e_shifted[valence_mask]
        first_peak_position = float(e_valence[peaks[0]])
    else:
        first_peak_position = 0.0
    
    # 11. Total DOS integral (total states)
    total_states = float(trapezoid(total_dos, e_shifted))
    
    return {
        "d_band_center": d_band_center,
        "d_band_width": d_band_width,
        "d_band_filling": d_band_filling,
        "DOS_EF": DOS_EF,
        "DOS_window_-0.3_0.3": DOS_window,
        "unoccupied_d_states_0_0.5": unoccupied_d_states,
        "epsilon_d_minus_EF": epsilon_d_minus_EF,
        "valence_DOS_slope": valence_DOS_slope,
        "num_DOS_peaks": num_DOS_peaks,
        "first_peak_position": first_peak_position,
        "total_states": total_states
    }


def main():
    # Load full DOS data
    print("Loading full DOS data...")
    with open(FULL_DOS_FILE, 'r') as f:
        full_dos_data = json.load(f)
    print(f"Loaded {len(full_dos_data)} materials")
    
    # Load formation energy data
    print("Loading formation energy data...")
    with open(FORMATION_FILE, 'r') as f:
        formation_data = json.load(f)
    
    # Create formation energy map
    formation_map = {}
    for entry in formation_data:
        mat_id = str(entry['material_id'])
        formation_map[mat_id] = {
            "formula": entry.get("formula", ""),
            "formation_energy": entry.get("formation_energy", None)
        }
    
    # Extract descriptors
    print("Extracting DOS descriptors...")
    results = []
    
    for mat_id, dos_entry in tqdm(full_dos_data.items(), desc="Processing"):
        try:
            energies = dos_entry["energies"]
            densities = dos_entry["densities"]
            efermi = dos_entry["efermi"]
            
            descriptors = compute_dos_descriptors(energies, densities, efermi)
            
            # Add formation energy and metadata
            entry = {
                "material_id": mat_id,
                "formula": formation_map.get(mat_id, {}).get("formula", ""),
                "formation_energy": formation_map.get(mat_id, {}).get("formation_energy"),
                "efermi": efermi,
                "num_energy_points": len(energies),
                "energy_range": [min(energies), max(energies)],
                **descriptors
            }
            
            results.append(entry)
            
        except Exception as e:
            print(f"Error processing {mat_id}: {e}")
            continue
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOS Descriptor Extraction Complete!")
    print("=" * 60)
    print(f"Materials processed: {len(results)}")
    print(f"Output saved to: {OUTPUT_FILE}")
    
    # Statistics
    print("\nDescriptor Statistics:")
    for key in ["d_band_center", "d_band_width", "DOS_EF", "formation_energy"]:
        values = [r[key] for r in results if r.get(key) is not None]
        if values:
            print(f"  {key}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, range=[{np.min(values):.3f}, {np.max(values):.3f}]")


if __name__ == "__main__":
    main()
