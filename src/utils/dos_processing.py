import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

def get_dos_fingerprint(dos, n_bins=400, e_min=-5.0, e_max=5.0):
    """
    Constructs a high-resolution DOS fingerprint (defaults to 400 bins).
    """
    energies = dos.energies - dos.efermi
    densities = dos.get_densities() 
    
    # Linear interpolation onto the fixed grid
    x_new = np.linspace(e_min, e_max, n_bins)
    y_new = np.interp(x_new, energies, densities, left=0, right=0)
    
    return y_new

def calculate_electronic_descriptors(dos, element_symbol=None):
    """
    Calculates 11 specific electronic descriptors from the DOS.
    
    Args:
        dos (pymatgen.electronic_structure.dos.CompleteDos): DOS object.
        element_symbol (str): Specific element to project d-band (e.g., 'Pt'). 
                              If None, uses Total DOS (which might be less physical for d-band metrics).
    
    Returns:
        dict: Dictionary containing the 11 descriptors.
    """
    # 1. Prepare Data
    energies = dos.energies - dos.efermi
    
    # Get d-orbital projection if element is specified, else approximation from total
    # ideally we want the d-band of the active site
    if element_symbol and element_symbol in dos.get_element_dos():
        # Sum d-orbitals for this element
        # properties: {Orbital.s: ..., Orbital.p: ..., Orbital.d: ...}
        el_dos = dos.get_element_dos()[element_symbol]
        # This returns a Dos object, need to sum specific orbitals manually or rely on property
        # Pymatgen Dos objects don't separate orbitals easily in `get_element_dos` return unless using projections
        # Let's try to get specific orbital densities directly if possible
        # Actually `get_element_dos` returns a Dos object summing all orbitals.
        # Check if we can get partial dos (PDOS)
        try:
            pdos = dos.get_element_spd_dos(element_symbol) # {Orbital.d: Dos object, ...}
            from pymatgen.electronic_structure.core import OrbitalType
            d_dos_obj = pdos.get(OrbitalType.d)
            s_dos_obj = pdos.get(OrbitalType.s)
            p_dos_obj = pdos.get(OrbitalType.p)
            
            y_d = d_dos_obj.get_densities() if d_dos_obj else np.zeros_like(energies)
            y_sp = (s_dos_obj.get_densities() if s_dos_obj else 0) + (p_dos_obj.get_densities() if p_dos_obj else 0)
        except:
            # Fallback if spd not available
            y_d = dos.get_densities()
            y_sp = np.zeros_like(y_d)
    else:
        # Fallback to total DOS if no element specified
        y_d = dos.get_densities()
        y_sp = np.zeros_like(y_d)

    # Helper for integration
    def integrate(y, x, x_min, x_max):
        mask = (x >= x_min) & (x <= x_max)
        if not np.any(mask): return 0.0
        return trapezoid(y[mask], x[mask])

    # --- Calculations ---
    
    # 1. d_band_center
    # Range: usually valence band (-10 to 0) or full range. Let's use -10 to 2
    mask_d = (energies >= -10) & (energies <= 2)
    start, end = -10, 2
    
    numerator = integrate(y_d * energies, energies, start, end)
    denominator = integrate(y_d, energies, start, end) + 1e-9
    d_band_center = numerator / denominator

    # 2. d_band_width (Second moment)
    numerator_width = integrate(y_d * (energies - d_band_center)**2, energies, start, end)
    d_band_width = np.sqrt(numerator_width / denominator)

    # 3. d_band_filling
    # Integration up to Fermi level (0)
    d_band_filling = integrate(y_d, energies, -20, 0)

    # 4. DOS_EF
    # Linear interp at 0
    dos_ef = np.interp(0, energies, y_d)

    # 5. DOS_window_-0.3_0.3
    dos_window = integrate(y_d, energies, -0.3, 0.3)

    # 6. unoccupied_d_states_0_0.5
    unocc_d = integrate(y_d, energies, 0, 0.5)

    # 7. epsilon_d_minus_EF
    # Simply d_band_center since EF is 0 in our scale
    epsilon_d_minus_ef = d_band_center 

    # 8. sp_d_hybridization
    # Definition: Overlap area in valence band? Or ratio?
    # Using Ratio of Integrated DOS in valence band (-5 to 0) as generic metric
    res_sp = integrate(y_sp, energies, -5, 0)
    res_d = integrate(y_d, energies, -5, 0) + 1e-9
    sp_d_hybridization = res_sp / res_d

    # 9. orbital_ratio_d
    # fraction of d states in total DOS (valence)
    total_dos = dos.get_densities()
    total_int = integrate(total_dos, energies, -10, 0) + 1e-9
    d_int = integrate(y_d, energies, -10, 0)
    orbital_ratio_d = d_int / total_int

    # 10. valence_DOS_slope
    # Slope of linear fit in range [-1, 0]
    mask_slope = (energies >= -1.0) & (energies <= 0)
    if np.any(mask_slope):
        coeffs = np.polyfit(energies[mask_slope], y_d[mask_slope], 1)
        valence_dos_slope = coeffs[0]
    else:
        valence_dos_slope = 0

    # 11. Peak Analysis (num_DOS_peaks, first_peak_position)
    # Smooth a bit before peak finding to avoid noise
    mask_peaks = (energies >= -5) & (energies <= 0) # Analyze valence band
    y_peaks = y_d[mask_peaks]
    x_peaks = energies[mask_peaks]
    
    if len(y_peaks) > 0:
        peaks, _ = find_peaks(y_peaks, prominence=0.1) # prominence filter
        num_dos_peaks = len(peaks)
        if num_dos_peaks > 0:
            # First peak below EF (closest to EF from left)
            # peaks are indices in y_peaks, which is -5 to 0. 
            # The "first peak" usually means "highest energy peak" (closest to 0) or "lowest"?
            # Typically "First peak below fermi level" implies the one closest to 0.
            last_peak_idx = peaks[-1] 
            first_peak_position = x_peaks[last_peak_idx]
        else:
            first_peak_position = -10.0 # Dummy
    else:
        num_dos_peaks = 0
        first_peak_position = -10.0

    return {
        "d_band_center": d_band_center,
        "d_band_width": d_band_width,
        "d_band_filling": d_band_filling,
        "DOS_EF": dos_ef,
        "DOS_window_-0.3_0.3": dos_window,
        "unoccupied_d_states_0_0.5": unocc_d,
        "epsilon_d_minus_EF": epsilon_d_minus_ef,
        "sp_d_hybridization": sp_d_hybridization,
        "orbital_ratio_d": orbital_ratio_d,
        "valence_DOS_slope": valence_dos_slope,
        "num_DOS_peaks": num_dos_peaks,
        "first_peak_position": first_peak_position
    }
