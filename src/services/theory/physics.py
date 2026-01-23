import numpy as np
from typing import Dict, Any, Optional
from src.core.logger import get_logger, log_exception

logger = get_logger(__name__)

class PhysicsCalc:
    """
    Service for physical property calculations (DOS descriptors, etc.).
    """
    
    @log_exception(logger)
    def extract_dos_descriptors(self, dos_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Extract 11 DOS descriptors from DOS data dictionary.
        dictionary must contain 'energies' (relative to E_f) and 'total_dos'.
        """
        try:
            from scipy.integrate import trapezoid
            from scipy.signal import find_peaks
        except ImportError:
            logger.warning("Scipy not installed. Skipping physics calc.")
            return None
            
        energies = np.array(dos_data.get('energies', []))
        total_dos = np.array(dos_data.get('total_dos', []))
        
        if len(energies) == 0 or len(total_dos) == 0:
            return None
            
        try:
            # 1. DOS at Fermi level (E=0)
            ef_idx = np.argmin(np.abs(energies))
            DOS_EF = float(total_dos[ef_idx])
            
            # 2-4. d-band descriptors [-10, 5]
            d_band_mask = (energies >= -10) & (energies <= 5)
            e_dband = energies[d_band_mask]
            dos_dband = total_dos[d_band_mask]
            
            if len(dos_dband) > 0 and np.sum(dos_dband) > 0:
                norm = trapezoid(dos_dband, e_dband)
                if norm > 0:
                    d_band_center = float(trapezoid(e_dband * dos_dband, e_dband) / norm)
                    variance = trapezoid((e_dband - d_band_center)**2 * dos_dband, e_dband) / norm
                    d_band_width = float(np.sqrt(max(0, variance)))
                    # Filling (occupation up to 0)
                    occ_mask = e_dband <= 0
                    d_band_filling = float(trapezoid(dos_dband[occ_mask], e_dband[occ_mask]) / norm) if np.any(occ_mask) else 0.0
                else:
                    d_band_center = d_band_width = d_band_filling = 0.0
            else:
                d_band_center = d_band_width = d_band_filling = 0.0
            
            # Other descriptors
            epsilon_d_minus_EF = d_band_center
            
            # DOS Window [-0.3, 0.3]
            window_mask = (energies >= -0.3) & (energies <= 0.3)
            DOS_window = float(np.mean(total_dos[window_mask])) if np.any(window_mask) else DOS_EF
            
            # Unoccupied d states [0, 0.5]
            unocc_mask = (energies >= 0) & (energies <= 0.5)
            unoccupied_d_states = float(trapezoid(total_dos[unocc_mask], energies[unocc_mask])) if np.any(unocc_mask) and len(energies[unocc_mask]) > 1 else 0.0
            
            # Valence DOS Slope
            valence_DOS_slope = 0.0
            if ef_idx > 0 and ef_idx < len(total_dos) - 1:
                de = energies[ef_idx + 1] - energies[ef_idx - 1]
                if abs(de) > 1e-6:
                    valence_DOS_slope = float((total_dos[ef_idx + 1] - total_dos[ef_idx - 1]) / de)
            
            # Peaks in Valence Band [-10, 0]
            valence_mask = (energies >= -10) & (energies <= 0)
            dos_valence = total_dos[valence_mask]
            num_DOS_peaks = 0
            first_peak_position = 0.0
            if len(dos_valence) > 10:
                peaks, _ = find_peaks(dos_valence, prominence=0.1 * np.max(dos_valence))
                num_DOS_peaks = len(peaks)
                if num_DOS_peaks > 0:
                    first_peak_position = float(energies[valence_mask][peaks[0]])
            
            total_states = float(trapezoid(total_dos, energies)) if len(energies) > 1 else 0.0
            
            return {
                "d_band_center": d_band_center,
                "d_band_width": d_band_width,
                "d_band_filling": d_band_filling,
                "DOS_EF": DOS_EF,
                "DOS_window": DOS_window,
                "unoccupied_d_states": unoccupied_d_states,
                "epsilon_d_minus_EF": epsilon_d_minus_EF,
                "valence_DOS_slope": valence_DOS_slope,
                "num_DOS_peaks": num_DOS_peaks,
                "first_peak_position": first_peak_position,
                "total_states": total_states
            }
            
        except Exception as e:
            logger.warning(f"Physics Calc Error: {e}")
            return None
