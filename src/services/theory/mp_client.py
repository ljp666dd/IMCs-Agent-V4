from typing import List, Dict, Optional, Any, Union
import os
import json
import numpy as np
from tqdm import tqdm
from src.core.logger import get_logger, log_exception

logger = get_logger(__name__)

try:
    from mp_api.client import MPRester
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False

class MPClient:
    """
    Service for interacting with Materials Project API.
    Wrapper for MPRester.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not HAS_MP_API:
            logger.warning("mp_api not installed. MPClient disabled.")

    @log_exception(logger)
    def search_materials(self, elements: List[str], fields: List[str] = None, limit: int = None, is_stable: bool = True):
        """Search materials by elements."""
        if not HAS_MP_API:
            return []
            
        if fields is None:
            fields = ["material_id", "structure", "formula_pretty", "formation_energy_per_atom"]
            
        logger.info(f"Searching MP for {elements} (stable={is_stable})...")
        try:
            with MPRester(self.api_key) as mpr:
                docs = mpr.materials.summary.search(
                    elements=elements,
                    is_stable=is_stable,
                    fields=fields
                )
                if limit and len(docs) > limit:
                    docs = docs[:limit]
                return docs
        except Exception as e:
            logger.error(f"MP selection failed: {e}")
            return []

    @log_exception(logger)
    def download_cifs(self, docs: List[Any], output_dir: str) -> int:
        """Save CIF files from search results."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        count = 0
        for doc in tqdm(docs, desc="Saving CIFs"):
            try:
                struct = doc.structure
                if struct:
                    path = os.path.join(output_dir, f"{doc.material_id}.cif")
                    struct.to(filename=path)
                    count += 1
            except Exception as e:
                # logger.debug(f"Failed to save CIF for {doc.material_id}: {e}")
                continue
        return count

    @log_exception(logger)
    def get_orbital_dos(self, material_ids: List[str], energy_range: tuple = (-15, 10), n_points: int = 2000) -> Dict[str, Any]:
        """
        Download and process Orbital DOS.
        Note: Uses heuristic approximation from original TheoryAgent.
        """
        if not HAS_MP_API:
            return {}
            
        results = {}
        try:
            with MPRester(self.api_key) as mpr:
                for mat_id in tqdm(material_ids, desc="Fetching DOS"):
                    try:
                        dos = mpr.get_dos_by_material_id(mat_id)
                        if dos is None: continue
                        
                        efermi = dos.efermi
                        energies = dos.energies - efermi
                        mask = (energies >= energy_range[0]) & (energies <= energy_range[1])
                        
                        # Target energy grid
                        e_new = np.linspace(energy_range[0], energy_range[1], n_points)
                        
                        # Heuristic Orbital Extraction (Ported from v2.0)
                        s_dos = np.zeros(len(energies))
                        p_dos = np.zeros(len(energies))
                        d_dos = np.zeros(len(energies))
                        
                        pdos = dos.get_element_dos()
                        for element, element_dos in pdos.items():
                            for spin, densities in element_dos.densities.items():
                                # Heuristic weights
                                s_dos += densities * 0.1
                                p_dos += densities * 0.3
                                d_dos += densities * 0.6
                                
                        # Interpolate
                        s_int = np.interp(e_new, energies, s_dos)
                        p_int = np.interp(e_new, energies, p_dos)
                        d_int = np.interp(e_new, energies, d_dos)
                        total_int = s_int + p_int + d_int
                        
                        results[str(mat_id)] = {
                            "material_id": str(mat_id),
                            "energies": e_new.tolist(),
                            "s_dos": s_int.tolist(),
                            "p_dos": p_int.tolist(),
                            "d_dos": d_int.tolist(),
                            "total_dos": total_int.tolist()
                        }
                    except Exception as e:
                        logger.warning(f"Failed to get DOS for {mat_id}: {e}")
                        continue
        except Exception as e:
            logger.error(f"DOS batch failed: {e}")
            
        return results
