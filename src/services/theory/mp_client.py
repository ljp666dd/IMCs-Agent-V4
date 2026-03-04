from typing import List, Dict, Optional, Any, Union
import os
import json
import time
import numpy as np
from src.core.logger import get_logger, log_exception
from src.services.common.api_cache import with_cache

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
    @with_cache(namespace="mp_search", limiter_key="materials_project")
    def search_materials(self, elements: List[str], fields: List[str] = None, limit: int = None, is_stable: bool = True) -> List[Any]:
        """Search materials by elements."""
        if not HAS_MP_API:
            return []
        if not self.api_key:
            logger.warning("MP API key missing. Skipping MP search.")
            return []
            
        if fields is None:
            fields = ["material_id", "structure", "formula_pretty", "formation_energy_per_atom"]
            
        logger.info(f"Searching MP for {len(elements)} elements (stable={is_stable})...")
        all_docs = []
        seen_ids = set()
        
        try:
            with MPRester(self.api_key) as mpr:
                # MP API 'elements' means AND (contains ALL).
                # To support broad search (OR), we iterate.
                # However, to avoid 30+ requests, we focus on the first 5 or just strictly follow the list if small.
                
                # Strategy: If list is huge (>3), assume specific single-element searches are desired.
                # If list is small (e.g. Pt, Ru), maybe user wants Pt-Ru alloy?
                # User Prompt: "Find HER catalysts" -> Planner gave 30 elements.
                # We should limit this to top candidates to avoid timeout/spam.
                
                max_elements = int(os.getenv("MP_MAX_ELEMENTS", "5") or "5")
                search_all = str(os.getenv("MP_SEARCH_ALL_ELEMENTS", "0")).lower() in ("1", "true", "yes")
                if search_all or len(elements) <= 3:
                    search_targets = elements
                else:
                    search_targets = elements[:max_elements]
                if len(elements) > len(search_targets):
                    logger.warning(
                        f"Truncating element list from {len(elements)} to {len(search_targets)} "
                        f"(MP_MAX_ELEMENTS={max_elements})."
                    )
                
                failure_count = 0
                max_failures = int(os.getenv("MP_MAX_FAILURES", "3") or "3")
                max_retries = int(os.getenv("MP_MAX_RETRIES", "2") or "2")
                sleep_s = float(os.getenv("MP_QUERY_SLEEP", "0.2") or "0.2")
                backoff_base = float(os.getenv("MP_BACKOFF_BASE", "1.5") or "1.5")

                for el in search_targets:
                    success = False
                    for attempt in range(max_retries + 1):
                        try:
                            docs = mpr.materials.summary.search(
                                elements=[el],  # Must contain this element
                                is_stable=is_stable,
                                fields=fields,
                                chunk_size=1
                            )
                            for d in docs:
                                if str(d.material_id) not in seen_ids:
                                    all_docs.append(d)
                                    seen_ids.add(str(d.material_id))
                            success = True
                            break
                        except Exception as loop_e:
                            logger.warning(f"Failed search for {el} (attempt {attempt + 1}): {loop_e}")
                            if attempt < max_retries:
                                time.sleep(backoff_base ** attempt)
                    if not success:
                        failure_count += 1
                        if failure_count >= max_failures:
                            logger.warning("Aborting MP search after repeated failures.")
                            break
                    if limit and len(all_docs) >= limit:
                        break
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                
            if limit and len(all_docs) > limit:
                all_docs = all_docs[:limit]
            
            logger.info(f"Found {len(all_docs)} unique materials.")
            return all_docs

        except Exception as e:
            logger.error(f"MP selection failed: {e}")
            return []

    @log_exception(logger)
    def download_cifs(self, docs: List[Any], output_dir: str) -> int:
        """Save CIF files from search results."""
        if not docs:
            return 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        count = 0
        for doc in docs:
            try:
                struct = doc.structure
                if struct:
                    path = os.path.join(output_dir, f"{doc.material_id}.cif")
                    if os.path.exists(path):
                        # Skip if already exists to save IO and preserve manually modified files
                        count += 1
                        continue
                        
                    struct.to(filename=path)
                    count += 1
            except Exception as e:
                # logger.debug(f"Failed to save CIF for {doc.material_id}: {e}")
                continue
        return count

    @log_exception(logger)
    @with_cache(namespace="mp_dos", limiter_key="materials_project")
    def get_orbital_dos(self, material_ids: List[str], energy_range: tuple = (-15, 10), n_points: int = 2000) -> Dict[str, Any]:
        """
        Download and process Orbital DOS.
        Note: Uses heuristic approximation from original TheoryAgent.
        """
        if not HAS_MP_API:
            return {}
        if not material_ids:
            return {}
            
        results = {}
        try:
            with MPRester(self.api_key) as mpr:
                for mat_id in material_ids:
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
