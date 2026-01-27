"""
Theory Data Agent (TheoryDataAgent)
Refactored (v3.3) to use Service-Oriented Architecture and SQLite Database.
"""

import os
import sys
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import warnings

from src.core.logger import get_logger, log_exception
from src.services.theory.mp_client import MPClient
from src.services.theory.external_db import ExternalDBClient
from src.services.theory.physics import PhysicsCalc
from src.services.db.database import DatabaseService
from src.config.config import config as app_config

logger = get_logger(__name__)

@dataclass
class TheoryDataConfig:
    """
    Configuration for Theory Data Agent.
    
    Attributes:
        api_key (str): Materials Project API Key.
        output_dir (str): Directory to save raw files (legacy/backup).
        elements (List[str]): List of elements to research.
    """
    api_key: str = app_config.MP_API_KEY
    output_dir: str = "data/theory"
    # 默认金属元素集合(限定搜索空间, 供任务图/理论检索/训练筛选统一使用)
    elements: List[str] = field(default_factory=lambda: [
        "Pt", "Pd", "Ni", "Co", "Fe", "Cu", "Au", "Ag", "Ir", "Rh", "Ru", "Os",
        "Mo", "W", "V", "Nb", "Ta", "Ti", "Zr", "Hf", "Mn", "Re", "Cr",
        "Zn", "Cd", "In", "Sn", "Ga", "Ge", "Al", "Sc", "Y", "La", "Ce"
    ])
    energy_above_hull: float = 0.0


class TheoryDataAgent:
    """
    Theory Data Agent for downloading and processing computational data.
    
    Architecture:
        - Facade Pattern: Coordinates services.
        - Services: MPClient (API), ExternalDB (Queries), PhysicsCalc (Science), DatabaseService (Storage).
    """
    
    def __init__(self, config: TheoryDataConfig = None):
        """
        Initialize the Theory Data Agent.
        
        Args:
            config (TheoryDataConfig): Configuration object.
        """
        self.config = config or TheoryDataConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize Services
        self.mp = MPClient(api_key=self.config.api_key)
        self.ext_db = ExternalDBClient()
        self.physics = PhysicsCalc()
        self.db = DatabaseService() # v3.3: Database Integration
        
        logger.info("TheoryDataAgent initialized with services and database.")

    # ========== Materials Project ==========
    
    @log_exception(logger)
    def download_structures(self, material_ids: List[str] = None, limit: int = None) -> int:
        """
        Download CIF structures from Materials Project and save to DB.
        
        Args:
            material_ids (List[str], optional): List of MP IDs to download.
            limit (int, optional): Max number of materials to download.
            
        Returns:
            int: Number of structures filtered/saved.
        """
        cif_dir = os.path.join(self.config.output_dir, "cifs")
        
        # 0. Check Local DB First (Local-First Strategy)
        # Fetch all materials to filter (assuming DB is handleable size < 10k)
        local_mats = self.db.list_materials(limit=5000, allowed_elements=self.config.elements)
        local_matches = []
        
        # Simple filter: checks if formula contains ANY of the target elements
        # (This mimics the broad search intension)
        target_elements = set(self.config.elements)
        for m in local_mats:
            # Heuristic check: does formula string contain element symbol?
            # Ideally use pymatgen Composition, but string check is fast proxy for now.
            # We trust the DB contains valid formulas.
            mat_formula = m.get("formula", "")
            if any(el in mat_formula for el in target_elements):
                local_matches.append(m)
                
        # If we have substantial local data (e.g. > 50% of request limit or > 20 items), skip API
        # This avoids re-downloading Pt every time.
        if len(local_matches) >= (limit or 10):
            logger.info(f"Found {len(local_matches)} local materials. Skipping API search.")
            return len(local_matches)

        # 1. Search Materials (API)
        docs = self.mp.search_materials(
            elements=self.config.elements, 
            limit=limit,
            fields=["material_id", "structure", "formula_pretty", "formation_energy_per_atom"]
        )
        
        # 2. Save raw CIFs (Backup)
        count = self.mp.download_cifs(docs, cif_dir)
        
        # 3. Save to Database (v3.3)
        for doc in docs:
            cif_path = os.path.join(cif_dir, f"{doc.material_id}.cif")
            self.db.save_material(
                material_id=str(doc.material_id),
                formula=str(doc.formula_pretty),
                energy=float(doc.formation_energy_per_atom) if doc.formation_energy_per_atom else None,
                cif_path=cif_path
            )
            
        return len(docs)

    @log_exception(logger)
    def download_formation_energy(self, material_ids: List[str] = None) -> int:
        """
        Download formation energy data and save to DB.
        
        Args:
            material_ids (List[str]): Optional list of IDs.
            
        Returns:
            int: Number of records saved.
        """
        docs = self.mp.search_materials(
            elements=self.config.elements,
            fields=["material_id", "formula_pretty", "formation_energy_per_atom"]
        )
        
        count = 0
        for doc in docs:
            if doc.formation_energy_per_atom is not None:
                # Update DB
                self.db.save_material(
                    material_id=str(doc.material_id),
                    formula=str(doc.formula_pretty),
                    energy=float(doc.formation_energy_per_atom)
                )
                count += 1
                
        # Also save legacy JSON for backward compatibility
        # ... logic skipped to encourage DB usage ...
        return count

    @log_exception(logger)
    def download_orbital_dos(self, material_ids: List[str] = None, energy_range: tuple = (-15, 10)) -> int:
        """
        Download Orbital DOS data.
        
        Args:
            material_ids (List[str]): MP IDs.
            energy_range (tuple): Energy range relative to Fermi level.
            
        Returns:
            int: Number of DOS records processed.
        """
        if not material_ids:
             docs = self.mp.search_materials(self.config.elements, fields=["material_id"], limit=20)
             material_ids = [str(d.material_id) for d in docs]
             
        results = self.mp.get_orbital_dos(material_ids, energy_range)
        
        # DB Storage for DOS: store compact descriptors to avoid huge blobs.
        updated = 0
        for mat_id, dos_data in results.items():
            descriptors = self.extract_dos_descriptors(dos_data)
            if descriptors:
                self.db.update_material_dos(str(mat_id), descriptors)
                updated += 1
            
        output_path = os.path.join(self.config.output_dir, "orbital_pdos.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return updated

    def _normalize_surface_formula(self, surface: str) -> Optional[str]:
        """Normalize surface composition string to a formula guess."""
        if not surface:
            return None
        import re
        token = re.sub(r'[^A-Za-z0-9]', '', surface)
        if not token:
            return None
        try:
            from pymatgen.core import Composition
            comp = Composition(token)
            return comp.reduced_formula
        except Exception:
            return token

    def _match_material_by_elements(self, formula_guess: str) -> Optional[Dict]:
        """Weak matching: find material with the same element set."""
        if not formula_guess:
            return None
        try:
            from pymatgen.core import Composition
            target_elements = set([el.symbol for el in Composition(formula_guess).elements])
        except Exception:
            return None
        # Limit scan for speed
        for m in self.db.list_materials(limit=500, allowed_elements=self.config.elements):
            try:
                els = set([el.symbol for el in Composition(m.get("formula", "")).elements])
                if els == target_elements:
                    return m
            except Exception:
                continue
        return None

    def download_adsorption_energies(self, adsorbates: List[str] = None, limit: int = 50) -> int:
        """
        Download adsorption energies (H*/OH*) from Catalysis-Hub and link to materials.
        """
        adsorbates = adsorbates or ["H*", "OH*"]
        total_saved = 0
        for ads in adsorbates:
            records = self.ext_db.query_catalysis_hub(adsorbate=ads, limit=limit)
            for rec in records:
                surface = rec.get("surface", "")
                facet = rec.get("facet", "")
                formula_guess = self._normalize_surface_formula(surface)
                material_rec = self.db.get_material_by_formula(formula_guess) if formula_guess else None
                if not material_rec and formula_guess:
                    material_rec = self._match_material_by_elements(formula_guess)
                material_id = material_rec.get("material_id") if material_rec else None

                rec_id = self.db.save_adsorption_energy(
                    material_id=material_id,
                    surface_composition=surface,
                    facet=facet,
                    adsorbate=ads,
                    reaction_energy=rec.get("reaction_energy"),
                    activation_energy=rec.get("activation_energy"),
                    source="Catalysis-Hub",
                    metadata={"equation": rec.get("equation")}
                )
                total_saved += 1

                # Evidence link if material resolved
                if material_id:
                    self.db.save_evidence(
                        material_id=material_id,
                        source_type="adsorption_energy",
                        source_id=str(rec_id),
                        score=1.1,
                        metadata={
                            "adsorbate": ads,
                            "reaction_energy": rec.get("reaction_energy"),
                            "activation_energy": rec.get("activation_energy"),
                            "surface": surface,
                            "facet": facet
                        }
                    )
        return total_saved

    def list_stored_materials(self, limit: int = 100) -> List[Dict]:
        """List materials stored in the database."""
        return self.db.list_materials(limit, allowed_elements=self.config.elements)

    def get_material_details(self, material_id: str) -> Optional[Dict]:
        """Get details for a specific material including evidence."""
        data = self.db.get_material_with_evidence(material_id, include_cif=True)
        if not data:
            return None
        if not self._formula_allowed(data.get("formula")):
            return None
        return data

    def get_material_details_simple(self, material_id: str) -> Optional[Dict]:
        """Get details for a specific material without CIF (batch-friendly)."""
        data = self.db.get_material_with_evidence(material_id, include_cif=False)
        if not data:
            return None
        if not self._formula_allowed(data.get("formula")):
            return None
        return data

    def _formula_allowed(self, formula: Optional[str]) -> bool:
        """Check formula elements subset of allowed list."""
        if not formula:
            return False
        try:
            from pymatgen.core import Composition
            elements = {el.symbol for el in Composition(formula).elements}
        except Exception:
            return False
        return elements.issubset(set(self.config.elements))

    # ========== External DBs ==========
    
    def query_oqmd(self, elements: List[str] = None, limit: int = 100) -> List[Dict]:
        """Query OQMD database."""
        return self.ext_db.query_oqmd(elements or self.config.elements, limit)
        
    def query_aflow(self, elements: List[str] = None, limit: int = 100) -> List[Dict]:
        """Query AFLOW database."""
        return self.ext_db.query_aflow(elements or self.config.elements, limit)
        
    def query_catalysis_hub(self, adsorbate: str = "H*", limit: int = 50) -> List[Dict]:
        """Query Catalysis-Hub for adsorption energies."""
        return self.ext_db.query_catalysis_hub(adsorbate=adsorbate, limit=limit)

    def download_adsorption_energies(self, save: bool = True) -> List[Dict]:
        """Download adsorption energies."""
        results = self.query_catalysis_hub(reaction="HER", limit=100)
        
        # Future: Save to DB
        return results

    # ========== Physics ==========
    
    def extract_dos_descriptors(self, dos_data: Dict) -> Dict[str, float]:
        """
        Calculate d-band center and other descriptors.
        
        Args:
            dos_data (Dict): DOS data dictionary.
            
        Returns:
            Dict: Calculated descriptors.
        """
        return self.physics.extract_dos_descriptors(dos_data)

    # ========== Utilities ==========
    
    def get_status(self) -> Dict[str, int]:
        """Get current data status."""
        cif_dir = os.path.join(self.config.output_dir, "cifs")
        return {
            "cif_files": len([f for f in os.listdir(cif_dir) if f.endswith('.cif')]) if os.path.exists(cif_dir) else 0,
            "formation_energy": 0, # Legacy count
            "orbital_dos": 0
        }
