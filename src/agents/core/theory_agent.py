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
        local_mats = self.db.list_materials(limit=5000)
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
        
        # DB Storage for DOS is complex (JSON blob or separate table).
        # v3.3: Current schema has 'dos_data' TEXT in materials table.
        for mat_id, dos_data in results.items():
            # We assume the material exists, so we update it.
            # However, sqlite INSERT OR REPLACE works if we have the PK.
            # Our save_material uses INSERT OR REPLACE on material_id (UNIQUE).
            # But wait, save_material arguments are specific.
            # We might need a raw update or extend save_material.
            # For now, let's keep JSON for DOS as it's large, but update DB connection if needed.
            pass
            
        output_path = os.path.join(self.config.output_dir, "orbital_pdos.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return len(results)

    def list_stored_materials(self, limit: int = 100) -> List[Dict]:
        """List materials stored in the database."""
        return self.db.list_materials(limit)

    def get_material_details(self, material_id: str) -> Optional[Dict]:
        """Get details for a specific material including evidence."""
        return self.db.get_material_with_evidence(material_id, include_cif=True)

    def get_material_details_simple(self, material_id: str) -> Optional[Dict]:
        """Get details for a specific material without CIF (batch-friendly)."""
        return self.db.get_material_with_evidence(material_id, include_cif=False)

    # ========== External DBs ==========
    
    def query_oqmd(self, elements: List[str] = None, limit: int = 100) -> List[Dict]:
        """Query OQMD database."""
        return self.ext_db.query_oqmd(elements or self.config.elements, limit)
        
    def query_aflow(self, elements: List[str] = None, limit: int = 100) -> List[Dict]:
        """Query AFLOW database."""
        return self.ext_db.query_aflow(elements or self.config.elements, limit)
        
    def query_catalysis_hub(self, reaction: str = "HER", limit: int = 50) -> List[Dict]:
        """Query Catalysis-Hub."""
        return self.ext_db.query_catalysis_hub(reaction, limit)

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
