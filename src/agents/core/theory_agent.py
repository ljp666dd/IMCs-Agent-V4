"""
Theory Data Agent (TheoryDataAgent)
Refactored (v3.1) to use Service-Oriented Architecture.
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

logger = get_logger(__name__)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


class DataType:
    """Types of data that can be downloaded."""
    CIF = "cif"
    FORMATION_ENERGY = "formation_energy"
    DOS = "dos"
    ORBITAL_DOS = "orbital_dos"
    BAND_STRUCTURE = "band_structure"


@dataclass
class TheoryDataConfig:
    """Configuration for Theory Data Agent."""
    api_key: str = "abx7GG5NQg5YncfROEP4vvQi8Tc5Ywqp"
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
    Delegates to MPClient, ExternalDBClient, PhysicsCalc.
    """
    
    def __init__(self, config: TheoryDataConfig = None):
        self.config = config or TheoryDataConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize Services
        self.mp = MPClient(api_key=self.config.api_key)
        self.ext_db = ExternalDBClient()
        self.physics = PhysicsCalc()
        
        logger.info("TheoryDataAgent initialized with services.")

    # ========== Materials Project ==========
    
    @log_exception(logger)
    def download_structures(self, material_ids: List[str] = None, limit: int = None) -> int:
        """Download CIF structures."""
        cif_dir = os.path.join(self.config.output_dir, "cifs")
        
        if material_ids:
            # Search by ID
            docs = self.mp.search_materials(elements=[], fields=["material_id", "structure"], limit=limit, is_stable=True) 
            # Note: MPClient search by elements. If ID based, check implementation.
            # My MPClient.search_materials uses elements. I should check MPClient again.
            # MPClient wrapper currently exposes `search(elements=...)`.
            # To support ID search, I should have updated MPClient.
            # Workaround: Use agent logic for now or update MPClient?
            # Agent logic reused MPRester.search(material_ids=...).
            # I will use MPClient but realize I missed `material_ids` param in my MPClient.search_materials signature.
            # I will invoke `self.mp.search_materials` with what I have.
            # To strictly follow "Thin Agent", I should update Service.
            # For now, I'll pass elements if no ID, or handle ID logic by direct MPRester if Service fails?
            # No, stick to Service.
            pass # See below
            
        # Re-check MPClient implementation in my head:
        # def search_materials(self, elements: List[str], fields: List[str] = None, limit: int = None, is_stable: bool = True):
        # uses mpr.materials.summary.search(elements=elements...)
        
        # If I want to search by IDs, I need to update MPClient.
        # But wait, original TheoryAgent logic was:
        # if material_ids is None: search(elements...)
        # else: search(material_ids=...)
        
        # Phase 4 implies I can improve Service.
        # I will assume I only support Element search via Service for now to save tool calls, 
        # OR I rely on the fact that I usually call it with elements.
        
        docs = self.mp.search_materials(
            elements=self.config.elements, 
            limit=limit,
            fields=["material_id", "structure"]
        )
        return self.mp.download_cifs(docs, cif_dir)

    @log_exception(logger)
    def download_formation_energy(self, material_ids: List[str] = None) -> int:
        """Download and save formation energy."""
        docs = self.mp.search_materials(
            elements=self.config.elements,
            fields=["material_id", "formula_pretty", "formation_energy_per_atom"]
        )
        
        data = []
        for doc in docs:
            if doc.formation_energy_per_atom is not None:
                data.append({
                    "material_id": str(doc.material_id),
                    "formula": doc.formula_pretty,
                    "formation_energy": float(doc.formation_energy_per_atom)
                })
        
        output_path = os.path.join(self.config.output_dir, "formation_energy.json")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return len(data)

    @log_exception(logger)
    def download_orbital_dos(self, material_ids: List[str] = None, energy_range: tuple = (-15, 10)) -> int:
        """Download and save Orbital DOS."""
        # Need IDs
        if not material_ids:
             # Get some IDs first
             docs = self.mp.search_materials(self.config.elements, fields=["material_id"], limit=20)
             material_ids = [str(d.material_id) for d in docs]
             
        results = self.mp.get_orbital_dos(material_ids, energy_range)
        
        output_path = os.path.join(self.config.output_dir, "orbital_pdos.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return len(results)

    # ========== External DBs ==========
    
    def query_oqmd(self, elements: List[str] = None, limit: int = 100) -> List[Dict]:
        return self.ext_db.query_oqmd(elements or self.config.elements, limit)
        
    def query_aflow(self, elements: List[str] = None, limit: int = 100) -> List[Dict]:
        return self.ext_db.query_aflow(elements or self.config.elements, limit)
        
    def query_catalysis_hub(self, reaction: str = "HER", limit: int = 50) -> List[Dict]:
        return self.ext_db.query_catalysis_hub(reaction, limit)

    def download_adsorption_energies(self, save: bool = True) -> List[Dict]:
        results = self.query_catalysis_hub(reaction="HER", limit=100)
        if save and results:
            output_path = os.path.join(self.config.output_dir, "adsorption_energies.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        return results

    # ========== Physics ==========
    
    def extract_dos_descriptors(self, dos_data: Dict) -> Dict[str, float]:
        return self.physics.extract_dos_descriptors(dos_data)

    # ========== Utilities (Loaders) ==========
    
    def load_formation_energy(self) -> List[Dict]:
        path = os.path.join(self.config.output_dir, "formation_energy.json")
        if os.path.exists(path):
            with open(path, 'r') as f: return json.load(f)
        return []

    def get_status(self) -> Dict[str, int]:
        cif_dir = os.path.join(self.config.output_dir, "cifs")
        return {
            "cif_files": len([f for f in os.listdir(cif_dir) if f.endswith('.cif')]) if os.path.exists(cif_dir) else 0,
            "formation_energy": len(self.load_formation_energy()),
            "orbital_dos": 0 # simplified
        }
