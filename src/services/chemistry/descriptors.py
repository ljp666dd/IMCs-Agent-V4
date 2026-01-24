import numpy as np
from pymatgen.core import Structure
from typing import List, Dict, Optional, Any
from src.core.logger import get_logger, log_exception

logger = get_logger(__name__)

class StructureFeaturizer:
    """
    Extracts physical/chemical/structural features from CIF structures.
    Service layer implementation - Decoupled from Agents.
    """
    
    # Whitelist of 42 physical/chemical/structural features
    KNOWN_FEATURES = [
         "n_atoms", "n_elements", "volume_per_atom", "density", "packing_fraction",
         "avg_Z", "std_Z", "max_Z", "min_Z", "range_Z",
         "avg_mass", "std_mass", "max_mass", "min_mass",
         "avg_electronegativity", "std_electronegativity", "max_electronegativity", "min_electronegativity", "range_electronegativity",
         "avg_radius", "std_radius", "max_radius", "min_radius", "radius_ratio",
         "composition_entropy", "composition_variance", "max_composition", "min_composition", "n_elements_comp",
         "lattice_a", "lattice_b", "lattice_c", "alpha", "beta", "gamma", "c_over_a",
         "avg_lattice", "lattice_distortion",
         "mixing_enthalpy_proxy", "avg_valence_electrons", "std_valence_electrons", "volume"
    ]

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.KNOWN_FEATURES

    @log_exception(logger)
    def extract(self, cif_path: str) -> Optional[np.ndarray]:
        """Extract features from a single CIF file."""
        try:
            structure = Structure.from_file(cif_path)
            return self._structure_to_features(structure)
        except Exception as e:
            logger.warning(f"Failed to extract features from {cif_path}: {e}")
            return None

    def _structure_to_features(self, structure: Structure) -> np.ndarray:
        """
        Convert structure to extended feature vector (42 dim).
        Private method containing the core chemical logic.
        """
        n_atoms = len(structure)
        elements = [site.specie for site in structure]
        unique_elements = list(set(str(e) for e in elements))
        n_elements = len(unique_elements)
        
        volume = structure.volume
        volume_per_atom = volume / n_atoms
        
        # Atomic numbers
        atomic_numbers = [e.Z for e in elements]
        avg_Z = np.mean(atomic_numbers)
        std_Z = np.std(atomic_numbers) if len(atomic_numbers) > 1 else 0
        max_Z = max(atomic_numbers)
        min_Z = min(atomic_numbers)
        range_Z = max_Z - min_Z
        
        # Atomic masses
        atomic_masses = [float(e.atomic_mass) for e in elements]
        avg_mass = np.mean(atomic_masses)
        std_mass = np.std(atomic_masses) if len(atomic_masses) > 1 else 0
        max_mass = max(atomic_masses)
        min_mass = min(atomic_masses)
        
        total_mass = sum(atomic_masses)
        density = total_mass / volume
        
        # Electronegativity (Pauling scale)
        electronegativities = []
        for e in elements:
            try:
                if hasattr(e, 'X') and e.X is not None:
                    electronegativities.append(e.X)
                else:
                    electronegativities.append(2.0)  # Default
            except:
                electronegativities.append(2.0)
        
        avg_en = np.mean(electronegativities)
        std_en = np.std(electronegativities) if len(electronegativities) > 1 else 0
        max_en = max(electronegativities)
        min_en = min(electronegativities)
        range_en = max_en - min_en
        
        # Ionic radius estimation (metallic radii in Angstrom)
        ionic_radii = {
            'Pt': 1.39, 'Pd': 1.37, 'Ni': 1.24, 'Co': 1.25, 'Fe': 1.26,
            'Cu': 1.28, 'Au': 1.44, 'Ag': 1.44, 'Ir': 1.36, 'Rh': 1.34,
            'Ru': 1.34, 'Os': 1.35, 'Mo': 1.39, 'W': 1.39, 'V': 1.34,
            'Nb': 1.46, 'Ta': 1.46, 'Ti': 1.47, 'Zr': 1.60, 'Hf': 1.59,
            'Mn': 1.27, 'Re': 1.37, 'Cr': 1.28, 'Zn': 1.34, 'Cd': 1.52,
            'In': 1.67, 'Sn': 1.58, 'Ga': 1.41, 'Ge': 1.37, 'Al': 1.43,
            'Sc': 1.64, 'Y': 1.80, 'La': 1.87, 'Ce': 1.82
        }
        
        radii = []
        for e in elements:
            el_str = str(e.element) if hasattr(e, 'element') else str(e)
            radii.append(ionic_radii.get(el_str, 1.40))  # Default 1.40 A
        
        avg_radius = np.mean(radii)
        std_radius = np.std(radii) if len(radii) > 1 else 0
        max_radius = max(radii)
        min_radius = min(radii)
        radius_ratio = max_radius / min_radius if min_radius > 0 else 1.0
        
        # Packing fraction
        sphere_volume = sum((4/3) * np.pi * (r ** 3) for r in radii)
        packing_fraction = sphere_volume / volume if volume > 0 else 0.5
        
        # Composition features
        element_counts = {}
        for e in elements:
            el_str = str(e.element) if hasattr(e, 'element') else str(e)
            element_counts[el_str] = element_counts.get(el_str, 0) + 1
        
        compositions = [c / n_atoms for c in element_counts.values()]
        composition_entropy = -sum(c * np.log(c + 1e-10) for c in compositions)
        composition_variance = np.var(compositions) if len(compositions) > 1 else 0
        max_composition = max(compositions)
        min_composition = min(compositions)
        
        # Lattice features
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        c_over_a = c / a if a > 0 else 1.0
        avg_lattice = (a + b + c) / 3
        lattice_distortion = np.std([a, b, c]) / avg_lattice if avg_lattice > 0 else 0
        
        # Mixing enthalpy estimation (simplified Miedema model)
        mixing_enthalpy_proxy = 0.0
        if n_elements > 1:
            for i, (el1, c1) in enumerate(zip(unique_elements, compositions)):
                for j, (el2, c2) in enumerate(zip(unique_elements, compositions)):
                    if i < j:
                        # Get electronegativities
                        en1 = electronegativities[i] if i < len(electronegativities) else 2.0
                        en2 = electronegativities[j] if j < len(electronegativities) else 2.0
                        # Simplified: H_mix ~ -k * delta_chi^2
                        mixing_enthalpy_proxy += 4 * c1 * c2 * (en1 - en2) ** 2
        
        # Valence electron estimation
        valence_electrons = {
            'Pt': 10, 'Pd': 10, 'Ni': 10, 'Co': 9, 'Fe': 8,
            'Cu': 11, 'Au': 11, 'Ag': 11, 'Ir': 9, 'Rh': 9,
            'Ru': 8, 'Os': 8, 'Mo': 6, 'W': 6, 'V': 5,
            'Ti': 4, 'Zr': 4, 'Hf': 4, 'Mn': 7, 'Cr': 6,
            'Zn': 12, 'Al': 3, 'Ga': 3, 'Sn': 4, 'Ge': 4
        }
        
        ve_list = []
        for e in elements:
            el_str = str(e.element) if hasattr(e, 'element') else str(e)
            ve_list.append(valence_electrons.get(el_str, 6))
        
        avg_ve = np.mean(ve_list)
        std_ve = np.std(ve_list) if len(ve_list) > 1 else 0
        
        # Build feature vector (42 features)
        features = [
            # Structural (5)
            n_atoms, n_elements, volume_per_atom, density, packing_fraction,
            # Atomic numbers (5)
            avg_Z, std_Z, max_Z, min_Z, range_Z,
            # Atomic masses (4)
            avg_mass, std_mass, max_mass, min_mass,
            # Electronegativity (5)
            avg_en, std_en, max_en, min_en, range_en,
            # Ionic radius (5)
            avg_radius, std_radius, max_radius, min_radius, radius_ratio,
            # Composition (5)
            composition_entropy, composition_variance, max_composition, min_composition, n_elements,
            # Lattice (7)
            a, b, c, alpha, beta, gamma, c_over_a,
            # Lattice stats (2)
            avg_lattice, lattice_distortion,
            # Mixing/Valence (4)
            mixing_enthalpy_proxy, avg_ve, std_ve, volume
        ]
        
        return np.array(features, dtype=np.float32)
