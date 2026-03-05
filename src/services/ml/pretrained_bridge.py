"""
IMCs Pretrained GNN Bridge (V5 - Phase E)

Bridges CHGNet / MACE universal potentials with the IMCs pipeline:
1. CIF → Graph automatic conversion (using pymatgen + torch_geometric)
2. CHGNet pretrained model loading for energy/force prediction
3. Single-shot CIF→activity inference pipeline
4. Graceful fallback when pretrained models are unavailable

This module does NOT replace the existing SimpleCGCNN/SchNet/MEGNet in gnn.py,
but provides an additional "pretrained foundation model" pathway.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from src.core.logger import get_logger

logger = get_logger(__name__)

# --- Availability checks ---
HAS_CHGNET = False
HAS_MACE = False
HAS_PYMATGEN = False

try:
    from pymatgen.core.structure import Structure
    HAS_PYMATGEN = True
except ImportError:
    pass

try:
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import MolecularDynamics
    HAS_CHGNET = True
except ImportError:
    pass

try:
    import mace
    HAS_MACE = True
except ImportError:
    pass


@dataclass
class PretrainedPrediction:
    """Prediction result from a pretrained model."""
    material_id: str
    formula: str = ""
    energy_per_atom_ev: float = 0.0
    forces_norm: float = 0.0
    stress_gpa: float = 0.0
    stability_score: float = 0.0  # 0-1, derived from energy
    model_used: str = ""
    success: bool = False
    error: str = ""


class PretrainedGNNBridge:
    """
    Bridge to pretrained universal potentials (CHGNet, MACE).
    """

    def __init__(self, preferred_model: str = "chgnet"):
        """
        Args:
            preferred_model: "chgnet" or "mace"
        """
        self.model = None
        self.model_name = "none"
        self.available = False

        if preferred_model == "chgnet" and HAS_CHGNET:
            try:
                self.model = CHGNet.load()
                self.model_name = "CHGNet"
                self.available = True
                logger.info("PretrainedGNNBridge: CHGNet loaded successfully.")
            except Exception as e:
                logger.warning(f"CHGNet load failed: {e}")

        if not self.available and HAS_MACE:
            try:
                # MACE loading placeholder — actual API may vary
                logger.info("PretrainedGNNBridge: MACE model available but not loaded (placeholder).")
                self.model_name = "MACE"
            except Exception as e:
                logger.warning(f"MACE load failed: {e}")

        if not self.available:
            logger.warning(
                "PretrainedGNNBridge: No pretrained model available. "
                "Install chgnet (`pip install chgnet`) for full functionality."
            )

    def predict_from_cif(self, cif_path: str, material_id: str = "") -> PretrainedPrediction:
        """
        Run a single-shot prediction from a CIF file.

        Args:
            cif_path: Path to .cif file
            material_id: Optional material identifier

        Returns:
            PretrainedPrediction with energy, forces, stability
        """
        if not HAS_PYMATGEN:
            return PretrainedPrediction(
                material_id=material_id, success=False,
                error="pymatgen not installed"
            )

        try:
            structure = Structure.from_file(cif_path)
            formula = structure.composition.reduced_formula
        except Exception as e:
            return PretrainedPrediction(
                material_id=material_id, success=False,
                error=f"CIF parse failed: {e}"
            )

        if not self.available or self.model is None:
            return self._fallback_prediction(structure, material_id, formula)

        return self._predict_chgnet(structure, material_id, formula)

    def predict_batch(self, cif_dir: str, limit: int = 100) -> List[PretrainedPrediction]:
        """
        Batch prediction over a directory of CIF files.
        """
        results = []
        cif_files = [f for f in os.listdir(cif_dir) if f.endswith(".cif")][:limit]

        for cif_file in cif_files:
            mat_id = cif_file.replace(".cif", "")
            cif_path = os.path.join(cif_dir, cif_file)
            pred = self.predict_from_cif(cif_path, material_id=mat_id)
            results.append(pred)

        logger.info(f"Batch prediction: {len(results)} materials processed.")
        return results

    def cif_to_graph(self, cif_path: str, radius: float = 8.0, max_neighbors: int = 12) -> Optional[Dict]:
        """
        Convert CIF to PyG-compatible graph dict (for custom GNN training).

        Returns:
            Dict with keys: x, edge_index, edge_attr, num_atoms, formula
        """
        if not HAS_PYMATGEN:
            return None

        try:
            import torch
            structure = Structure.from_file(cif_path)

            # Node features: atomic numbers
            atomic_numbers = [site.specie.number for site in structure]
            x = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(-1)

            # Edge features: neighbor distances
            all_neighbors = structure.get_all_neighbors(radius, include_index=True)
            edge_indices = []
            edge_distances = []

            for i, neighbors in enumerate(all_neighbors):
                sorted_neighbors = sorted(neighbors, key=lambda n: n[1])[:max_neighbors]
                for neighbor in sorted_neighbors:
                    j = neighbor[2]
                    dist = neighbor[1]
                    edge_indices.append([i, j])
                    edge_distances.append(dist)

            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(-1)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 1), dtype=torch.float)

            return {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "num_atoms": len(atomic_numbers),
                "formula": structure.composition.reduced_formula,
                "positions": torch.tensor(structure.cart_coords, dtype=torch.float)
            }

        except Exception as e:
            logger.error(f"CIF to graph conversion failed: {e}")
            return None

    # --- Internal methods ---

    def _predict_chgnet(self, structure, material_id: str, formula: str) -> PretrainedPrediction:
        """Run CHGNet prediction."""
        try:
            prediction = self.model.predict_structure(structure)

            energy_per_atom = float(prediction['e']) / len(structure)
            forces = prediction.get('f', np.zeros((len(structure), 3)))
            forces_norm = float(np.linalg.norm(forces, axis=1).mean())

            # Stability heuristic: lower energy = more stable
            # Normalize around typical intermetallic range
            stability = max(0, min(1, 1.0 - abs(energy_per_atom + 3.0) / 5.0))

            return PretrainedPrediction(
                material_id=material_id,
                formula=formula,
                energy_per_atom_ev=energy_per_atom,
                forces_norm=forces_norm,
                stability_score=stability,
                model_used="CHGNet",
                success=True
            )
        except Exception as e:
            logger.error(f"CHGNet prediction failed for {formula}: {e}")
            return PretrainedPrediction(
                material_id=material_id, formula=formula,
                success=False, error=str(e)
            )

    def _fallback_prediction(self, structure, material_id: str, formula: str) -> PretrainedPrediction:
        """Heuristic fallback when no pretrained model is available."""
        # Use simple composition-based heuristics
        num_elements = len(structure.composition.elements)
        avg_atomic_number = np.mean([site.specie.number for site in structure])

        # Very rough stability estimate based on composition
        stability = max(0, min(1, 0.5 + (num_elements - 1) * 0.1 - abs(avg_atomic_number - 30) / 100))

        return PretrainedPrediction(
            material_id=material_id,
            formula=formula,
            stability_score=stability,
            model_used="heuristic_fallback",
            success=True,
            error="No pretrained model; using composition heuristic"
        )


def get_pretrained_bridge(model: str = "chgnet") -> PretrainedGNNBridge:
    """Factory function."""
    return PretrainedGNNBridge(preferred_model=model)
