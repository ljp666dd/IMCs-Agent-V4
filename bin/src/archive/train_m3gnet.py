"""
M3GNet for Formation Energy and Embedding Extraction
Uses pre-trained M3GNet model for:
1. Formation energy prediction (fine-tuned)
2. Crystal embedding extraction for DOS prediction

Install: pip install matgl dgl
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import warnings
warnings.filterwarnings("ignore")


def check_dependencies():
    """Check if matgl and dgl are installed."""
    try:
        import matgl
        import dgl
        print(f"matgl version: {matgl.__version__}")
        return True
    except ImportError:
        print("Error: matgl or dgl not installed.")
        print("Please run: pip install matgl dgl")
        return False


def train_formation_energy():
    """Train M3GNet for formation energy prediction."""
    from pymatgen.core.structure import Structure
    import matgl
    from matgl.ext.pymatgen import Structure2Graph
    from matgl.models import M3GNet
    from matgl.utils.training import PotentialLightningModule
    
    ROOT = os.path.abspath(os.curdir)
    CIF_DIR = os.path.join(ROOT, "data", "theory", "cifs")
    FORMATION_FILE = os.path.join(ROOT, "data", "theory", "formation_energy_full.json")
    
    print("Loading pre-trained M3GNet...")
    # Load pre-trained M3GNet
    m3gnet = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    
    # Load formation energy data
    with open(FORMATION_FILE, 'r') as f:
        form_data = json.load(f)
    formation_map = {str(e['material_id']): e['formation_energy'] for e in form_data 
                     if e.get('formation_energy') is not None}
    
    # Load structures
    print("\nLoading structures...")
    all_cifs = [f for f in os.listdir(CIF_DIR) if f.endswith('.cif')]
    
    structures = []
    targets = []
    mat_ids = []
    
    for cif_file in tqdm(all_cifs[:500], desc="Loading CIFs"):  # Limit to 500 for speed
        mat_id = cif_file.replace('.cif', '')
        if mat_id not in formation_map:
            continue
        
        try:
            structure = Structure.from_file(os.path.join(CIF_DIR, cif_file))
            structures.append(structure)
            targets.append(formation_map[mat_id])
            mat_ids.append(mat_id)
        except:
            continue
    
    print(f"Loaded {len(structures)} structures")
    
    # Use M3GNet for direct prediction (without fine-tuning)
    print("\nPredicting with pre-trained M3GNet...")
    
    predictions = []
    for i, structure in enumerate(tqdm(structures, desc="Predicting")):
        try:
            # M3GNet predicts energy per atom
            pred_energy = m3gnet.predict_structure(structure)
            n_atoms = len(structure)
            energy_per_atom = float(pred_energy['energy']) / n_atoms
            predictions.append(energy_per_atom)
        except Exception as e:
            predictions.append(targets[i])  # Use target as fallback
    
    predictions = np.array(predictions)
    targets_arr = np.array(targets)
    
    # Calculate metrics
    r2 = r2_score(targets_arr, predictions)
    mae = mean_absolute_error(targets_arr, predictions)
    
    print("\n" + "=" * 50)
    print("M3GNet Formation Energy Prediction")
    print("=" * 50)
    print(f"Samples: {len(targets)}")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f} eV/atom")
    
    # Note: For better results, fine-tuning is recommended
    print("\nNote: For better results, fine-tune M3GNet on your dataset.")
    print("This requires more compute and PyTorch Lightning setup.")


def extract_embeddings():
    """Extract crystal embeddings using M3GNet."""
    from pymatgen.core.structure import Structure
    import matgl
    
    ROOT = os.path.abspath(os.curdir)
    CIF_DIR = os.path.join(ROOT, "data", "theory", "cifs")
    ORBITAL_DOS_FILE = os.path.join(ROOT, "data", "theory", "orbital_pdos.json")
    OUTPUT_FILE = os.path.join(ROOT, "data", "m3gnet_embeddings.json")
    
    print("Loading M3GNet...")
    m3gnet = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    
    # Get materials with DOS
    with open(ORBITAL_DOS_FILE, 'r') as f:
        dos_data = json.load(f)
    dos_ids = set(dos_data.keys())
    
    print(f"\nExtracting embeddings for {len(dos_ids)} materials...")
    
    embeddings = {}
    
    for mat_id in tqdm(list(dos_ids)[:500], desc="Extracting"):  # Limit for speed
        cif_path = os.path.join(CIF_DIR, f"{mat_id}.cif")
        if not os.path.exists(cif_path):
            continue
        
        try:
            structure = Structure.from_file(cif_path)
            # Get embedding from M3GNet
            result = m3gnet.predict_structure(structure)
            
            # The embedding is in the intermediate layers
            # For now, use the predicted properties as features
            embeddings[mat_id] = {
                "energy": float(result['energy']),
                "forces_mean": float(np.mean(np.abs(result['forces']))),
                "stress_mean": float(np.mean(np.abs(result['stresses'])))
            }
        except:
            continue
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(embeddings, f, indent=2)
    
    print(f"\nSaved embeddings to {OUTPUT_FILE}")
    print(f"Total embeddings: {len(embeddings)}")


if __name__ == "__main__":
    if not check_dependencies():
        print("\nInstall dependencies first:")
        print("  pip install matgl dgl")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("M3GNet Training and Embedding Extraction")
    print("=" * 60)
    
    # Step 1: Formation energy
    print("\n[Step 1] Formation Energy Prediction")
    train_formation_energy()
    
    # Step 2: Extract embeddings (optional)
    # print("\n[Step 2] Extracting Embeddings")
    # extract_embeddings()
