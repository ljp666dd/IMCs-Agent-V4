"""
Quick Baseline: Random Forest for Formation Energy
Tests if the data itself contains predictive information.
"""

import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from pymatgen.core.structure import Structure
import warnings
warnings.filterwarnings("ignore")


def get_composition_features(structure):
    """Extract simple composition-based features."""
    # Element counts
    element_counts = {}
    for site in structure:
        el = str(site.specie)
        element_counts[el] = element_counts.get(el, 0) + 1
    
    # Composition features
    n_atoms = len(structure)
    n_elements = len(element_counts)
    
    # Volume per atom
    volume_per_atom = structure.volume / n_atoms
    
    # Density proxy
    total_mass = sum(site.specie.atomic_mass for site in structure)
    density = total_mass / structure.volume
    
    # Average atomic properties
    avg_z = np.mean([site.specie.Z for site in structure])
    avg_radius = np.mean([site.specie.atomic_radius for site in structure if hasattr(site.specie, 'atomic_radius') and site.specie.atomic_radius])
    
    # Lattice parameters
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    
    features = [
        n_atoms,
        n_elements,
        volume_per_atom,
        density,
        avg_z,
        avg_radius if avg_radius else 0,
        a, b, c,
        alpha, beta, gamma
    ]
    
    return np.array(features)


def main():
    ROOT = os.path.abspath(os.curdir)
    CIF_DIR = os.path.join(ROOT, "data", "theory", "cifs")
    FORMATION_FILE = os.path.join(ROOT, "data", "theory", "formation_energy_full.json")
    
    # Load formation energy
    with open(FORMATION_FILE, 'r') as f:
        data = json.load(f)
    formation_map = {str(e['material_id']): e['formation_energy'] for e in data 
                     if e.get('formation_energy') is not None}
    
    print(f"Materials with formation energy: {len(formation_map)}")
    
    # Extract features
    print("\nExtracting features from CIF files...")
    X = []
    y = []
    
    cif_files = [f for f in os.listdir(CIF_DIR) if f.endswith('.cif')]
    
    for cif_file in tqdm(cif_files):
        mat_id = cif_file.replace('.cif', '')
        if mat_id not in formation_map:
            continue
        
        try:
            structure = Structure.from_file(os.path.join(CIF_DIR, cif_file))
            features = get_composition_features(structure)
            
            if np.any(np.isnan(features)):
                continue
            
            X.append(features)
            y.append(formation_map[mat_id])
        except:
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Extracted features: {X.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Random Forest
    print("\n" + "=" * 50)
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    
    print(f"Random Forest:")
    print(f"  R2:  {r2_rf:.4f}")
    print(f"  MAE: {mae_rf:.4f} eV/atom")
    
    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    
    y_pred_gb = gb.predict(X_test)
    r2_gb = r2_score(y_test, y_pred_gb)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    
    print(f"Gradient Boosting:")
    print(f"  R2:  {r2_gb:.4f}")
    print(f"  MAE: {mae_gb:.4f} eV/atom")
    
    print("\n" + "=" * 50)
    print("Conclusion:")
    if r2_rf > 0.3 or r2_gb > 0.3:
        print("Data contains predictive signal - GNN implementation may have bugs")
    else:
        print("Data may not contain enough structural variation for prediction")
        print("Formation energy range: [{:.3f}, {:.3f}] eV/atom".format(y.min(), y.max()))
        print("Formation energy std: {:.4f}".format(y.std()))


if __name__ == "__main__":
    main()
