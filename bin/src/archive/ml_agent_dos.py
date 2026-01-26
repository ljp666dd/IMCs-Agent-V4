"""
ML Agent for DOS and Descriptor Prediction
Predicts:
1. Orbital DOS (s, p, d) using PCA + MultiOutput ML
2. 11 DOS Descriptors directly

Uses the same structure features as formation energy ML agent.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
import pickle
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from pymatgen.core.structure import Structure

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class DOSMLAgent:
    """ML Agent for DOS and Descriptor prediction."""
    
    def __init__(self, cif_dir: str, orbital_dos_file: str, descriptors_file: str, output_dir: str):
        self.cif_dir = cif_dir
        self.orbital_dos_file = orbital_dos_file
        self.descriptors_file = descriptors_file
        self.output_dir = output_dir
        
        self.X = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
        # DOS data
        self.dos_s = None
        self.dos_p = None
        self.dos_d = None
        self.pca_s = None
        self.pca_p = None
        self.pca_d = None
        
        # Descriptor data
        self.descriptors = None
        self.descriptor_names = [
            'd_band_center', 'd_band_width', 'd_band_filling', 'DOS_EF', 'DOS_window',
            'unoccupied_d_states', 'epsilon_d_minus_EF', 'valence_DOS_slope',
            'num_DOS_peaks', 'first_peak_position', 'total_states'
        ]
        
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_features(self) -> np.ndarray:
        """Extract structure features from CIF files."""
        
        print("\n" + "=" * 60)
        print("Step 1: Feature Extraction")
        print("=" * 60)
        
        # Load orbital DOS
        with open(self.orbital_dos_file, 'r') as f:
            dos_data = json.load(f)
        dos_ids = set(dos_data.keys())
        
        # Load descriptors
        with open(self.descriptors_file, 'r') as f:
            desc_data = json.load(f)
        desc_map = {e['material_id']: e for e in desc_data}
        
        self.feature_names = [
            'n_atoms', 'n_elements', 'volume_per_atom', 'density',
            'avg_atomic_number', 'std_atomic_number', 
            'avg_atomic_mass', 'std_atomic_mass',
            'avg_electronegativity', 'std_electronegativity',
            'lattice_a', 'lattice_b', 'lattice_c',
            'alpha', 'beta', 'gamma',
            'max_atomic_number', 'min_atomic_number',
            'volume', 'packing_fraction'
        ]
        
        X_list = []
        dos_s_list, dos_p_list, dos_d_list = [], [], []
        desc_list = []
        mat_ids = []
        
        cif_files = [f for f in os.listdir(self.cif_dir) if f.endswith('.cif')]
        
        for cif_file in tqdm(cif_files, desc="Extracting features"):
            mat_id = cif_file.replace('.cif', '')
            
            if mat_id not in dos_ids or mat_id not in desc_map:
                continue
            
            try:
                structure = Structure.from_file(os.path.join(self.cif_dir, cif_file))
                features = self._extract_structure_features(structure)
                
                if features is None or np.any(np.isnan(features)):
                    continue
                
                # Get DOS
                dos_entry = dos_data[mat_id]
                s_dos = np.array(dos_entry['s_dos'])
                p_dos = np.array(dos_entry['p_dos'])
                d_dos = np.array(dos_entry['d_dos'])
                
                if len(s_dos) != 2000:
                    continue
                
                # Get descriptors
                desc = desc_map[mat_id]
                desc_values = [desc.get(name, 0) for name in self.descriptor_names]
                if any(v is None for v in desc_values):
                    continue
                
                X_list.append(features)
                dos_s_list.append(s_dos)
                dos_p_list.append(p_dos)
                dos_d_list.append(d_dos)
                desc_list.append(desc_values)
                mat_ids.append(mat_id)
                
            except Exception as e:
                continue
        
        self.X = np.array(X_list)
        self.dos_s = np.array(dos_s_list)
        self.dos_p = np.array(dos_p_list)
        self.dos_d = np.array(dos_d_list)
        self.descriptors = np.array(desc_list)
        self.mat_ids = mat_ids
        
        print(f"Samples: {len(self.X)}")
        print(f"Features: {self.X.shape[1]}")
        print(f"DOS shape: {self.dos_s.shape}")
        print(f"Descriptors: {self.descriptors.shape}")
        
        return self.X
    
    def _extract_structure_features(self, structure: Structure) -> np.ndarray:
        """Extract features from a single structure."""
        try:
            n_atoms = len(structure)
            elements = [site.specie for site in structure]
            n_elements = len(set(str(e) for e in elements))
            
            volume = structure.volume
            volume_per_atom = volume / n_atoms
            
            atomic_numbers = [e.Z for e in elements]
            atomic_masses = [float(e.atomic_mass) for e in elements]
            
            electronegativities = []
            for e in elements:
                try:
                    en = e.X
                    if en is not None:
                        electronegativities.append(en)
                except:
                    pass
            if not electronegativities:
                electronegativities = [1.0]
            
            total_mass = sum(atomic_masses)
            density = total_mass / volume
            
            a, b, c = structure.lattice.abc
            alpha, beta, gamma = structure.lattice.angles
            
            avg_radius = 1.5
            sphere_volume = n_atoms * (4/3) * np.pi * (avg_radius ** 3)
            packing_fraction = sphere_volume / volume
            
            features = [
                n_atoms, n_elements, volume_per_atom, density,
                np.mean(atomic_numbers), np.std(atomic_numbers) if len(atomic_numbers) > 1 else 0,
                np.mean(atomic_masses), np.std(atomic_masses) if len(atomic_masses) > 1 else 0,
                np.mean(electronegativities), np.std(electronegativities) if len(electronegativities) > 1 else 0,
                a, b, c, alpha, beta, gamma,
                max(atomic_numbers), min(atomic_numbers),
                volume, packing_fraction
            ]
            
            return np.array(features)
        except:
            return None
    
    def train_dos_models(self, n_components: int = 64):
        """Train models to predict orbital DOS using PCA compression."""
        
        print("\n" + "=" * 60)
        print("Step 2: DOS Prediction (PCA + ML)")
        print("=" * 60)
        
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Fit PCA for each orbital
        print(f"\nFitting PCA with {n_components} components...")
        self.pca_s = PCA(n_components=n_components)
        self.pca_p = PCA(n_components=n_components)
        self.pca_d = PCA(n_components=n_components)
        
        dos_s_pca = self.pca_s.fit_transform(self.dos_s)
        dos_p_pca = self.pca_p.fit_transform(self.dos_p)
        dos_d_pca = self.pca_d.fit_transform(self.dos_d)
        
        print(f"Variance explained: s={self.pca_s.explained_variance_ratio_.sum():.4f}, "
              f"p={self.pca_p.explained_variance_ratio_.sum():.4f}, "
              f"d={self.pca_d.explained_variance_ratio_.sum():.4f}")
        
        # Train models for each orbital
        results = {}
        
        for name, dos_pca, pca in [
            ('s_orbital', dos_s_pca, self.pca_s),
            ('p_orbital', dos_p_pca, self.pca_p),
            ('d_orbital', dos_d_pca, self.pca_d)
        ]:
            print(f"\nTraining {name}...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, dos_pca, test_size=0.2, random_state=42
            )
            
            # Use MultiOutput with tree model
            if HAS_XGBOOST:
                base_model = xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbosity=0
                )
            else:
                base_model = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )
            
            model = MultiOutputRegressor(base_model)
            model.fit(X_train, y_train)
            
            # Evaluate in PCA space
            y_pred = model.predict(X_test)
            r2_pca = r2_score(y_test, y_pred)
            
            # Evaluate in original DOS space
            y_test_dos = pca.inverse_transform(y_test)
            y_pred_dos = pca.inverse_transform(y_pred)
            
            r2_dos = r2_score(y_test_dos.flatten(), y_pred_dos.flatten())
            mae_dos = mean_absolute_error(y_test_dos.flatten(), y_pred_dos.flatten())
            
            # Per-sample R2
            r2_samples = []
            for i in range(len(y_test_dos)):
                r2 = r2_score(y_test_dos[i], y_pred_dos[i])
                if not np.isnan(r2):
                    r2_samples.append(r2)
            mean_sample_r2 = np.mean(r2_samples) if r2_samples else 0
            
            results[name] = {
                'model': model,
                'pca': pca,
                'r2_pca': r2_pca,
                'r2_dos': r2_dos,
                'mean_sample_r2': mean_sample_r2,
                'mae_dos': mae_dos
            }
            
            print(f"  R2 (PCA space):  {r2_pca:.4f}")
            print(f"  R2 (DOS space):  {r2_dos:.4f}")
            print(f"  Mean Sample R2:  {mean_sample_r2:.4f}")
            print(f"  MAE:             {mae_dos:.4f}")
        
        # Save DOS models
        dos_model_path = os.path.join(self.output_dir, "dos_ml_models.pkl")
        with open(dos_model_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                's_model': results['s_orbital']['model'],
                'p_model': results['p_orbital']['model'],
                'd_model': results['d_orbital']['model'],
                'pca_s': self.pca_s,
                'pca_p': self.pca_p,
                'pca_d': self.pca_d
            }, f)
        print(f"\nDOS models saved: {dos_model_path}")
        
        return results
    
    def train_descriptor_models(self):
        """Train models to predict 11 DOS descriptors."""
        
        print("\n" + "=" * 60)
        print("Step 3: Descriptor Prediction")
        print("=" * 60)
        
        X_scaled = self.scaler.fit_transform(self.X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.descriptors, test_size=0.2, random_state=42
        )
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Model selection
        models = {
            'RandomForest': MultiOutputRegressor(
                RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
            ),
            'ExtraTrees': MultiOutputRegressor(
                ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
            ),
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = MultiOutputRegressor(
                xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, verbosity=0)
            )
        
        if HAS_LIGHTGBM:
            models['LightGBM'] = MultiOutputRegressor(
                lgb.LGBMRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
            )
        
        best_model = None
        best_r2 = -float('inf')
        best_name = None
        
        print(f"\nEvaluating {len(models)} models...")
        print("-" * 60)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Overall R2
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name:15s} | R2: {r2:.4f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name
        
        print(f"\nBest model: {best_name} (R2 = {best_r2:.4f})")
        
        # Per-descriptor R2
        y_pred = best_model.predict(X_test)
        
        print("\nPer-descriptor performance:")
        print("-" * 40)
        descriptor_results = {}
        
        for i, name in enumerate(self.descriptor_names):
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            descriptor_results[name] = {'r2': r2, 'mae': mae}
            print(f"{name:25s}: R2={r2:.4f}, MAE={mae:.4f}")
        
        # Save descriptor model
        desc_model_path = os.path.join(self.output_dir, f"descriptor_ml_model_{best_name}.pkl")
        with open(desc_model_path, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'descriptor_names': self.descriptor_names,
                'results': descriptor_results
            }, f)
        print(f"\nDescriptor model saved: {desc_model_path}")
        
        return best_model, descriptor_results


def main():
    ROOT = os.path.abspath(os.curdir)
    
    agent = DOSMLAgent(
        cif_dir=os.path.join(ROOT, "data", "theory", "cifs"),
        orbital_dos_file=os.path.join(ROOT, "data", "theory", "orbital_pdos.json"),
        descriptors_file=os.path.join(ROOT, "data", "theory", "dos_descriptors_full.json"),
        output_dir=os.path.join(ROOT, "data", "ml_agent")
    )
    
    # Step 1: Extract features
    agent.extract_features()
    
    # Step 2: Train DOS models
    dos_results = agent.train_dos_models(n_components=64)
    
    # Step 3: Train descriptor models
    desc_model, desc_results = agent.train_descriptor_models()
    
    print("\n" + "=" * 60)
    print("✅ DOS ML Agent completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
