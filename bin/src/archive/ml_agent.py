"""
ML Agent: Multi-Model Training and Comparison
A comprehensive ML library for material property prediction with:
- Multiple ML models (RF, GB, XGBoost, LightGBM, SVR, MLP, etc.)
- Automatic model comparison and selection
- SHAP interpretability analysis
- Feature importance ranking

Usage:
    python src/agents/ml_agent.py

Install dependencies if needed:
    pip install xgboost lightgbm shap
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
import pickle
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor,
    AdaBoostRegressor,
    BaggingRegressor
)
from sklearn.linear_model import (
    Ridge, 
    Lasso, 
    ElasticNet,
    BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pymatgen.core.structure import Structure

# Optional imports
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

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class ModelResult:
    """Store model evaluation results."""
    name: str
    r2_train: float
    r2_test: float
    mae_test: float
    rmse_test: float
    cv_r2_mean: float
    cv_r2_std: float
    model: object


class MLAgent:
    """
    Machine Learning Agent for Material Property Prediction.
    
    Features:
    - Automatic feature extraction from CIF files
    - Multiple ML model comparison
    - Best model selection
    - SHAP interpretability analysis
    """
    
    def __init__(self, cif_dir: str, target_file: str, output_dir: str):
        self.cif_dir = cif_dir
        self.target_file = target_file
        self.output_dir = output_dir
        
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.results: List[ModelResult] = []
        self.best_model = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract comprehensive features from CIF structures."""
        
        print("\n" + "=" * 60)
        print("Step 1: Feature Extraction")
        print("=" * 60)
        
        # Load targets
        with open(self.target_file, 'r') as f:
            data = json.load(f)
        target_map = {str(e['material_id']): e['formation_energy'] for e in data 
                      if e.get('formation_energy') is not None}
        
        print(f"Materials with targets: {len(target_map)}")
        
        # Define feature names
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
        y_list = []
        mat_ids = []
        
        cif_files = [f for f in os.listdir(self.cif_dir) if f.endswith('.cif')]
        
        for cif_file in tqdm(cif_files, desc="Extracting features"):
            mat_id = cif_file.replace('.cif', '')
            if mat_id not in target_map:
                continue
            
            try:
                structure = Structure.from_file(os.path.join(self.cif_dir, cif_file))
                features = self._extract_structure_features(structure)
                
                if features is not None and not np.any(np.isnan(features)):
                    X_list.append(features)
                    y_list.append(target_map[mat_id])
                    mat_ids.append(mat_id)
            except:
                continue
        
        self.X = np.array(X_list)
        self.y = np.array(y_list)
        self.mat_ids = mat_ids
        
        print(f"Extracted features: {self.X.shape}")
        print(f"Feature names: {len(self.feature_names)}")
        
        return self.X, self.y
    
    def _extract_structure_features(self, structure: Structure) -> Optional[np.ndarray]:
        """Extract features from a single structure."""
        try:
            n_atoms = len(structure)
            elements = [site.specie for site in structure]
            
            # Basic counts
            n_elements = len(set(str(e) for e in elements))
            
            # Volume features
            volume = structure.volume
            volume_per_atom = volume / n_atoms
            
            # Atomic properties
            atomic_numbers = [e.Z for e in elements]
            atomic_masses = [float(e.atomic_mass) for e in elements]
            
            # Electronegativity (with fallback)
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
            
            # Density
            total_mass = sum(atomic_masses)
            density = total_mass / volume
            
            # Lattice parameters
            a, b, c = structure.lattice.abc
            alpha, beta, gamma = structure.lattice.angles
            
            # Packing fraction estimate (assuming spherical atoms)
            avg_radius = 1.5  # Angstrom, rough estimate
            sphere_volume = n_atoms * (4/3) * np.pi * (avg_radius ** 3)
            packing_fraction = sphere_volume / volume
            
            features = [
                n_atoms,
                n_elements,
                volume_per_atom,
                density,
                np.mean(atomic_numbers),
                np.std(atomic_numbers) if len(atomic_numbers) > 1 else 0,
                np.mean(atomic_masses),
                np.std(atomic_masses) if len(atomic_masses) > 1 else 0,
                np.mean(electronegativities),
                np.std(electronegativities) if len(electronegativities) > 1 else 0,
                a, b, c,
                alpha, beta, gamma,
                max(atomic_numbers),
                min(atomic_numbers),
                volume,
                packing_fraction
            ]
            
            return np.array(features)
        except:
            return None
    
    def get_models(self) -> Dict[str, object]:
        """Get dictionary of ML models to evaluate."""
        
        models = {
            # Ensemble Methods
            "RandomForest": RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            "AdaBoost": AdaBoostRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42
            ),
            "Bagging": BaggingRegressor(
                n_estimators=50, random_state=42, n_jobs=-1
            ),
            
            # Linear Models
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01, max_iter=5000),
            "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
            "BayesianRidge": BayesianRidge(),
            
            # Other
            "SVR_RBF": SVR(kernel='rbf', C=10, gamma='scale'),
            "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
            "MLP": MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=1000,
                early_stopping=True, random_state=42
            ),
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0
            )
        
        # Add LightGBM if available
        if HAS_LIGHTGBM:
            models["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbose=-1
            )
        
        return models
    
    def train_and_compare(self, test_size: float = 0.2) -> List[ModelResult]:
        """Train all models and compare performance."""
        
        print("\n" + "=" * 60)
        print("Step 2: Model Training and Comparison")
        print("=" * 60)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=test_size, random_state=42
        )
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        models = self.get_models()
        self.results = []
        
        print(f"\nEvaluating {len(models)} models...")
        print("-" * 60)
        
        for name, model in models.items():
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrics
                r2_train = r2_score(y_train, y_pred_train)
                r2_test = r2_score(y_test, y_pred_test)
                mae_test = mean_absolute_error(y_test, y_pred_test)
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, self.y, cv=5, scoring='r2')
                
                result = ModelResult(
                    name=name,
                    r2_train=r2_train,
                    r2_test=r2_test,
                    mae_test=mae_test,
                    rmse_test=rmse_test,
                    cv_r2_mean=cv_scores.mean(),
                    cv_r2_std=cv_scores.std(),
                    model=model
                )
                self.results.append(result)
                
                print(f"{name:20s} | R2: {r2_test:.4f} | MAE: {mae_test:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"{name:20s} | ERROR: {str(e)[:40]}")
        
        # Sort by test R2
        self.results.sort(key=lambda x: x.r2_test, reverse=True)
        
        return self.results
    
    def get_best_model(self) -> ModelResult:
        """Get the best performing model."""
        if not self.results:
            raise ValueError("No models trained yet. Run train_and_compare() first.")
        
        self.best_model = self.results[0]
        return self.best_model
    
    def shap_analysis(self, model_name: Optional[str] = None, n_samples: int = 100):
        """Perform SHAP analysis on the best or specified model."""
        
        if not HAS_SHAP:
            print("SHAP not installed. Run: pip install shap")
            return None
        
        print("\n" + "=" * 60)
        print("Step 3: SHAP Interpretability Analysis")
        print("=" * 60)
        
        # Get model
        if model_name:
            model_result = next((r for r in self.results if r.name == model_name), None)
            if not model_result:
                raise ValueError(f"Model '{model_name}' not found")
        else:
            model_result = self.get_best_model()
        
        print(f"Analyzing: {model_result.name}")
        
        model = model_result.model
        X_scaled = self.scaler.transform(self.X)
        
        # Use subset for speed
        if len(X_scaled) > n_samples:
            indices = np.random.choice(len(X_scaled), n_samples, replace=False)
            X_sample = X_scaled[indices]
        else:
            X_sample = X_scaled
        
        # Create SHAP explainer
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based model
                explainer = shap.TreeExplainer(model)
            else:
                # Model-agnostic
                explainer = shap.KernelExplainer(model.predict, X_sample[:50])
            
            shap_values = explainer.shap_values(X_sample)
            
            # Feature importance from SHAP
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)
            
            print("\nSHAP Feature Importance:")
            print("-" * 40)
            for _, row in feature_importance.head(10).iterrows():
                print(f"{row['feature']:25s}: {row['importance']:.4f}")
            
            # Save SHAP results
            shap_path = os.path.join(self.output_dir, f"shap_{model_result.name}.pkl")
            with open(shap_path, 'wb') as f:
                pickle.dump({
                    'shap_values': shap_values,
                    'feature_names': self.feature_names,
                    'feature_importance': feature_importance
                }, f)
            
            print(f"\nSHAP results saved to: {shap_path}")
            
            return shap_values, feature_importance
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            return None
    
    def save_results(self):
        """Save all results and best model."""
        
        print("\n" + "=" * 60)
        print("Step 4: Saving Results")
        print("=" * 60)
        
        # Save results summary
        summary = []
        for r in self.results:
            summary.append({
                'name': r.name,
                'r2_train': r.r2_train,
                'r2_test': r.r2_test,
                'mae_test': r.mae_test,
                'rmse_test': r.rmse_test,
                'cv_r2_mean': r.cv_r2_mean,
                'cv_r2_std': r.cv_r2_std
            })
        
        summary_path = os.path.join(self.output_dir, "model_comparison.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")
        
        # Save best model
        best = self.get_best_model()
        model_path = os.path.join(self.output_dir, f"best_model_{best.name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': best.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metrics': {
                    'r2_test': best.r2_test,
                    'mae_test': best.mae_test
                }
            }, f)
        print(f"Best model saved: {model_path}")
        
        # Print final summary
        print("\n" + "=" * 60)
        print("BEST MODEL SUMMARY")
        print("=" * 60)
        print(f"Model:     {best.name}")
        print(f"R2 Test:   {best.r2_test:.4f}")
        print(f"MAE Test:  {best.mae_test:.4f} eV/atom")
        print(f"CV R2:     {best.cv_r2_mean:.4f} ± {best.cv_r2_std:.4f}")


def main():
    ROOT = os.path.abspath(os.curdir)
    
    agent = MLAgent(
        cif_dir=os.path.join(ROOT, "data", "theory", "cifs"),
        target_file=os.path.join(ROOT, "data", "theory", "formation_energy_full.json"),
        output_dir=os.path.join(ROOT, "data", "ml_agent")
    )
    
    # Step 1: Extract features
    agent.extract_features()
    
    # Step 2: Train and compare models
    agent.train_and_compare()
    
    # Step 3: SHAP analysis on best model
    agent.shap_analysis()
    
    # Step 4: Save results
    agent.save_results()
    
    print("\n✅ ML Agent completed successfully!")


if __name__ == "__main__":
    main()
