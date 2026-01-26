"""
DOS Fingerprint Evaluation Script for CGCNN
Evaluates: 400-dimensional DOS vector prediction quality
Generates: 
  1. Overall MSE/R² metrics
  2. Per-dimension R² analysis
  3. Example DOS curve comparisons (True vs Predicted)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.cgcnn import CGCNN
from src.data_ingestion.dataset import CIFDataset

def evaluate_dos():
    # 1. Load Data
    print("Loading Dataset...")
    dataset = CIFDataset(
        root_dir="data/theory/cifs",
        target_file="data/theory/mp_data_summary.json",
        dos_file="data/theory/dos_features.json"
    )
    
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
        
    # Split
    torch.manual_seed(42)
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    test_len = total_len - train_len
    
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    # 2. Load Model
    print("Loading Model...")
    model_path = "data/cgcnn_best_model.pth"
    if not os.path.exists(model_path):
        print("Model file not found.")
        return

    model = CGCNN(orig_atom_fea_len=92, n_conv=5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    def collect_dos(loader):
        """Collects true and predicted DOS vectors."""
        dos_true, dos_pred = [], []
        mat_ids = []
        
        with torch.no_grad():
            for batch in loader:
                out = model(batch)
                
                if hasattr(batch, 'y_dos'):
                    # DOS shape: (batch_size, 400)
                    dos_true.extend(batch.y_dos.view(-1, 400).numpy().tolist())
                    dos_pred.extend(out['dos'].view(-1, 400).detach().numpy().tolist())
                    
                    # Collect material IDs for labeling
                    if hasattr(batch, 'mat_id'):
                        mat_ids.extend(batch.mat_id)
        
        return np.array(dos_true), np.array(dos_pred), mat_ids

    print("Collecting Test Set DOS Predictions...")
    dos_true, dos_pred, mat_ids = collect_dos(test_loader)
    
    print(f"Collected {len(dos_true)} samples with shape {dos_true.shape}")
    
    # 3. Overall Metrics (Flattened)
    dos_true_flat = dos_true.flatten()
    dos_pred_flat = dos_pred.flatten()
    
    overall_r2 = r2_score(dos_true_flat, dos_pred_flat)
    overall_mse = mean_squared_error(dos_true_flat, dos_pred_flat)
    overall_mae = mean_absolute_error(dos_true_flat, dos_pred_flat)
    
    print("\n" + "=" * 50)
    print("DOS Fingerprint (400-dim) Evaluation (Test Set)")
    print("=" * 50)
    print(f"Overall R2 (All Samples x All Dims): {overall_r2:.4f}")
    print(f"Overall MSE: {overall_mse:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")
    print("=" * 50)
    
    # 4. Per-Dimension R² (Which energy regions are well predicted?)
    per_dim_r2 = []
    for dim in range(400):
        r2 = r2_score(dos_true[:, dim], dos_pred[:, dim])
        per_dim_r2.append(r2)
    
    per_dim_r2 = np.array(per_dim_r2)
    energy_axis = np.linspace(-5, 5, 400)  # Assuming -5 to +5 eV range
    
    # Plot per-dimension R²
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Per-dimension R²
    ax1 = axes[0]
    ax1.plot(energy_axis, per_dim_r2, c='blue', lw=1)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax1.fill_between(energy_axis, 0, per_dim_r2, where=per_dim_r2 > 0, alpha=0.3, color='green')
    ax1.fill_between(energy_axis, 0, per_dim_r2, where=per_dim_r2 < 0, alpha=0.3, color='red')
    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("R2 Score")
    ax1.set_title("Per-Dimension R2 of DOS Prediction")
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Average DOS (True vs Pred)
    ax2 = axes[1]
    avg_true = dos_true.mean(axis=0)
    avg_pred = dos_pred.mean(axis=0)
    ax2.plot(energy_axis, avg_true, label='True (Avg)', c='black', lw=2)
    ax2.plot(energy_axis, avg_pred, label='Predicted (Avg)', c='orange', lw=2, linestyle='--')
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel("DOS (a.u.)")
    ax2.set_title("Average DOS: True vs Predicted")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = "data/dos_evaluation_overview.png"
    plt.savefig(save_path, dpi=200)
    print(f"Saved overview plot to {save_path}")
    
    # 5. Example DOS Curve Comparisons (Pick 6 random samples)
    n_examples = min(6, len(dos_true))
    indices = np.random.choice(len(dos_true), n_examples, replace=False)
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    axes2 = axes2.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes2[i]
        ax.plot(energy_axis, dos_true[idx], label='True', c='black', lw=1.5)
        ax.plot(energy_axis, dos_pred[idx], label='Pred', c='orange', lw=1.5, linestyle='--')
        
        # Per-sample R²
        sample_r2 = r2_score(dos_true[idx], dos_pred[idx])
        mat_id = mat_ids[idx] if idx < len(mat_ids) else f"Sample {idx}"
        ax.set_title(f"{mat_id}\nR2={sample_r2:.2f}", fontsize=9)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path2 = "data/dos_example_comparisons.png"
    plt.savefig(save_path2, dpi=200)
    print(f"Saved example comparisons to {save_path2}")
    
    # Summary statistics
    print(f"\nPer-Dimension R2 Statistics:")
    print(f"  Mean: {per_dim_r2.mean():.4f}")
    print(f"  Std:  {per_dim_r2.std():.4f}")
    print(f"  Min:  {per_dim_r2.min():.4f} @ E={energy_axis[per_dim_r2.argmin()]:.2f} eV")
    print(f"  Max:  {per_dim_r2.max():.4f} @ E={energy_axis[per_dim_r2.argmax()]:.2f} eV")

if __name__ == "__main__":
    evaluate_dos()
