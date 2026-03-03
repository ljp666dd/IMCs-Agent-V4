"""
Multi-Descriptor Evaluation Script for CGCNN
Evaluates: Formation Energy + 12 Physicochemical Descriptors
Generates: Grid of Parity Plots with R2/MAE for each target.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.cgcnn import CGCNN
from src.data_ingestion.dataset import CIFDataset

DESCRIPTOR_KEYS = [
    "d_band_center", "d_band_width", "d_band_filling", "DOS_EF", 
    "DOS_window_-0.3_0.3", "unoccupied_d_states_0_0.5", "epsilon_d_minus_EF",
    "sp_d_hybridization", "orbital_ratio_d", "valence_DOS_slope",
    "num_DOS_peaks", "first_peak_position"
]

def evaluate_all():
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
        
    # 3-Way Split
    torch.manual_seed(42)
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    
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
    
    def collect_predictions(loader):
        """Collects true and predicted values for all targets."""
        form_true, form_pred = [], []
        desc_true, desc_pred = [], []
        
        with torch.no_grad():
            for batch in loader:
                out = model(batch)
                
                if hasattr(batch, 'y_formation'):
                    form_true.extend(batch.y_formation.view(-1).tolist())
                    form_pred.extend(out['formation_energy'].view(-1).tolist())
                
                if hasattr(batch, 'y_desc'):
                    # y_desc is batched as (N, 12) where N = batch samples
                    # Reshape to ensure proper 2D structure
                    y_desc_batch = batch.y_desc.view(-1, 12).numpy()
                    pred_desc_batch = out['descriptors'].view(-1, 12).detach().numpy()
                    
                    desc_true.extend(y_desc_batch.tolist())
                    desc_pred.extend(pred_desc_batch.tolist())
        
        # Ensure 2D arrays for descriptors
        if desc_true:
            desc_true_2d = np.row_stack(desc_true)
            desc_pred_2d = np.row_stack(desc_pred)
        else:
            desc_true_2d = np.array([]).reshape(0, 12)
            desc_pred_2d = np.array([]).reshape(0, 12)
        
        return {
            "formation_energy": (np.array(form_true), np.array(form_pred)),
            "descriptors": (desc_true_2d, desc_pred_2d)
        }

    print("Collecting Training Set Predictions...")
    train_data = collect_predictions(train_loader)
    
    print("Collecting Test Set Predictions...")
    test_data = collect_predictions(test_loader)
    
    # 3. Calculate Metrics
    results = []
    
    # Formation Energy
    for name, (tr_true, tr_pred), (te_true, te_pred) in [
        ("Formation Energy", train_data["formation_energy"], test_data["formation_energy"])
    ]:
        results.append({
            "Target": name,
            "Train R2": r2_score(tr_true, tr_pred),
            "Train MAE": mean_absolute_error(tr_true, tr_pred),
            "Test R2": r2_score(te_true, te_pred),
            "Test MAE": mean_absolute_error(te_true, te_pred)
        })
    
    # 12 Descriptors
    desc_tr_true, desc_tr_pred = train_data["descriptors"]
    desc_te_true, desc_te_pred = test_data["descriptors"]
    
    for i, key in enumerate(DESCRIPTOR_KEYS):
        tr_true = desc_tr_true[:, i]
        tr_pred = desc_tr_pred[:, i]
        te_true = desc_te_true[:, i]
        te_pred = desc_te_pred[:, i]
        
        results.append({
            "Target": key,
            "Train R2": r2_score(tr_true, tr_pred),
            "Train MAE": mean_absolute_error(tr_true, tr_pred),
            "Test R2": r2_score(te_true, te_pred),
            "Test MAE": mean_absolute_error(te_true, te_pred)
        })
    
    # Print Table
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("Multi-Target Evaluation Summary")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    
    # Save to CSV
    df.to_csv("data/multi_target_evaluation.csv", index=False)
    print("Saved to data/multi_target_evaluation.csv")
    
    # 4. Plotting Grid (4x4 for 13 targets + 3 empty)
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()
    
    all_targets = ["Formation Energy"] + DESCRIPTOR_KEYS
    all_data = [
        (train_data["formation_energy"], test_data["formation_energy"])
    ] + [
        ((train_data["descriptors"][0][:, i], train_data["descriptors"][1][:, i]),
         (test_data["descriptors"][0][:, i], test_data["descriptors"][1][:, i]))
        for i in range(12)
    ]
    
    for idx, (name, ((tr_y, tr_p), (te_y, te_p))) in enumerate(zip(all_targets, all_data)):
        ax = axes[idx]
        
        # Scatter
        ax.scatter(tr_y, tr_p, alpha=0.3, c='blue', s=10, label='Train')
        ax.scatter(te_y, te_p, alpha=0.6, c='orange', s=15, marker='^', label='Test')
        
        # Diagonal
        all_vals = np.concatenate([tr_y, tr_p, te_y, te_p])
        lims = [all_vals.min(), all_vals.max()]
        ax.plot(lims, lims, 'r--', alpha=0.7)
        
        # Metrics
        te_r2 = r2_score(te_y, te_p)
        ax.set_title(f"{name}\nTest R²={te_r2:.2f}", fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=7)
        
    # Turn off unused axes
    for i in range(len(all_targets), 16):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = "data/multi_target_parity_grid.png"
    plt.savefig(save_path, dpi=200)
    print(f"Grid Plot saved to {save_path}")

if __name__ == "__main__":
    evaluate_all()
