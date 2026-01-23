
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.cgcnn import CGCNN
from src.data_ingestion.dataset import CIFDataset

def evaluate():
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
        
    # 3-Way Split (70% Train, 15% Val, 15% Test)
    torch.manual_seed(42)
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
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
    
    def get_metrics_and_preds(loader):
        actuals = []
        preds = []
        with torch.no_grad():
            for batch in loader:
                out = model(batch)
                if hasattr(batch, 'y_formation'):
                    actuals.extend(batch.y_formation.view(-1).tolist())
                    preds.extend(out['formation_energy'].view(-1).tolist())
        
        act_np = np.array(actuals)
        pre_np = np.array(preds)
        
        if len(act_np) == 0: return [], [], 0, 0
            
        r2 = r2_score(act_np, pre_np)
        mae = mean_absolute_error(act_np, pre_np)
        return act_np, pre_np, r2, mae

    print("Evaluating Training Set...")
    tr_y, tr_pred, tr_r2, tr_mae = get_metrics_and_preds(train_loader)
    
    print("Evaluating Validation Set...")
    va_y, va_pred, va_r2, va_mae = get_metrics_and_preds(val_loader)
    
    print("Evaluating Test Set...")
    te_y, te_pred, te_r2, te_mae = get_metrics_and_preds(test_loader)
    
    print("-" * 40)
    print(f"Dataset Split Analysis:")
    print(f"Training Set   (70%): R2 = {tr_r2:.4f}, MAE = {tr_mae:.4f} eV")
    print(f"Validation Set (15%): R2 = {va_r2:.4f}, MAE = {va_mae:.4f} eV")
    print(f"Test Set       (15%): R2 = {te_r2:.4f}, MAE = {te_mae:.4f} eV")
    print("-" * 40)
    
    # 5. Plotting
    plt.figure(figsize=(8, 7))
    
    # Train
    plt.scatter(tr_y, tr_pred, alpha=0.4, c='#1f77b4', edgecolors='none', s=20, label=f'Train (R2={tr_r2:.2f})')
    # Val
    plt.scatter(va_y, va_pred, alpha=0.6, c='#2ca02c', edgecolors='k', s=25, marker='s', label=f'Val   (R2={va_r2:.2f})')
    # Test
    plt.scatter(te_y, te_pred, alpha=0.7, c='#ff7f0e', edgecolors='k', s=30, marker='^', label=f'Test  (R2={te_r2:.2f})')
    
    # Line
    all_y = np.concatenate([tr_y, va_y, te_y])
    all_p = np.concatenate([tr_pred, va_pred, te_pred])
    lims = [np.min([all_y.min(), all_p.min()]), np.max([all_y.max(), all_p.max()])]
    plt.plot(lims, lims, 'r--', alpha=0.8, zorder=0)
    
    plt.xlabel('DFT Calculated (eV/atom)')
    plt.ylabel('CGCNN Predicted (eV/atom)')
    plt.title('Parity Plot: Train / Val / Test')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = "data/evaluation_split_plot.png"
    plt.savefig(save_path, dpi=300)
    print(f"Split Plot saved to {save_path}")

if __name__ == "__main__":
    evaluate()
