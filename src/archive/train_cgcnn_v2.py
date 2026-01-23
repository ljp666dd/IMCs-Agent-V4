"""
CGCNN Training Script v2 - Improved Multi-Task Learning
Key Improvements:
1. Target Normalization (Z-score) for descriptors and DOS
2. Balanced Loss Weights (1:0.5:1.0 for form:dos:desc)
3. Saves normalization parameters for inference
"""

import os
import sys
import json
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import torch.optim as optim

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.data_ingestion.dataset import CIFDataset
from src.models.cgcnn import CGCNN

def compute_normalization_stats(dataset):
    """Compute mean and std for all targets from the dataset."""
    all_form = []
    all_dos = []
    all_desc = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        if data is None:
            continue
        all_form.append(data.y_formation.numpy())
        all_dos.append(data.y_dos.numpy())
        all_desc.append(data.y_desc.numpy())
    
    all_form = np.vstack(all_form)
    all_dos = np.vstack(all_dos)
    all_desc = np.vstack(all_desc)
    
    stats = {
        "formation": {
            "mean": float(all_form.mean()),
            "std": float(all_form.std())
        },
        "dos": {
            "mean": all_dos.mean(axis=0).tolist(),
            "std": all_dos.std(axis=0).tolist()
        },
        "descriptors": {
            "mean": all_desc.mean(axis=0).tolist(),
            "std": all_desc.std(axis=0).tolist()
        }
    }
    
    # Avoid division by zero
    stats["dos"]["std"] = [max(s, 1e-6) for s in stats["dos"]["std"]]
    stats["descriptors"]["std"] = [max(s, 1e-6) for s in stats["descriptors"]["std"]]
    
    return stats

def train_theorist_v2():
    # Paths
    ROOT_DIR = os.path.abspath(os.curdir)
    CIF_DIR = os.path.join(ROOT_DIR, "data", "theory", "cifs")
    TARGET_JSON = os.path.join(ROOT_DIR, "data", "theory", "mp_data_summary.json")
    DOS_JSON = os.path.join(ROOT_DIR, "data", "theory", "dos_features.json")
    MODEL_PATH = os.path.join(ROOT_DIR, "data", "cgcnn_best_model_v2.pth")
    NORM_PATH = os.path.join(ROOT_DIR, "data", "normalization_params.json")
    
    if not os.path.exists(DOS_JSON):
        print(f"Warning: {DOS_JSON} not found. Training might fail.")

    # 1. Load Dataset
    print(f"Loading Dataset from {CIF_DIR}...")
    dataset = CIFDataset(
        root_dir=CIF_DIR, 
        target_file=TARGET_JSON, 
        dos_file=DOS_JSON,
        radius=8.0
    )
    
    if len(dataset) == 0:
        print("Dataset is empty. Check if CIFs and JSON labels match.")
        return

    # 2. Compute Normalization Stats
    print("Computing normalization statistics...")
    norm_stats = compute_normalization_stats(dataset)
    
    # Save for inference
    with open(NORM_PATH, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization params to {NORM_PATH}")
    
    # Convert to tensors
    form_mean = torch.tensor(norm_stats["formation"]["mean"])
    form_std = torch.tensor(norm_stats["formation"]["std"])
    dos_mean = torch.tensor(norm_stats["dos"]["mean"])
    dos_std = torch.tensor(norm_stats["dos"]["std"])
    desc_mean = torch.tensor(norm_stats["descriptors"]["mean"])
    desc_std = torch.tensor(norm_stats["descriptors"]["std"])
    
    # Split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CGCNN(orig_atom_fea_len=92, n_conv=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Move normalization tensors to device
    form_mean, form_std = form_mean.to(device), form_std.to(device)
    dos_mean, dos_std = dos_mean.to(device), dos_std.to(device)
    desc_mean, desc_std = desc_mean.to(device), desc_std.to(device)
    
    mse_criterion = torch.nn.MSELoss()
    
    epochs = 400
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    best_loss = float('inf')
    
    # IMPROVED LOSS WEIGHTS
    w_form = 1.0
    w_dos = 0.5
    w_desc = 1.0  # Increased from 0.1
    
    print(f"Start Training on {device} for {epochs} epochs...")
    print(f"Loss Weights: Form={w_form}, DOS={w_dos}, Desc={w_desc}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_form_mae = 0
        total_desc_mae = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward
            outputs = model(batch)
            
            pred_form = outputs["formation_energy"]
            pred_dos = outputs["dos"]
            pred_desc = outputs["descriptors"]
            
            # Raw targets
            target_form = batch.y_formation.view(-1, 1)
            target_dos = batch.y_dos.view(-1, 400)
            target_desc = batch.y_desc.view(-1, 12)
            
            # NORMALIZE TARGETS
            target_form_norm = (target_form - form_mean) / form_std
            target_dos_norm = (target_dos - dos_mean) / dos_std
            target_desc_norm = (target_desc - desc_mean) / desc_std
            
            # Loss on NORMALIZED targets
            loss_form = mse_criterion(pred_form, target_form_norm)
            loss_dos = mse_criterion(pred_dos, target_dos_norm)
            loss_desc = mse_criterion(pred_desc, target_desc_norm)
            
            # Combined Loss with balanced weights
            loss = w_form * loss_form + w_dos * loss_dos + w_desc * loss_desc
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            
            # Un-normalize for MAE logging
            pred_form_raw = pred_form * form_std + form_mean
            total_form_mae += torch.abs(pred_form_raw - target_form).sum().item()
            
            pred_desc_raw = pred_desc * desc_std + desc_mean
            total_desc_mae += torch.abs(pred_desc_raw[:, 0] - target_desc[:, 0]).sum().item()
            
        avg_loss = total_loss / len(train_dataset)
        avg_form_mae = total_form_mae / len(train_dataset)
        avg_dbc_mae = total_desc_mae / len(train_dataset)
        
        # Validation
        val_loss = 0
        val_form_mae = 0
        val_dbc_mae = 0
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                
                target_form = batch.y_formation.view(-1, 1)
                target_dos = batch.y_dos.view(-1, 400)
                target_desc = batch.y_desc.view(-1, 12)
                
                # Normalize
                target_form_norm = (target_form - form_mean) / form_std
                target_dos_norm = (target_dos - dos_mean) / dos_std
                target_desc_norm = (target_desc - desc_mean) / desc_std
                
                loss_form = mse_criterion(outputs["formation_energy"], target_form_norm)
                loss_dos = mse_criterion(outputs["dos"], target_dos_norm)
                loss_desc = mse_criterion(outputs["descriptors"], target_desc_norm)
                
                loss = w_form * loss_form + w_dos * loss_dos + w_desc * loss_desc
                val_loss += loss.item() * batch.num_graphs
                
                # Un-normalize for MAE
                pred_form_raw = outputs["formation_energy"] * form_std + form_mean
                val_form_mae += torch.abs(pred_form_raw - target_form).sum().item()
                
                pred_desc_raw = outputs["descriptors"] * desc_std + desc_mean
                val_dbc_mae += torch.abs(pred_desc_raw[:, 0] - target_desc[:, 0]).sum().item()
        
        avg_val_loss = val_loss / len(test_dataset)
        avg_val_form_mae = val_form_mae / len(test_dataset)
        avg_val_dbc_mae = val_dbc_mae / len(test_dataset)
        
        # LR Scheduler
        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        after_lr = optimizer.param_groups[0]['lr']
        
        if after_lr != before_lr:
            print(f"  -> LR Reduced to {after_lr:.6f}")
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (Val: {avg_val_loss:.4f}) | Form MAE: {avg_val_form_mae:.3f} eV | d-band MAE: {avg_val_dbc_mae:.3f} eV")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            
    print(f"Training Complete. Best Model Saved to {MODEL_PATH}")
    print(f"Normalization params saved to {NORM_PATH}")

if __name__ == "__main__":
    train_theorist_v2()
