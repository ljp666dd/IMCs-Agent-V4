"""
DOSnet2: DOS → ΔG_H Adsorption Energy Prediction
Uses 400-dim DOS fingerprint as input to predict H adsorption energy.
Based on the concept from vxfung/DOSnet paper.
"""

import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


class DOSnet2(nn.Module):
    """
    Simple MLP that predicts ΔG_H from 400-dim DOS fingerprint.
    Uses LayerNorm instead of BatchNorm for small batch sizes.
    """
    def __init__(self, input_dim=400, hidden_dims=[256, 128, 64], dropout=0.2):
        super(DOSnet2, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),  # LayerNorm works with batch_size=1
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, dos):
        """
        Args:
            dos: DOS fingerprint tensor of shape [Batch, 400]
        Returns:
            Predicted ΔG_H of shape [Batch, 1]
        """
        return self.network(dos)


def load_dos_and_adsorption_data():
    """
    Load DOS fingerprints and matched ΔG_H values.
    Returns only samples that have both DOS and ΔG_H data.
    """
    ROOT_DIR = os.path.abspath(os.curdir)
    DOS_JSON = os.path.join(ROOT_DIR, "data", "theory", "dos_features.json")
    ADS_JSON = os.path.join(ROOT_DIR, "data", "adsorption", "h_adsorption_aggregated.json")
    
    # Load DOS data
    with open(DOS_JSON, 'r') as f:
        dos_data = json.load(f)
    
    dos_map = {}
    for entry in dos_data:
        mat_id = str(entry['material_id'])
        dos_map[mat_id] = entry['dos_fingerprint']
    
    # Load adsorption data
    with open(ADS_JSON, 'r') as f:
        ads_data = json.load(f)
    
    # Find intersection
    dos_list = []
    delta_g_h_list = []
    mat_ids = []
    
    for mat_id, entry in ads_data.items():
        if mat_id in dos_map:
            dos_list.append(dos_map[mat_id])
            delta_g_h_list.append(entry['delta_g_h'])
            mat_ids.append(mat_id)
    
    print(f"Found {len(mat_ids)} samples with both DOS and ΔG_H data")
    
    dos_tensor = torch.tensor(dos_list, dtype=torch.float32)
    delta_g_h_tensor = torch.tensor(delta_g_h_list, dtype=torch.float32).unsqueeze(1)
    
    return dos_tensor, delta_g_h_tensor, mat_ids


def train_dosnet2():
    """Train DOSnet2 model on DOS → ΔG_H task."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ROOT_DIR = os.path.abspath(os.curdir)
    MODEL_PATH = os.path.join(ROOT_DIR, "data", "dosnet2_best_model.pth")
    
    # 1. Load Data
    dos_tensor, delta_g_h_tensor, mat_ids = load_dos_and_adsorption_data()
    
    n_samples = dos_tensor.shape[0]
    print(f"Total samples: {n_samples}")
    print(f"ΔG_H range: {delta_g_h_tensor.min().item():.3f} to {delta_g_h_tensor.max().item():.3f} eV")
    print(f"ΔG_H mean: {delta_g_h_tensor.mean().item():.3f} eV")
    
    # 2. Normalize DOS (Z-score)
    dos_mean = dos_tensor.mean(dim=0)
    dos_std = dos_tensor.std(dim=0).clamp(min=1e-6)
    dos_norm = (dos_tensor - dos_mean) / dos_std
    
    # Normalize ΔG_H
    delta_g_h_mean = delta_g_h_tensor.mean()
    delta_g_h_std = delta_g_h_tensor.std().clamp(min=1e-6)
    delta_g_h_norm = (delta_g_h_tensor - delta_g_h_mean) / delta_g_h_std
    
    # 3. Split Data (80% train, 20% validation)
    torch.manual_seed(42)
    n_train = int(0.8 * n_samples)
    indices = torch.randperm(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dos = dos_norm[train_indices]
    train_target = delta_g_h_norm[train_indices]
    val_dos = dos_norm[val_indices]
    val_target = delta_g_h_norm[val_indices]
    val_target_raw = delta_g_h_tensor[val_indices]
    
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    train_dataset = TensorDataset(train_dos, train_target)
    val_dataset = TensorDataset(val_dos, val_target)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 4. Create Model
    model = DOSnet2(input_dim=400, hidden_dims=[256, 128, 64], dropout=0.2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    criterion = nn.MSELoss()
    
    # Move normalization params to device
    delta_g_h_mean = delta_g_h_mean.to(device)
    delta_g_h_std = delta_g_h_std.to(device)
    
    # 5. Training Loop
    epochs = 500
    best_val_mae = float('inf')
    
    print(f"\nStarting DOSnet2 training for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for dos_batch, target_batch in train_loader:
            dos_batch = dos_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(dos_batch)
            loss = criterion(pred, target_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * dos_batch.size(0)
        
        avg_train_loss = train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for dos_batch, target_batch in val_loader:
                dos_batch = dos_batch.to(device)
                pred = model(dos_batch)
                
                # Denormalize
                pred_raw = pred * delta_g_h_std + delta_g_h_mean
                val_preds.append(pred_raw.cpu())
                
            val_targets = val_target_raw.numpy()
        
        val_preds = torch.cat(val_preds, dim=0).numpy()
        
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets.flatten(), val_preds.flatten())
        
        # LR Scheduler
        scheduler.step(val_mae)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val MAE: {val_mae:.3f} eV | Val R2: {val_r2:.3f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_r2 = val_r2
            torch.save(model.state_dict(), MODEL_PATH)
    
    # 6. Final Report
    print("=" * 60)
    print("DOSnet2 Training Complete!")
    print(f"Best Val MAE: {best_val_mae:.3f} eV")
    print(f"Best Val R2:  {best_val_r2:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
    
    # Save normalization params for inference
    norm_params = {
        "dos_mean": dos_mean.tolist(),
        "dos_std": dos_std.tolist(),
        "delta_g_h_mean": float(delta_g_h_mean.cpu()),
        "delta_g_h_std": float(delta_g_h_std.cpu())
    }
    
    norm_path = os.path.join(ROOT_DIR, "data", "dosnet2_normalization.json")
    with open(norm_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    print(f"Normalization params saved to: {norm_path}")


if __name__ == "__main__":
    train_dosnet2()
