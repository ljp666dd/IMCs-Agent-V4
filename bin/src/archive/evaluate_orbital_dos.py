"""
Evaluate Orbital DOS Model
Calculates R² for DOS reconstruction in original 2000-dim space.
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_geometric.loader import DataLoader
from src.agents.train_orbital_dos import OrbitalDOSDataset, CGCNNDOSPredictor, LATENT_DIM, N_CHANNELS, DOS_DIM


def evaluate():
    ROOT = os.path.abspath(os.curdir)
    CIF_DIR = os.path.join(ROOT, "data", "theory", "cifs")
    ORBITAL_DOS_FILE = os.path.join(ROOT, "data", "theory", "orbital_pdos.json")
    MODEL_PATH = os.path.join(ROOT, "data", "cgcnn_orbital_dos.pth")
    PCA_PATH = os.path.join(ROOT, "data", "dos_pca_models.pkl")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load PCA models
    print("\nLoading PCA models...")
    with open(PCA_PATH, 'rb') as f:
        pca_models = pickle.load(f)
    
    # Load dataset
    print("Loading dataset...")
    dataset = OrbitalDOSDataset(CIF_DIR, ORBITAL_DOS_FILE, pca_models=pca_models)
    
    valid_indices = []
    for i in tqdm(range(len(dataset)), desc="Filtering"):
        if dataset[i] is not None:
            valid_indices.append(i)
    
    from torch.utils.data import Subset
    dataset = Subset(dataset, valid_indices)
    
    # Same split as training
    torch.manual_seed(42)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    _, test_set = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    print(f"Test samples: {n_test}")
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = CGCNNDOSPredictor(atom_fea_len=64, n_conv=5, latent_dim=LATENT_DIM, n_channels=N_CHANNELS).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    latent_mean = checkpoint['latent_mean'].reshape(1, N_CHANNELS, LATENT_DIM)
    latent_std = checkpoint['latent_std'].reshape(1, N_CHANNELS, LATENT_DIM)
    
    # Evaluate
    print("\nEvaluating...")
    all_pred_s, all_pred_p, all_pred_d = [], [], []
    all_true_s, all_true_p, all_true_d = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            
            # Predict latent
            pred_latent = model(batch).cpu().numpy()  # [B, 3, 128]
            
            # Denormalize
            pred_latent = pred_latent * latent_std + latent_mean
            
            # Get true DOS
            true_dos = batch.y_dos.cpu().numpy()
            
            # Debug: print shape on first batch
            if len(all_pred_s) == 0:
                print(f"  y_dos shape: {true_dos.shape}")
                print(f"  pred_latent shape: {pred_latent.shape}")
            
            batch_size = pred_latent.shape[0]
            
            # Reshape true_dos if needed [B*3, 2000] -> [B, 3, 2000]
            if len(true_dos.shape) == 2:
                true_dos = true_dos.reshape(batch_size, 3, -1)
            
            for i in range(batch_size):
                # Reconstruct each channel
                pred_s = pca_models[0].inverse_transform(pred_latent[i, 0:1, :])[0]
                pred_p = pca_models[1].inverse_transform(pred_latent[i, 1:2, :])[0]
                pred_d = pca_models[2].inverse_transform(pred_latent[i, 2:3, :])[0]
                
                all_pred_s.append(pred_s)
                all_pred_p.append(pred_p)
                all_pred_d.append(pred_d)
                
                all_true_s.append(true_dos[i, 0, :])
                all_true_p.append(true_dos[i, 1, :])
                all_true_d.append(true_dos[i, 2, :])
    
    # Convert to arrays
    all_pred = [np.array(all_pred_s), np.array(all_pred_p), np.array(all_pred_d)]
    all_true = [np.array(all_true_s), np.array(all_true_p), np.array(all_true_d)]
    
    # Calculate R² for each channel
    channel_names = ['s_orbital', 'p_orbital', 'd_orbital']
    
    print("\n" + "=" * 60)
    print("Orbital DOS Evaluation Results")
    print("=" * 60)
    
    for c in range(N_CHANNELS):
        pred = all_pred[c]
        true = all_true[c]
        
        print(f"\nShape check: pred={pred.shape}, true={true.shape}")
        
        # Overall R² (flattened)
        r2_overall = r2_score(true.flatten(), pred.flatten())
        
        # Per-sample R²
        r2_per_sample = []
        for i in range(len(pred)):
            r2 = r2_score(true[i], pred[i])
            if not np.isnan(r2) and not np.isinf(r2):
                r2_per_sample.append(r2)
        
        mean_r2 = np.mean(r2_per_sample) if r2_per_sample else 0
        
        # MAE
        mae = mean_absolute_error(true.flatten(), pred.flatten())
        
        print(f"\n{channel_names[c]}:")
        print(f"  Overall R2:    {r2_overall:.4f}")
        print(f"  Mean Sample R2: {mean_r2:.4f}")
        print(f"  MAE:           {mae:.4f}")
    
    # Plot example comparison
    print("\nGenerating comparison plot...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # Energy grid (assuming -15 to 10 eV)
    energy = np.linspace(-15, 10, DOS_DIM)
    
    for row, c in enumerate(range(N_CHANNELS)):
        for col in range(3):
            idx = col * 10  # Sample every 10th
            ax = axes[row, col]
            ax.plot(energy, all_true_dos[c][idx], 'b-', label='True', alpha=0.7)
            ax.plot(energy, all_pred_dos[c][idx], 'r--', label='Pred', alpha=0.7)
            ax.set_title(f'{channel_names[c]} - Sample {idx}')
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('DOS')
            if row == 0 and col == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'data', 'orbital_dos_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: data/orbital_dos_comparison.png")


if __name__ == "__main__":
    evaluate()
