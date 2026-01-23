"""
Evaluate CGCNN V4 Model
Calculates R² and MAE for formation energy, DOS, and descriptors.
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

from torch_geometric.loader import DataLoader
from src.agents.train_cgcnn_v4 import CIFDatasetV4
from src.models.cgcnn import CGCNN


def evaluate():
    ROOT = os.path.abspath(os.curdir)
    CIF_DIR = os.path.join(ROOT, "data", "theory", "cifs")
    FORMATION_FILE = os.path.join(ROOT, "data", "theory", "formation_energy_full.json")
    ORBITAL_DOS_FILE = os.path.join(ROOT, "data", "theory", "orbital_pdos.json")
    DESCRIPTORS_FILE = os.path.join(ROOT, "data", "theory", "dos_descriptors_full.json")
    MODEL_PATH = os.path.join(ROOT, "data", "cgcnn_best_model_v4.pth")
    NORM_PATH = os.path.join(ROOT, "data", "normalization_params_v4.json")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = CIFDatasetV4(
        root_dir=CIF_DIR,
        formation_file=FORMATION_FILE,
        orbital_dos_file=ORBITAL_DOS_FILE,
        descriptors_file=DESCRIPTORS_FILE,
        dos_dim=400
    )
    
    # Filter None entries
    valid_indices = []
    for i in tqdm(range(len(dataset)), desc="Filtering"):
        if dataset[i] is not None:
            valid_indices.append(i)
    
    from torch.utils.data import Subset
    dataset = Subset(dataset, valid_indices)
    print(f"Valid samples: {len(dataset)}")
    
    # Split same as training
    torch.manual_seed(42)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = CGCNN(orig_atom_fea_len=92, n_conv=5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Load normalization
    with open(NORM_PATH, 'r') as f:
        norm = json.load(f)
    
    form_mean = norm["formation"]["mean"]
    form_std = norm["formation"]["std"]
    dos_mean = np.array(norm["dos"]["mean"])
    dos_std = np.array(norm["dos"]["std"])
    desc_mean = np.array(norm["descriptors"]["mean"])
    desc_std = np.array(norm["descriptors"]["std"])
    
    # Evaluate
    all_pred_form = []
    all_true_form = []
    all_pred_dos = []
    all_true_dos = []
    all_pred_desc = []
    all_true_desc = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            out = model(batch)
            
            # Denormalize predictions
            pred_form = out["formation_energy"].cpu().numpy().flatten() * form_std + form_mean
            pred_dos = out["dos"].cpu().numpy() * dos_std + dos_mean
            pred_desc = out["descriptors"].cpu().numpy() * desc_std + desc_mean
            
            # True values
            true_form = batch.y_formation.cpu().numpy().flatten()
            true_dos = batch.y_dos.cpu().numpy().reshape(-1, 400)
            true_desc = batch.y_descriptors.cpu().numpy().reshape(-1, 11)
            
            all_pred_form.extend(pred_form)
            all_true_form.extend(true_form)
            all_pred_dos.append(pred_dos)
            all_true_dos.append(true_dos)
            all_pred_desc.append(pred_desc)
            all_true_desc.append(true_desc)
    
    all_pred_dos = np.vstack(all_pred_dos)
    all_true_dos = np.vstack(all_true_dos)
    all_pred_desc = np.vstack(all_pred_desc)
    all_true_desc = np.vstack(all_true_desc)
    
    # Metrics
    print("\n" + "=" * 50)
    print("CGCNN V4 Evaluation Results")
    print("=" * 50)
    
    # Formation Energy
    r2_form = r2_score(all_true_form, all_pred_form)
    mae_form = mean_absolute_error(all_true_form, all_pred_form)
    print(f"\nFormation Energy:")
    print(f"  R2:  {r2_form:.4f}")
    print(f"  MAE: {mae_form:.4f} eV/atom")
    
    # DOS (average R2 across all points)
    r2_dos_list = []
    for i in range(len(all_true_dos)):
        r2 = r2_score(all_true_dos[i], all_pred_dos[i])
        if not np.isnan(r2):
            r2_dos_list.append(r2)
    mean_r2_dos = np.mean(r2_dos_list) if r2_dos_list else 0
    print(f"\nDOS Fingerprint (400 dim):")
    print(f"  Mean R2: {mean_r2_dos:.4f}")
    
    # Descriptors
    desc_names = ['d_band_center', 'd_band_width', 'd_band_filling', 'DOS_EF', 'DOS_window',
                  'unoccupied_d_states', 'epsilon_d_minus_EF', 'valence_DOS_slope',
                  'num_DOS_peaks', 'first_peak_position', 'total_states']
    
    print(f"\nDescriptors (11 dim):")
    for i, name in enumerate(desc_names):
        r2 = r2_score(all_true_desc[:, i], all_pred_desc[:, i])
        mae = mean_absolute_error(all_true_desc[:, i], all_pred_desc[:, i])
        print(f"  {name:20s}: R2={r2:.4f}, MAE={mae:.4f}")


if __name__ == "__main__":
    evaluate()
