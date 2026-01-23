"""
Improved CGCNN for Formation Energy and DOS Prediction
Enhancements:
1. More convolutional layers (6 instead of 4)
2. Physical atom features (not just one-hot)
3. Larger hidden dimensions (128)
4. Dropout for regularization
5. Learning rate warmup
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import CGConv, global_mean_pool, global_max_pool
from pymatgen.core.structure import Structure
import warnings
warnings.filterwarnings("ignore")


# Physical atomic properties
ATOM_FEATURES = {
    # Atomic number: [electronegativity, atomic_radius, ionization_energy, electron_affinity, group, period]
    # Values normalized to 0-1 range
    1: [0.89, 0.25, 1.00, 0.53, 0.06, 0.14],   # H
    3: [0.41, 0.67, 0.40, 0.44, 0.06, 0.29],   # Li
    4: [0.67, 0.49, 0.69, 0.00, 0.11, 0.29],   # Be
    5: [0.85, 0.42, 0.61, 0.22, 0.17, 0.29],   # B
    6: [1.04, 0.31, 0.83, 0.89, 0.22, 0.29],   # C
    7: [1.25, 0.30, 1.00, 0.00, 0.28, 0.29],   # N
    8: [1.41, 0.27, 1.00, 1.00, 0.33, 0.29],   # O
    # Transition metals
    22: [0.67, 0.83, 0.50, 0.06, 0.22, 0.57],  # Ti
    23: [0.70, 0.69, 0.50, 0.39, 0.28, 0.57],  # V
    24: [0.74, 0.69, 0.50, 0.50, 0.33, 0.57],  # Cr
    25: [0.67, 0.69, 0.55, 0.00, 0.39, 0.57],  # Mn
    26: [0.78, 0.69, 0.58, 0.15, 0.44, 0.57],  # Fe
    27: [0.81, 0.67, 0.58, 0.49, 0.50, 0.57],  # Co
    28: [0.81, 0.67, 0.56, 0.82, 0.56, 0.57],  # Ni
    29: [0.81, 0.67, 0.57, 0.87, 0.61, 0.57],  # Cu
    30: [0.74, 0.75, 0.70, 0.00, 0.67, 0.57],  # Zn
    31: [0.78, 0.75, 0.44, 0.23, 0.72, 0.57],  # Ga
    39: [0.56, 1.00, 0.47, 0.24, 0.17, 0.71],  # Y
    40: [0.67, 0.92, 0.50, 0.32, 0.22, 0.71],  # Zr
    41: [0.70, 0.83, 0.51, 0.61, 0.28, 0.71],  # Nb
    42: [0.89, 0.75, 0.53, 0.54, 0.33, 0.71],  # Mo
    44: [0.93, 0.75, 0.54, 0.76, 0.44, 0.71],  # Ru
    46: [0.93, 0.75, 0.61, 0.41, 0.56, 0.71],  # Pd
    48: [0.78, 0.83, 0.66, 0.00, 0.67, 0.71],  # Cd
    49: [0.78, 0.92, 0.43, 0.28, 0.72, 0.71],  # In
    50: [0.85, 0.83, 0.54, 0.81, 0.78, 0.71],  # Sn
    57: [0.52, 1.08, 0.41, 0.38, 0.17, 0.86],  # La
    58: [0.52, 1.06, 0.42, 0.38, 0.18, 0.86],  # Ce
    59: [0.52, 1.06, 0.40, 0.38, 0.19, 0.86],  # Pr
    60: [0.52, 1.06, 0.40, 0.38, 0.20, 0.86],  # Nd
    73: [0.67, 0.83, 0.57, 0.26, 0.28, 0.86],  # Ta
    74: [0.96, 0.75, 0.58, 0.58, 0.33, 0.86],  # W
    78: [0.93, 0.75, 0.66, 1.50, 0.56, 0.86],  # Pt
}
DEFAULT_FEATURES = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]


class ImprovedCGCNN(nn.Module):
    """Improved CGCNN with deeper architecture and physical features."""
    
    def __init__(self, atom_fea_len=128, n_conv=6, n_phys_features=6, dropout=0.2):
        super().__init__()
        
        # Embedding: one-hot (92) + physical features (6)
        self.embedding = nn.Linear(92 + n_phys_features, atom_fea_len)
        
        # More convolutional layers with residual connections
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(n_conv):
            self.convs.append(CGConv(atom_fea_len, dim=41, batch_norm=True))
            self.batch_norms.append(nn.BatchNorm1d(atom_fea_len))
        
        self.dropout = nn.Dropout(dropout)
        
        # Combined pooling (mean + max)
        self.fc_pool = nn.Linear(atom_fea_len * 2, atom_fea_len)
        
        # Formation energy head
        self.formation_head = nn.Sequential(
            nn.Linear(atom_fea_len, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        # DOS latent head (for orbital DOS prediction)
        self.dos_head = nn.Sequential(
            nn.Linear(atom_fea_len, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 384)  # 128 * 3 channels
        )
    
    def forward(self, data, return_embedding=False):
        x = self.embedding(data.x)
        
        # Convolutional layers with residual
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_res = x
            x = conv(x, data.edge_index, data.edge_attr)
            x = bn(x)
            if i > 0:  # Residual from layer 2 onwards
                x = x + x_res
            x = self.dropout(x)
        
        # Combined pooling
        x_mean = global_mean_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)
        x_pool = torch.cat([x_mean, x_max], dim=-1)
        embedding = torch.relu(self.fc_pool(x_pool))
        
        if return_embedding:
            return embedding
        
        formation = self.formation_head(embedding)
        dos_latent = self.dos_head(embedding)
        
        return {
            "formation_energy": formation,
            "dos_latent": dos_latent.view(-1, 3, 128)
        }


class ImprovedDataset(Dataset):
    """Dataset with physical atom features."""
    
    def __init__(self, root_dir, formation_file, orbital_dos_file=None, pca_models=None,
                 radius=8.0, max_neighbors=12):
        self.root_dir = root_dir
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.pca_models = pca_models
        
        # Load formation energy
        with open(formation_file, 'r') as f:
            data = json.load(f)
        self.formation_map = {str(e['material_id']): e['formation_energy'] for e in data 
                              if e.get('formation_energy') is not None}
        
        # Load DOS if provided
        self.dos_map = {}
        if orbital_dos_file and os.path.exists(orbital_dos_file):
            with open(orbital_dos_file, 'r') as f:
                dos_data = json.load(f)
            self.dos_map = dos_data
        
        # Find valid CIFs
        all_cifs = [f for f in os.listdir(root_dir) if f.endswith('.cif')]
        self.valid_cifs = [c for c in all_cifs if c.replace('.cif', '') in self.formation_map]
        
        print(f"Improved Dataset: {len(self.valid_cifs)} valid samples")
        super().__init__()
    
    def len(self):
        return len(self.valid_cifs)
    
    def get(self, idx):
        cif_name = self.valid_cifs[idx]
        mat_id = cif_name.replace('.cif', '')
        cif_path = os.path.join(self.root_dir, cif_name)
        
        try:
            structure = Structure.from_file(cif_path)
            x = self._get_atom_features(structure)
            edge_index, edge_attr = self._get_edges(structure)
            
            if edge_index.shape[1] == 0:
                return None
            
            formation = self.formation_map[mat_id]
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y_formation=torch.tensor([formation], dtype=torch.float),
                mat_id=mat_id
            )
            
            # Add DOS if available
            if mat_id in self.dos_map and self.pca_models is not None:
                dos_entry = self.dos_map[mat_id]
                s_dos = np.array(dos_entry['s_dos'])
                p_dos = np.array(dos_entry['p_dos'])
                d_dos = np.array(dos_entry['d_dos'])
                
                # Transform to latent
                latent = []
                for i, pca in enumerate(self.pca_models):
                    dos = [s_dos, p_dos, d_dos][i]
                    lat = pca.transform(dos.reshape(1, -1))[0]
                    latent.append(lat)
                
                data.y_dos_latent = torch.tensor(np.array(latent), dtype=torch.float)
                data.has_dos = True
            else:
                data.has_dos = False
            
            return data
        except:
            return None
    
    def _get_atom_features(self, structure):
        """Combined one-hot + physical features."""
        features = []
        for site in structure:
            # One-hot
            one_hot = np.zeros(92)
            z = site.specie.Z
            if z <= 92:
                one_hot[z - 1] = 1
            
            # Physical features
            phys = ATOM_FEATURES.get(z, DEFAULT_FEATURES)
            
            features.append(np.concatenate([one_hot, phys]))
        
        return torch.tensor(np.array(features), dtype=torch.float)
    
    def _get_edges(self, structure):
        edges = []
        edge_attrs = []
        
        try:
            for i, site in enumerate(structure):
                neighbors = structure.get_neighbors(site, self.radius)
                neighbors = sorted(neighbors, key=lambda x: x.nn_distance)[:self.max_neighbors]
                
                for neighbor in neighbors:
                    edges.append([i, neighbor.index])
                    attr = np.exp(-((np.linspace(0, self.radius, 41) - neighbor.nn_distance) ** 2) / 1.0)
                    edge_attrs.append(attr)
        except:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 41), dtype=torch.float)
        
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 41), dtype=torch.float)
        
        return (torch.tensor(edges, dtype=torch.long).T,
                torch.tensor(np.array(edge_attrs), dtype=torch.float))


def train():
    ROOT = os.path.abspath(os.curdir)
    CIF_DIR = os.path.join(ROOT, "data", "theory", "cifs")
    FORMATION_FILE = os.path.join(ROOT, "data", "theory", "formation_energy_full.json")
    MODEL_PATH = os.path.join(ROOT, "data", "cgcnn_improved.pth")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = ImprovedDataset(CIF_DIR, FORMATION_FILE)
    
    # Filter None
    valid_indices = []
    for i in tqdm(range(len(dataset)), desc="Filtering"):
        if dataset[i] is not None:
            valid_indices.append(i)
    
    from torch.utils.data import Subset
    dataset = Subset(dataset, valid_indices)
    print(f"Valid: {len(dataset)}")
    
    # Split
    torch.manual_seed(42)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    print(f"Train: {n_train}, Test: {n_test}")
    
    # Compute normalization
    all_y = []
    for batch in train_loader:
        all_y.extend(batch.y_formation.numpy().flatten())
    y_mean, y_std = np.mean(all_y), np.std(all_y)
    print(f"Target: mean={y_mean:.4f}, std={y_std:.4f}")
    
    # Model
    model = ImprovedCGCNN(atom_fea_len=128, n_conv=6, dropout=0.2).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    mse = nn.MSELoss()
    
    y_mean_t = torch.tensor([y_mean], device=device)
    y_std_t = torch.tensor([y_std], device=device)
    
    # Training
    epochs = 500
    best_r2 = -float('inf')
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            target = (batch.y_formation - y_mean_t) / y_std_t
            loss = mse(out["formation_energy"], target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        
        scheduler.step()
        avg_train = train_loss / n_train
        
        # Validate
        model.eval()
        all_pred, all_true = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out["formation_energy"] * y_std_t + y_mean_t
                all_pred.extend(pred.cpu().numpy().flatten())
                all_true.extend(batch.y_formation.cpu().numpy().flatten())
        
        val_mae = mean_absolute_error(all_true, all_pred)
        val_r2 = r2_score(all_true, all_pred)
        
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Train: {avg_train:.4f} | MAE: {val_mae:.4f} | R2: {val_r2:.4f}")
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_mae = val_mae
            torch.save({
                'model': model.state_dict(),
                'y_mean': y_mean,
                'y_std': y_std
            }, MODEL_PATH)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best Val R2: {best_r2:.4f}")
    print(f"Best Val MAE: {best_mae:.4f} eV/atom")
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()
