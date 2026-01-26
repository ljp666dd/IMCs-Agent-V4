"""
CGCNN to Orbital DOS Prediction
Predicts s, p, d orbital DOS using PCA compression.
- Compresses 2000-dim DOS to 256-dim latent space
- CGCNN predicts latent representation
- Decoder reconstructs full DOS
"""

import os
import sys
import json
import torch
import numpy as np
from torch.nn import MSELoss, Linear, Softplus, Sequential
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from tqdm import tqdm
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import CGConv, global_mean_pool
from pymatgen.core.structure import Structure
import warnings
warnings.filterwarnings("ignore")


# Configuration
DOS_DIM = 2000
LATENT_DIM = 128  # PCA components per channel
N_CHANNELS = 3    # s, p, d


class CGCNNDOSPredictor(torch.nn.Module):
    """CGCNN that predicts latent DOS representation (128 x 3 channels)."""
    
    def __init__(self, atom_fea_len=64, n_conv=5, latent_dim=128, n_channels=3):
        super().__init__()
        
        self.embedding = Linear(92, atom_fea_len)
        
        self.convs = torch.nn.ModuleList([
            CGConv(atom_fea_len, dim=41, batch_norm=True)
            for _ in range(n_conv)
        ])
        
        # Output: latent_dim * n_channels
        self.dos_head = Sequential(
            Linear(atom_fea_len, 256),
            Softplus(),
            Linear(256, 256),
            Softplus(),
            Linear(256, latent_dim * n_channels)
        )
        
        self.latent_dim = latent_dim
        self.n_channels = n_channels
    
    def forward(self, data):
        x = self.embedding(data.x)
        
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
        
        x = global_mean_pool(x, data.batch)
        dos_latent = self.dos_head(x)
        
        return dos_latent.view(-1, self.n_channels, self.latent_dim)


class OrbitalDOSDataset(Dataset):
    """Dataset for orbital DOS prediction."""
    
    def __init__(self, root_dir, orbital_dos_file, pca_models=None, radius=8.0, max_neighbors=12):
        self.root_dir = root_dir
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.pca_models = pca_models
        
        with open(orbital_dos_file, 'r') as f:
            self.orbital_dos = json.load(f)
        
        all_cifs = [f for f in os.listdir(root_dir) if f.endswith('.cif')]
        self.valid_cifs = [c for c in all_cifs if c.replace('.cif', '') in self.orbital_dos]
        
        print(f"Orbital DOS Dataset: {len(self.valid_cifs)} valid samples")
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
            
            dos_data = self.orbital_dos[mat_id]
            s_dos = np.array(dos_data['s_dos'])
            p_dos = np.array(dos_data['p_dos'])
            d_dos = np.array(dos_data['d_dos'])
            
            if len(s_dos) != DOS_DIM:
                return None
            
            # Stack as [3, 2000]
            full_dos = np.stack([s_dos, p_dos, d_dos], axis=0)
            
            # Transform to latent if PCA models provided
            if self.pca_models is not None:
                latent = []
                for i, pca in enumerate(self.pca_models):
                    lat = pca.transform(full_dos[i:i+1, :])[0]
                    latent.append(lat)
                latent = np.array(latent)  # [3, latent_dim]
            else:
                latent = full_dos  # Return raw during PCA fitting phase
            
            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y_dos=torch.tensor(full_dos, dtype=torch.float),
                y_latent=torch.tensor(latent, dtype=torch.float),
                mat_id=mat_id
            )
        except Exception as e:
            return None
    
    def _get_atom_features(self, structure):
        features = []
        for site in structure:
            vec = np.zeros(92)
            vec[site.specie.Z - 1] = 1
            features.append(vec)
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
    ORBITAL_DOS_FILE = os.path.join(ROOT, "data", "theory", "orbital_pdos.json")
    MODEL_PATH = os.path.join(ROOT, "data", "cgcnn_orbital_dos.pth")
    PCA_PATH = os.path.join(ROOT, "data", "dos_pca_models.pkl")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Step 1: Load data and fit PCA
    print("\nStep 1: Loading data for PCA fitting...")
    dataset_raw = OrbitalDOSDataset(CIF_DIR, ORBITAL_DOS_FILE, pca_models=None)
    
    # Collect all DOS data
    all_s_dos = []
    all_p_dos = []
    all_d_dos = []
    
    valid_indices = []
    for i in tqdm(range(len(dataset_raw)), desc="Collecting DOS"):
        data = dataset_raw[i]
        if data is not None:
            valid_indices.append(i)
            full_dos = data.y_dos.numpy()
            all_s_dos.append(full_dos[0])
            all_p_dos.append(full_dos[1])
            all_d_dos.append(full_dos[2])
    
    all_s_dos = np.array(all_s_dos)
    all_p_dos = np.array(all_p_dos)
    all_d_dos = np.array(all_d_dos)
    
    print(f"Valid samples: {len(valid_indices)}")
    print(f"DOS shape: {all_s_dos.shape}")
    
    # Fit PCA for each channel
    print("\nFitting PCA models...")
    pca_s = PCA(n_components=LATENT_DIM)
    pca_p = PCA(n_components=LATENT_DIM)
    pca_d = PCA(n_components=LATENT_DIM)
    
    pca_s.fit(all_s_dos)
    pca_p.fit(all_p_dos)
    pca_d.fit(all_d_dos)
    
    print(f"PCA variance explained: s={pca_s.explained_variance_ratio_.sum():.4f}, "
          f"p={pca_p.explained_variance_ratio_.sum():.4f}, d={pca_d.explained_variance_ratio_.sum():.4f}")
    
    pca_models = [pca_s, pca_p, pca_d]
    
    with open(PCA_PATH, 'wb') as f:
        pickle.dump(pca_models, f)
    print(f"Saved PCA models to {PCA_PATH}")
    
    # Step 2: Create dataset with PCA transform
    print("\nStep 2: Creating dataset with PCA transform...")
    dataset = OrbitalDOSDataset(CIF_DIR, ORBITAL_DOS_FILE, pca_models=pca_models)
    
    from torch.utils.data import Subset
    dataset = Subset(dataset, valid_indices)
    
    # Split
    torch.manual_seed(42)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    print(f"Train: {n_train}, Test: {n_test}")
    
    # Compute normalization for latent space
    all_latent = []
    for batch in train_loader:
        all_latent.append(batch.y_latent.numpy().reshape(-1, N_CHANNELS * LATENT_DIM))
    all_latent = np.vstack(all_latent)
    latent_mean = all_latent.mean(axis=0)
    latent_std = np.maximum(all_latent.std(axis=0), 1e-6)
    
    # Model
    model = CGCNNDOSPredictor(atom_fea_len=64, n_conv=5, latent_dim=LATENT_DIM, n_channels=N_CHANNELS).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    mse = MSELoss()
    
    latent_mean_t = torch.tensor(latent_mean.reshape(1, N_CHANNELS, LATENT_DIM), device=device, dtype=torch.float)
    latent_std_t = torch.tensor(latent_std.reshape(1, N_CHANNELS, LATENT_DIM), device=device, dtype=torch.float)
    
    # Training
    epochs = 300
    best_val_loss = float('inf')
    
    print(f"\nStep 3: Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)  # [B, 3, 128]
            target = (batch.y_latent.view(-1, N_CHANNELS, LATENT_DIM) - latent_mean_t) / latent_std_t
            
            loss = mse(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        
        avg_train = train_loss / n_train
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                target = (batch.y_latent.view(-1, N_CHANNELS, LATENT_DIM) - latent_mean_t) / latent_std_t
                loss = mse(pred, target)
                val_loss += loss.item() * batch.num_graphs
        
        avg_val = val_loss / n_test
        scheduler.step(avg_val)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'model': model.state_dict(),
                'latent_mean': latent_mean,
                'latent_std': latent_std
            }, MODEL_PATH)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"PCA models saved: {PCA_PATH}")


if __name__ == "__main__":
    train()
