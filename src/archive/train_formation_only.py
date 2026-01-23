"""
CGCNN Formation Energy Only Training
Single-task training for formation energy prediction.
"""

import os
import sys
import json
import torch
import numpy as np
from torch.nn import MSELoss, Linear, Softplus, Sequential
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import CGConv, global_mean_pool
from pymatgen.core.structure import Structure
import warnings
warnings.filterwarnings("ignore")


class CGCNNFormation(torch.nn.Module):
    """Simplified CGCNN for formation energy only."""
    
    def __init__(self, atom_fea_len=64, n_conv=4):
        super().__init__()
        
        self.embedding = Linear(92, atom_fea_len)
        
        self.convs = torch.nn.ModuleList([
            CGConv(atom_fea_len, dim=41, batch_norm=True)
            for _ in range(n_conv)
        ])
        
        self.fc = Sequential(
            Linear(atom_fea_len, 128),
            Softplus(),
            Linear(128, 64),
            Softplus(),
            Linear(64, 1)
        )
    
    def forward(self, data):
        x = self.embedding(data.x)
        
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
        
        x = global_mean_pool(x, data.batch)
        out = self.fc(x)
        return out


class FormationDataset(Dataset):
    """Dataset for formation energy only."""
    
    def __init__(self, root_dir, formation_file, radius=8.0, max_neighbors=12):
        self.root_dir = root_dir
        self.radius = radius
        self.max_neighbors = max_neighbors
        
        with open(formation_file, 'r') as f:
            data = json.load(f)
        self.formation_map = {str(e['material_id']): e['formation_energy'] for e in data 
                              if e.get('formation_energy') is not None}
        
        all_cifs = [f for f in os.listdir(root_dir) if f.endswith('.cif')]
        self.valid_cifs = [c for c in all_cifs if c.replace('.cif', '') in self.formation_map]
        
        print(f"Formation Dataset: {len(self.valid_cifs)} valid samples")
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
            
            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([formation], dtype=torch.float),
                mat_id=mat_id
            )
        except:
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
    FORMATION_FILE = os.path.join(ROOT, "data", "theory", "formation_energy_full.json")
    MODEL_PATH = os.path.join(ROOT, "data", "cgcnn_formation_only.pth")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = FormationDataset(CIF_DIR, FORMATION_FILE)
    
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
    
    # Compute mean/std
    all_y = []
    for batch in train_loader:
        all_y.extend(batch.y.numpy().flatten())
    y_mean, y_std = np.mean(all_y), np.std(all_y)
    print(f"Target: mean={y_mean:.4f}, std={y_std:.4f}")
    
    # Model
    model = CGCNNFormation(atom_fea_len=64, n_conv=4).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    mse = MSELoss()
    
    y_mean_t = torch.tensor([y_mean], device=device)
    y_std_t = torch.tensor([y_std], device=device)
    
    # Training
    epochs = 300
    best_val_loss = float('inf')
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)
            target = (batch.y - y_mean_t) / y_std_t
            loss = mse(pred, target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        
        avg_train = train_loss / n_train
        
        # Validate
        model.eval()
        all_pred, all_true = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch) * y_std_t + y_mean_t
                all_pred.extend(pred.cpu().numpy().flatten())
                all_true.extend(batch.y.cpu().numpy().flatten())
        
        val_mae = mean_absolute_error(all_true, all_pred)
        val_r2 = r2_score(all_true, all_pred)
        
        scheduler.step(val_mae)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train:.4f} | Val MAE: {val_mae:.4f} | Val R2: {val_r2:.4f}")
        
        if val_mae < best_val_loss:
            best_val_loss = val_mae
            best_r2 = val_r2
            torch.save({
                'model': model.state_dict(),
                'y_mean': y_mean,
                'y_std': y_std
            }, MODEL_PATH)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best Val MAE: {best_val_loss:.4f} eV/atom")
    print(f"Best Val R2:  {best_r2:.4f}")
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()
