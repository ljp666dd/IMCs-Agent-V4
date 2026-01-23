"""
CGCNN Training V4 - Updated for expanded dataset
Trains CGCNN to predict:
- Formation Energy
- DOS Fingerprint (from orbital DOS total_dos, resampled to 400 points)
- 11 DOS Descriptors

Uses the new data format:
- orbital_pdos.json (轨道投影 DOS)
- dos_descriptors_full.json (11 个描述符)
- formation_energy_full.json (形成能)
"""

import os
import sys
import json
import torch
import numpy as np
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score
from tqdm import tqdm

# Add project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from pymatgen.core.structure import Structure
import warnings
warnings.filterwarnings("ignore")


class CIFDatasetV4(Dataset):
    """Updated dataset for new data format."""
    
    def __init__(self, root_dir, formation_file, orbital_dos_file, descriptors_file, 
                 dos_dim=400, radius=8.0, max_neighbors=12):
        self.root_dir = root_dir
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.dos_dim = dos_dim
        
        # Load formation energy
        with open(formation_file, 'r') as f:
            form_data = json.load(f)
        self.formation_map = {str(e['material_id']): e['formation_energy'] for e in form_data}
        
        # Load orbital DOS
        with open(orbital_dos_file, 'r') as f:
            orbital_data = json.load(f)
        self.orbital_dos_map = orbital_data
        
        # Load descriptors
        with open(descriptors_file, 'r') as f:
            desc_data = json.load(f)
        self.descriptors_map = {e['material_id']: e for e in desc_data}
        
        # Find valid CIFs (have formation + orbital DOS + descriptors)
        all_cifs = [f for f in os.listdir(root_dir) if f.endswith('.cif')]
        self.valid_cifs = []
        
        for cif in all_cifs:
            mat_id = cif.replace('.cif', '')
            if (mat_id in self.formation_map and 
                mat_id in self.orbital_dos_map and 
                mat_id in self.descriptors_map):
                self.valid_cifs.append(cif)
        
        print(f"Dataset: {len(all_cifs)} CIFs, {len(self.valid_cifs)} valid (have all targets)")
        
        super().__init__()
    
    def len(self):
        return len(self.valid_cifs)
    
    def get(self, idx):
        cif_name = self.valid_cifs[idx]
        mat_id = cif_name.replace('.cif', '')
        cif_path = os.path.join(self.root_dir, cif_name)
        
        try:
            structure = Structure.from_file(cif_path)
        except:
            return None
        
        # Build graph
        x = self._get_atom_features(structure)
        edge_index, edge_attr = self._get_edges(structure)
        
        if edge_index.shape[1] == 0:
            return None
        
        # Targets
        # Check formation energy
        formation_energy = self.formation_map.get(mat_id)
        if formation_energy is None:
            return None
            
        # DOS fingerprint (resample to dos_dim points)
        orbital_dos = self.orbital_dos_map.get(mat_id)
        if orbital_dos is None or 'total_dos' not in orbital_dos:
            return None
            
        total_dos = np.array(orbital_dos['total_dos'])
        if len(total_dos) == 0:
            return None
            
        try:
            dos_fingerprint = self._resample_dos(total_dos, self.dos_dim)
            if np.isnan(dos_fingerprint).any():
                return None
        except:
            return None
        
        # Descriptors
        desc = self.descriptors_map.get(mat_id)
        if desc is None:
            return None
            
        desc_keys = [
            'd_band_center', 'd_band_width', 'd_band_filling', 'DOS_EF', 'DOS_window',
            'unoccupied_d_states', 'epsilon_d_minus_EF', 'valence_DOS_slope',
            'num_DOS_peaks', 'first_peak_position', 'total_states'
        ]
        
        desc_values = []
        for key in desc_keys:
            val = desc.get(key)
            if val is None:
                return None
            desc_values.append(val)
            
        descriptors = np.array(desc_values)
        if np.isnan(descriptors).any():
            return None
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_formation=torch.tensor([formation_energy], dtype=torch.float),
            y_dos=torch.tensor(dos_fingerprint, dtype=torch.float),
            y_descriptors=torch.tensor(descriptors, dtype=torch.float),
            mat_id=mat_id
        )
        return data
    
    def _resample_dos(self, dos, target_len):
        """Resample DOS to target length."""
        x_old = np.linspace(0, 1, len(dos))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, dos)
    
    def _get_atom_features(self, structure):
        """One-hot encoding of atom types."""
        features = []
        for site in structure:
            vec = np.zeros(92)
            vec[site.specie.Z - 1] = 1
            features.append(vec)
        return torch.tensor(np.array(features), dtype=torch.float)
    
    def _get_edges(self, structure):
        """Get edges based on neighbor distances."""
        edges = []
        edge_attrs = []
        
        try:
            for i, site in enumerate(structure):
                neighbors = structure.get_neighbors(site, self.radius)
                # Sort by distance and take top max_neighbors
                neighbors = sorted(neighbors, key=lambda x: x.nn_distance)[:self.max_neighbors]
                
                for neighbor in neighbors:
                    j = neighbor.index  # Use the index attribute directly
                    dist = neighbor.nn_distance
                    
                    edges.append([i, j])
                    
                    # Gaussian expansion of distance
                    attr = np.exp(-((np.linspace(0, self.radius, 41) - dist) ** 2) / 1.0)
                    edge_attrs.append(attr)
        except Exception as e:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 41), dtype=torch.float)
        
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 41), dtype=torch.float)
        
        return (torch.tensor(edges, dtype=torch.long).T,
                torch.tensor(np.array(edge_attrs), dtype=torch.float))


def train_v4():
    # Paths
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
    valid_indices = [i for i in range(len(dataset)) if dataset[i] is not None]
    from torch.utils.data import Subset
    dataset = Subset(dataset, valid_indices)
    print(f"Valid samples: {len(dataset)}")
    
    # Split
    torch.manual_seed(42)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Train: {n_train}, Test: {n_test}")
    
    # Compute normalization stats
    print("\nComputing normalization statistics...")
    all_formation = []
    all_dos = []
    all_desc = []
    
    for batch in train_loader:
        all_formation.extend(batch.y_formation.numpy().flatten())
        all_dos.append(batch.y_dos.numpy().reshape(-1, 400))
        all_desc.append(batch.y_descriptors.numpy().reshape(-1, 11))
    
    all_dos = np.vstack(all_dos)
    all_desc = np.vstack(all_desc)
    
    norm_stats = {
        "formation": {"mean": float(np.mean(all_formation)), "std": float(np.std(all_formation))},
        "dos": {"mean": all_dos.mean(axis=0).tolist(), "std": np.maximum(all_dos.std(axis=0), 1e-6).tolist()},
        "descriptors": {"mean": all_desc.mean(axis=0).tolist(), "std": np.maximum(all_desc.std(axis=0), 1e-6).tolist()}
    }
    
    with open(NORM_PATH, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization to {NORM_PATH}")
    
    # Create model
    from src.models.cgcnn import CGCNN
    model = CGCNN(orig_atom_fea_len=92, n_conv=5).to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    mse = MSELoss()
    
    # Move norm stats to device
    form_mean = torch.tensor([norm_stats["formation"]["mean"]], device=device)
    form_std = torch.tensor([norm_stats["formation"]["std"]], device=device)
    dos_mean = torch.tensor(norm_stats["dos"]["mean"], device=device)
    dos_std = torch.tensor(norm_stats["dos"]["std"], device=device)
    desc_mean = torch.tensor(norm_stats["descriptors"]["mean"], device=device)
    desc_std = torch.tensor(norm_stats["descriptors"]["std"], device=device)
    
    # Training
    epochs = 400
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            
            # Normalize targets
            target_form = (batch.y_formation - form_mean) / form_std
            target_dos = (batch.y_dos.view(-1, 400) - dos_mean) / dos_std
            target_desc = (batch.y_descriptors.view(-1, 11) - desc_mean) / desc_std
            
            # Losses
            loss_form = mse(out["formation_energy"], target_form)
            loss_dos = mse(out["dos"], target_dos)
            loss_desc = mse(out["descriptors"], target_desc)
            
            loss = loss_form + 0.5 * loss_dos + loss_desc
            
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
                out = model(batch)
                
                target_form = (batch.y_formation - form_mean) / form_std
                target_dos = (batch.y_dos.view(-1, 400) - dos_mean) / dos_std
                target_desc = (batch.y_descriptors.view(-1, 11) - desc_mean) / desc_std
                
                loss_form = mse(out["formation_energy"], target_form)
                loss_dos = mse(out["dos"], target_dos)
                loss_desc = mse(out["descriptors"], target_desc)
                
                loss = loss_form + 0.5 * loss_dos + loss_desc
                val_loss += loss.item() * batch.num_graphs
        
        avg_val = val_loss / n_test
        
        scheduler.step(avg_val)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODEL_PATH)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    train_v4()
