import os
import json
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from pymatgen.core.structure import Structure
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class CIFDataset(Dataset):
    def __init__(self, root_dir, target_file=None, dos_file=None, adsorption_file=None, radius=8.0, max_neighbors=12):
        """
        Args:
            root_dir (str): Directory containing .cif files.
            target_file (str): Path to mp_data_summary.json (Formation Energy).
            dos_file (str): Path to dos_features.json (DOS Fingerprint + Descriptors).
            adsorption_file (str): Path to h_adsorption_aggregated.json (ΔG_H).
            radius (float): Cutoff radius for neighbor finding.
        """
        self.root_dir = root_dir
        self.radius = radius
        self.max_neighbors = max_neighbors
        
        # 1. Load Formation Energy Targets
        self.formation_targets = {}
        if target_file and os.path.exists(target_file):
            with open(target_file, "r") as f:
                data = json.load(f)
                for entry in data:
                    self.formation_targets[str(entry['material_id'])] = entry.get('formation_energy', 0.0)
                    
        # 2. Load DOS Targets
        self.dos_targets = {}
        if dos_file and os.path.exists(dos_file):
            with open(dos_file, "r") as f:
                dos_data = json.load(f)
                for entry in dos_data:
                    mat_id = str(entry['material_id'])
                    # Store tuple: (fingerprint, descriptors_dict)
                    self.dos_targets[mat_id] = (entry['dos_fingerprint'], entry['descriptors'])
        
        # 3. Load Adsorption Energy Targets (ΔG_H)
        self.adsorption_targets = {}
        if adsorption_file and os.path.exists(adsorption_file):
            with open(adsorption_file, "r") as f:
                ads_data = json.load(f)
                for mat_id, entry in ads_data.items():
                    self.adsorption_targets[mat_id] = entry.get('delta_g_h', float('nan'))
            print(f"Loaded {len(self.adsorption_targets)} ΔG_H targets")

        # 4. Filter Valid Files (Must have Structure AND DOS)
        all_cifs = [f for f in os.listdir(root_dir) if f.endswith(".cif")]
        self.valid_cifs = []
        
        for cif_file in all_cifs:
            mat_id = cif_file.replace(".cif", "")
            # Strict Intersection: Must have formation energy AND DOS data
            if mat_id in self.formation_targets and mat_id in self.dos_targets:
                self.valid_cifs.append(cif_file)
                
        print(f"Dataset Initialized. Total Files: {len(all_cifs)}. Valid (Intersection): {len(self.valid_cifs)}")
                    
        super(CIFDataset, self).__init__()

    def len(self):
        return len(self.valid_cifs)

    def get(self, idx):
        cif_name = self.valid_cifs[idx]
        mat_id = cif_name.replace(".cif", "")
        cif_path = os.path.join(self.root_dir, cif_name)
        
        # 1. Parse Structure
        try:
            structure = Structure.from_file(cif_path)
        except:
            return None 
            
        # 2. Extract Features (Simplified Atomic Number One-Hot)
        atomic_numbers = [site.specie.number for site in structure]
        x = torch.zeros((len(atomic_numbers), 92))
        for i, z in enumerate(atomic_numbers):
            if z <= 92:
                x[i, z-1] = 1.0
                
        # 3. Find Neighbors (Edges)
        all_neighbors = structure.get_all_neighbors(self.radius, include_index=True)
        all_neighbors = [sorted(n, key=lambda x: x[1])[:self.max_neighbors] for n in all_neighbors]
        
        edge_indices = []
        edge_dist = []
        
        for i, neighbors in enumerate(all_neighbors):
            for neighbor in neighbors:
                j = neighbor[2]
                dist = neighbor[1]
                edge_indices.append([i, j])
                edge_dist.append(dist)
                
        if len(edge_indices) == 0:
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_attr = torch.tensor([], dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            dist_tensor = torch.tensor(edge_dist, dtype=torch.float)
            edge_attr = self._gaussian_expansion(dist_tensor)

        # 4. Targets
        # A. Formation Energy
        y_form = torch.tensor([self.formation_targets.get(mat_id, 0.0)], dtype=torch.float)
        
        # B. DOS & Descriptors
        fingerprint, desc_dict = self.dos_targets[mat_id]
        y_dos = torch.tensor(fingerprint, dtype=torch.float) # Size 400
        
        # Convert descriptor dict to fixed vector
        # Order: d_band_center, d_band_width, d_band_filling, DOS_EF, DOS_window...
        # We explicitly list the 12 keys
        keys = [
            "d_band_center", "d_band_width", "d_band_filling", "DOS_EF", 
            "DOS_window_-0.3_0.3", "unoccupied_d_states_0_0.5", "epsilon_d_minus_EF",
            "sp_d_hybridization", "orbital_ratio_d", "valence_DOS_slope",
            "num_DOS_peaks", "first_peak_position"
        ]
        desc_vec = [desc_dict.get(k, 0.0) for k in keys]
        y_desc = torch.tensor(desc_vec, dtype=torch.float) # Size 12
        
        # C. Adsorption Energy (ΔG_H) - Optional
        delta_g_h = self.adsorption_targets.get(mat_id, float('nan'))
        y_delta_g_h = torch.tensor([delta_g_h], dtype=torch.float)
        has_delta_g_h = not np.isnan(delta_g_h)
        
        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y_formation=y_form,
            y_dos=y_dos,
            y_desc=y_desc,
            y_delta_g_h=y_delta_g_h,
            has_delta_g_h=has_delta_g_h
        )
        data.mat_id = mat_id
        return data

    def _gaussian_expansion(self, distances, dmin=0, dmax=8, step=0.2):
        filter_points = torch.arange(dmin, dmax + step, step)
        sigma = step
        return torch.exp(-(distances.unsqueeze(1) - filter_points.unsqueeze(0))**2 / sigma**2)
