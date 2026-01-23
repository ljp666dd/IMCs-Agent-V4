import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding, Sequential, ModuleList, BatchNorm1d, Softplus
from torch_geometric.nn import CGConv, GlobalAttention, Set2Set
from torch_geometric.data import Data

class CGCNN(torch.nn.Module):
    def __init__(self, 
                 orig_atom_fea_len=92, 
                 nbr_fea_len=41, 
                 atom_fea_len=64, 
                 n_conv=3, 
                 h_fea_len=128, 
                 n_h=1):
        """
        CGCNN with Multi-Task Heads.
        
        Args:
            orig_atom_fea_len: Number of atom features in the input.
            nbr_fea_len: Number of bond features.
            atom_fea_len: Number of hidden atom features in the convolutional layers.
            n_conv: Number of convolutional layers.
            h_fea_len: Number of hidden features after pooling.
            n_h: Number of hidden layers in the output MLP.
        """
        super(CGCNN, self).__init__()
        
        # 1. Embedding Block
        self.embedding = Linear(orig_atom_fea_len, atom_fea_len)
        
        # 2. Convolutional Block
        self.convs = ModuleList([
            CGConv(channels=atom_fea_len, dim=nbr_fea_len, batch_norm=True)
            for _ in range(n_conv)
        ])
        
        # 3. Global Pooling (for Bulk Properties)
        # Using soft attention to learn which atoms matter most for formaton energy
        self.pooling = GlobalAttention(Linear(atom_fea_len, 1))
        
        # 4. Prediction Heads
        
        # Head A: Formation Energy (Graph Level)
        # Input: Graph Embedding -> Output: Scalar (eV/atom)
        self.formation_head = Sequential(
            Linear(atom_fea_len, h_fea_len),
            Softplus(),
            Linear(h_fea_len, h_fea_len),
            Softplus(),
            Linear(h_fea_len, 1)
        )
        
        # Head B: Site-Specific H Adsorption (Node Level)
        # Input: Atom Embedding (Skip pooling) -> Output: Scalar (eV)
        # This implementation predicts the 'potential activity' of each atom site.
        self.site_activity_head = Sequential(
            Linear(atom_fea_len, h_fea_len),
            Softplus(),
            Linear(h_fea_len, 1)  # Output: adsorption energy at this node
        )
        
        # Head C: High-Res DOS Fingerprint (Graph Level, Vector Output)
        # Input: Graph Embedding -> Output: Vector (Size 400)
        self.dos_head = Sequential(
            Linear(atom_fea_len, h_fea_len),
            Softplus(),
            Linear(h_fea_len, 400) 
        )
        
        # Head D: Physical Descriptors (Graph Level, Vector Output)
        # Input: Graph Embedding -> Output: Vector (Size 11)
        # Predicts: d-band center, width, filling, etc.
        self.descriptor_head = Sequential(
            Linear(atom_fea_len, h_fea_len),
            Softplus(),
            Linear(h_fea_len, 11) 
        )
        
        # Head E: Adsorption Energy (ΔG_H) (Graph Level, Scalar Output)
        # Input: Graph Embedding -> Output: Scalar
        self.delta_g_h_head = Sequential(
            Linear(atom_fea_len, h_fea_len),
            Softplus(),
            Linear(h_fea_len, 1)
        )

    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyG Data object containing x (atom features), edge_index, edge_attr, batch.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. Embedding
        x = self.embedding(x)
        
        # 2. Convolution
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        
        # ---------------------------
        # Task 2: Site Activity (Node Level)
        # ---------------------------
        site_activities = self.site_activity_head(x)
        
        # ---------------------------
        # Task 1, 3, 4: Bulk Properties (Graph Level)
        # ---------------------------
        # Pool atom features into a single crystal vector
        crystal_feature = self.pooling(x, batch)
        
        formation_energy = self.formation_head(crystal_feature)
        dos_fingerprint = self.dos_head(crystal_feature)
        descriptors = self.descriptor_head(crystal_feature)
        delta_g_h = self.delta_g_h_head(crystal_feature)
        
        return {
            "formation_energy": formation_energy,  # [Batch_Size, 1]
            "site_activities": site_activities,    # [Total_Atoms, 1]
            "dos": dos_fingerprint,                # [Batch_Size, 400]
            "descriptors": descriptors,            # [Batch_Size, 12]
            "delta_g_h": delta_g_h                 # [Batch_Size, 1]
        }
