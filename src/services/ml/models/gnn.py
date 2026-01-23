import torch
import torch.nn as nn
from src.core.logger import get_logger

logger = get_logger(__name__)

# Check for GNN libraries
try:
    import torch_geometric
    from torch_geometric.data import Data, DataLoader as PyGDataLoader
    from torch_geometric.nn import CGConv, global_mean_pool, SchNet as PyGSchNet, MessagePassing
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

class SimpleCGCNN(nn.Module):
    """Simple CGCNN model for crystal property prediction."""
    
    def __init__(self, atom_fea_len: int = 64, n_conv: int = 3, h_fea_len: int = 128):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric required for CGCNN")
        
        self.atom_fea_len = atom_fea_len
        self.embedding = nn.Embedding(100, atom_fea_len)  # Max 100 elements
        
        self.convs = nn.ModuleList([
            CGConv(atom_fea_len, dim=32, batch_norm=True) 
            for _ in range(n_conv)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(atom_fea_len, h_fea_len),
            nn.ReLU(),
            nn.Linear(h_fea_len, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embed atomic numbers
        x = self.embedding(x.long().squeeze(-1))
        
        # Graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        return self.fc(x)

class SimpleSchNet(nn.Module):
    """Simplified SchNet model."""
    
    def __init__(self, hidden_channels: int = 64, num_filters: int = 32,
                 num_interactions: int = 3, cutoff: float = 10.0):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric required for SchNet")
        
        self.model = PyGSchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=50,
            cutoff=cutoff,
            max_num_neighbors=32
        )
    
    def forward(self, z, pos, batch):
        return self.model(z, pos, batch)

class SimpleMEGNet(nn.Module):
    """Simplified MEGNet model."""
    
    def __init__(self, node_features: int = 64, edge_features: int = 32,
                 global_features: int = 16, n_blocks: int = 3):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric required for MEGNet")
        
        # Node embedding
        self.node_embed = nn.Embedding(100, node_features)
        
        # Edge embedding (distance-based)
        self.edge_embed = nn.Sequential(
            nn.Linear(1, edge_features),
            nn.SiLU(),
            nn.Linear(edge_features, edge_features)
        )
        
        # MEGNet blocks (simplified)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                'edge_update': nn.Sequential(
                    nn.Linear(edge_features + 2 * node_features, edge_features),
                    nn.SiLU(),
                    nn.Linear(edge_features, edge_features)
                ),
                'node_update': nn.Sequential(
                    nn.Linear(node_features + edge_features, node_features),
                    nn.SiLU(),
                    nn.Linear(node_features, node_features)
                )
            }))
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(node_features, global_features),
            nn.SiLU(),
            nn.Linear(global_features, 1)
        )
    
    def forward(self, data):
        # Simplified implementation (placeholder for actual MEGNet logic)
        # Ported from ml_agent.py as is
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embed nodes
        x = self.node_embed(x.long().squeeze(-1))
        
        # Embed edges (distance)
        edge_feat = self.edge_embed(edge_attr)
        
        # MEGNet blocks
        for block in self.blocks:
            row, col = edge_index
            edge_input = torch.cat([x[row], x[col], edge_feat], dim=-1)
            edge_feat = edge_feat + block['edge_update'](edge_input)
            
            from torch_scatter import scatter_mean
            agg_edge = scatter_mean(edge_feat, col, dim=0, dim_size=x.size(0))
            node_input = torch.cat([x, agg_edge], dim=-1)
            x = x + block['node_update'](node_input)
        
        x = global_mean_pool(x, batch)
        return self.output(x)

def get_gnn_model(name: str):
    if not HAS_TORCH_GEOMETRIC:
        return None
        
    if name == "CGCNN":
        return SimpleCGCNN()
    elif name == "SchNet":
        return SimpleSchNet()
    elif name == "MEGNet":
        return SimpleMEGNet()
    return None
