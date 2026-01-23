"""
DOSNet: Specialized DOS Fingerprint Decoder
Predicts 400-point DOS fingerprint from crystal embedding.
Uses residual MLP architecture for improved reconstruction.
"""

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout

class DOSNet(nn.Module):
    """
    Standalone DOS decoder that takes crystal embeddings and predicts DOS fingerprint.
    Can be used with pre-trained CGCNN embeddings.
    """
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=400, num_layers=4, dropout=0.1):
        """
        Args:
            input_dim: Dimension of input crystal embedding (from CGCNN).
            hidden_dim: Dimension of hidden layers.
            output_dim: Dimension of DOS fingerprint (400).
            num_layers: Number of residual blocks.
            dropout: Dropout probability.
        """
        super(DOSNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = Sequential(
            Linear(input_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Output projection with progressive upsampling
        self.output_proj = Sequential(
            Linear(hidden_dim, hidden_dim * 2),  # 256 -> 512
            BatchNorm1d(hidden_dim * 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim * 2, output_dim),  # 512 -> 400
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Crystal embedding tensor of shape [Batch, input_dim]
            
        Returns:
            DOS fingerprint of shape [Batch, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        
        # Residual blocks
        for block in self.res_blocks:
            h = block(h)
        
        # Output projection
        out = self.output_proj(h)
        
        return out


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        
        self.block = Sequential(
            Linear(dim, dim),
            BatchNorm1d(dim),
            ReLU(),
            Dropout(dropout),
            Linear(dim, dim),
            BatchNorm1d(dim)
        )
        self.relu = ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out


class CGCNNWithDOSNet(nn.Module):
    """
    Combined model: Uses CGCNN for crystal embedding extraction,
    then DOSNet for specialized DOS prediction.
    Other heads (formation, descriptors, delta_g_h) remain in CGCNN.
    """
    
    def __init__(self, cgcnn_model, dosnet_model, freeze_cgcnn=False):
        """
        Args:
            cgcnn_model: Pre-trained CGCNN model.
            dosnet_model: DOSNet decoder.
            freeze_cgcnn: If True, freeze CGCNN parameters during DOSNet training.
        """
        super(CGCNNWithDOSNet, self).__init__()
        
        self.cgcnn = cgcnn_model
        self.dosnet = dosnet_model
        
        if freeze_cgcnn:
            for param in self.cgcnn.parameters():
                param.requires_grad = False
                
    def forward(self, data):
        """
        Forward pass extracting crystal embedding and using DOSNet for DOS.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # CGCNN Embedding extraction
        x = self.cgcnn.embedding(x)
        for conv in self.cgcnn.convs:
            x = conv(x, edge_index, edge_attr)
        
        # Pool to crystal-level embedding
        crystal_feature = self.cgcnn.pooling(x, batch)
        
        # CGCNN heads (except DOS)
        formation_energy = self.cgcnn.formation_head(crystal_feature)
        site_activities = self.cgcnn.site_activity_head(x)  # Node-level
        descriptors = self.cgcnn.descriptor_head(crystal_feature)
        delta_g_h = self.cgcnn.delta_g_h_head(crystal_feature)
        
        # DOSNet for DOS prediction
        dos_fingerprint = self.dosnet(crystal_feature)
        
        return {
            "formation_energy": formation_energy,
            "site_activities": site_activities,
            "dos": dos_fingerprint,
            "descriptors": descriptors,
            "delta_g_h": delta_g_h
        }


# Utility function to extract embeddings from CGCNN
def extract_crystal_embeddings(cgcnn_model, data_loader, device='cpu'):
    """
    Extract crystal embeddings from a pre-trained CGCNN model.
    
    Returns:
        embeddings: Tensor of shape [N_samples, embedding_dim]
        dos_targets: Tensor of shape [N_samples, 400]
    """
    cgcnn_model.eval()
    embeddings = []
    dos_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Extract embedding
            x = cgcnn_model.embedding(batch.x)
            for conv in cgcnn_model.convs:
                x = conv(x, batch.edge_index, batch.edge_attr)
            crystal_feature = cgcnn_model.pooling(x, batch.batch)
            
            embeddings.append(crystal_feature.cpu())
            dos_targets.append(batch.y_dos.view(-1, 400).cpu())
    
    return torch.cat(embeddings, dim=0), torch.cat(dos_targets, dim=0)
