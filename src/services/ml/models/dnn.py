import torch
import torch.nn as nn
from typing import List

class DeepNeuralNetwork(nn.Module):
    """Deep Neural Network for regression."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TransformerRegressor(nn.Module):
    """Transformer-based regressor for tabular data."""
    
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool over sequence
        return self.output_proj(x)

def get_dnn_model(name: str, input_dim: int):
    """Factory for DNN models."""
    if name == "DNN_256_128_64":
        return DeepNeuralNetwork(input_dim, [256, 128, 64])
    elif name == "DNN_512_256_128":
        return DeepNeuralNetwork(input_dim, [512, 256, 128])
    elif name == "DNN_128_64_32":
        return DeepNeuralNetwork(input_dim, [128, 64, 32])
    elif name.startswith("Transformer"):
        # Parse logic if needed, or default
        if "128" in name:
            return TransformerRegressor(input_dim, d_model=128, num_layers=3)
        return TransformerRegressor(input_dim, d_model=64, num_layers=2)
    else:
        # Default
        return DeepNeuralNetwork(input_dim)
