"""
DOSNet Training Script
Trains the specialized DOS decoder using pre-trained CGCNN embeddings.
Option B: Uses existing 400-point DOS fingerprint.
"""

import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import r2_score

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.models.cgcnn import CGCNN
from src.models.dosnet import DOSNet
from src.data_ingestion.dataset import CIFDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

def train_dosnet():
    # Paths
    ROOT_DIR = os.path.abspath(os.curdir)
    CIF_DIR = os.path.join(ROOT_DIR, "data", "theory", "cifs")
    TARGET_JSON = os.path.join(ROOT_DIR, "data", "theory", "mp_data_summary.json")
    DOS_JSON = os.path.join(ROOT_DIR, "data", "theory", "dos_features.json")
    ADS_JSON = os.path.join(ROOT_DIR, "data", "adsorption", "h_adsorption_aggregated.json")
    
    # Pre-trained CGCNN model (use V2 or V3)
    CGCNN_PATH = os.path.join(ROOT_DIR, "data", "cgcnn_best_model_v2.pth")
    DOSNET_PATH = os.path.join(ROOT_DIR, "data", "dosnet_best_model.pth")
    NORM_PATH = os.path.join(ROOT_DIR, "data", "normalization_params.json")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    print("Loading Dataset...")
    dataset = CIFDataset(
        root_dir=CIF_DIR, 
        target_file=TARGET_JSON, 
        dos_file=DOS_JSON,
        adsorption_file=ADS_JSON,
        radius=8.0
    )
    
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
    
    # 2. Load Pre-trained CGCNN
    print("Loading Pre-trained CGCNN...")
    if not os.path.exists(CGCNN_PATH):
        print(f"Pre-trained CGCNN not found at {CGCNN_PATH}")
        print("Please train CGCNN first using train_cgcnn_v2.py or train_cgcnn_v3.py")
        return
    
    cgcnn_model = CGCNN(orig_atom_fea_len=92, n_conv=5)
    # Use strict=False to allow loading older models without delta_g_h_head
    state_dict = torch.load(CGCNN_PATH, map_location='cpu')
    missing_keys, unexpected_keys = cgcnn_model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in state_dict (will be randomly initialized): {missing_keys}")
    cgcnn_model = cgcnn_model.to(device)
    cgcnn_model.eval()
    
    # 3. Load Normalization Params
    if os.path.exists(NORM_PATH):
        with open(NORM_PATH, 'r') as f:
            norm_stats = json.load(f)
        dos_mean = torch.tensor(norm_stats["dos"]["mean"], device=device)
        dos_std = torch.tensor(norm_stats["dos"]["std"], device=device)
        print("Loaded normalization params")
    else:
        print("Warning: No normalization params found, using defaults")
        dos_mean = torch.zeros(400, device=device)
        dos_std = torch.ones(400, device=device)
    
    # 4. Split Data
    torch.manual_seed(42)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = PyGDataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = PyGDataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 5. Extract Crystal Embeddings from CGCNN
    print("Extracting crystal embeddings from CGCNN...")
    
    def extract_embeddings(loader):
        embeddings = []
        dos_targets = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                
                # Extract embedding
                x = cgcnn_model.embedding(batch.x)
                for conv in cgcnn_model.convs:
                    x = conv(x, batch.edge_index, batch.edge_attr)
                crystal_feature = cgcnn_model.pooling(x, batch.batch)
                
                embeddings.append(crystal_feature.cpu())
                dos_targets.append(batch.y_dos.view(-1, 400).cpu())
        
        return torch.cat(embeddings, dim=0), torch.cat(dos_targets, dim=0)
    
    train_embeddings, train_dos = extract_embeddings(train_loader)
    test_embeddings, test_dos = extract_embeddings(test_loader)
    
    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Test embeddings: {test_embeddings.shape}")
    
    # Create TensorDatasets
    train_tensor_dataset = TensorDataset(train_embeddings, train_dos)
    test_tensor_dataset = TensorDataset(test_embeddings, test_dos)
    
    train_tensor_loader = DataLoader(train_tensor_dataset, batch_size=64, shuffle=True)
    test_tensor_loader = DataLoader(test_tensor_dataset, batch_size=64, shuffle=False)
    
    # 6. Create DOSNet Model
    print("Creating DOSNet model...")
    dosnet = DOSNet(
        input_dim=64,  # CGCNN embedding dim (atom_fea_len)
        hidden_dim=256,
        output_dim=400,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    optimizer = optim.Adam(dosnet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    mse_criterion = torch.nn.MSELoss()
    
    # Move normalization to device
    dos_mean_gpu = dos_mean.to(device)
    dos_std_gpu = dos_std.to(device)
    
    # 7. Training Loop
    epochs = 1000
    best_val_loss = float('inf')
    
    print(f"Start DOSNet Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        dosnet.train()
        train_loss = 0
        
        for emb_batch, dos_batch in train_tensor_loader:
            emb_batch = emb_batch.to(device)
            dos_batch = dos_batch.to(device)
            
            optimizer.zero_grad()
            
            # Normalize target
            dos_norm = (dos_batch - dos_mean_gpu) / dos_std_gpu
            
            # Forward
            pred_dos = dosnet(emb_batch)
            loss = mse_criterion(pred_dos, dos_norm)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * emb_batch.size(0)
        
        avg_train_loss = train_loss / len(train_tensor_dataset)
        
        # Validation
        dosnet.eval()
        val_loss = 0
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for emb_batch, dos_batch in test_tensor_loader:
                emb_batch = emb_batch.to(device)
                dos_batch = dos_batch.to(device)
                
                dos_norm = (dos_batch - dos_mean_gpu) / dos_std_gpu
                pred_dos = dosnet(emb_batch)
                loss = mse_criterion(pred_dos, dos_norm)
                val_loss += loss.item() * emb_batch.size(0)
                
                # Denormalize for R2 calculation
                pred_dos_raw = pred_dos * dos_std_gpu + dos_mean_gpu
                all_true.append(dos_batch.cpu().numpy())
                all_pred.append(pred_dos_raw.cpu().numpy())
        
        avg_val_loss = val_loss / len(test_tensor_dataset)
        
        # Calculate R2
        all_true = np.concatenate(all_true, axis=0).flatten()
        all_pred = np.concatenate(all_pred, axis=0).flatten()
        val_r2 = r2_score(all_true, all_pred)
        
        # LR Scheduler
        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        after_lr = optimizer.param_groups[0]['lr']
        
        if after_lr != before_lr:
            print(f"  -> LR Reduced to {after_lr:.6f}")
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val R2: {val_r2:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(dosnet.state_dict(), DOSNET_PATH)
    
    # Final Evaluation
    print("\n" + "=" * 50)
    print("DOSNet Training Complete")
    print("=" * 50)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Validation R2: {val_r2:.4f}")
    print(f"Model saved to {DOSNET_PATH}")

if __name__ == "__main__":
    train_dosnet()
