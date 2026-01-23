import os
import sys
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.data_ingestion.dataset import CIFDataset
from src.models.cgcnn import CGCNN

def train_theorist():
    # Paths
    # We assume we are in project root
    ROOT_DIR = os.path.abspath(os.curdir)
    CIF_DIR = os.path.join(ROOT_DIR, "data", "theory", "cifs")
    TARGET_JSON = os.path.join(ROOT_DIR, "data", "theory", "mp_data_summary.json")
    DOS_JSON = os.path.join(ROOT_DIR, "data", "theory", "dos_features.json")
    MODEL_Path = os.path.join(ROOT_DIR, "data", "cgcnn_best_model.pth")
    
    if not os.path.exists(DOS_JSON):
        print(f"Warning: {DOS_JSON} not found. Training might fail.")

    # 1. Load Dataset
    print(f"Loading Dataset from {CIF_DIR}...")
    dataset = CIFDataset(
        root_dir=CIF_DIR, 
        target_file=TARGET_JSON, 
        dos_file=DOS_JSON, # Load DOS
        radius=8.0
    )
    
    if len(dataset) == 0:
        print("Dataset is empty. Check if CIFs and JSON labels match.")
        return

    # Split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Multi-head CGCNN (Increased depth for better precision)
    model = CGCNN(orig_atom_fea_len=92, n_conv=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Losses
    mse_criterion = torch.nn.MSELoss()
    
    epochs = 400
    # Fix: Removed 'verbose=True' which causes TypeError in newer PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_loss = float('inf')
    
    print(f"Start Training on {device} for {epochs} epochs (Deep CGCNN, n_conv=5)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_form_mae = 0
        total_dos_mse = 0
        total_desc_mae = 0 # For d-band center
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward
            outputs = model(batch)
            
            pred_form = outputs["formation_energy"]
            pred_dos = outputs["dos"]
            pred_desc = outputs["descriptors"]
            
            # Targets (from modified dataset)
            target_form = batch.y_formation.view(-1, 1)
            target_dos = batch.y_dos.view(-1, 400)
            target_desc = batch.y_desc.view(-1, 12)
            
            # Loss Calculation
            # 1. Formation Energy Loss (Main Task)
            loss_form = mse_criterion(pred_form, target_form)
            # 2. DOS Fingerprint Loss (Auxiliary)
            loss_dos = mse_criterion(pred_dos, target_dos)
            # 3. Descriptor Loss (Auxiliary)
            loss_desc = mse_criterion(pred_desc, target_desc)
            
            # Combined Loss
            loss = loss_form + 0.1 * loss_dos + 0.1 * loss_desc
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            total_form_mae += torch.abs(pred_form - target_form).sum().item()
            total_dos_mse += loss_dos.item() * batch.num_graphs
            # d-band center is index 0 of descriptors
            total_desc_mae += torch.abs(pred_desc[:, 0] - target_desc[:, 0]).sum().item()
            
        # Logging per epoch
        avg_loss = total_loss / len(train_dataset)
        avg_form_mae = total_form_mae / len(train_dataset)
        avg_dbc_mae = total_desc_mae / len(train_dataset)
        
        # Validation
        val_loss = 0
        val_form_mae = 0
        val_dbc_mae = 0
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                
                # Targets
                target_form = batch.y_formation.view(-1, 1)
                target_dos = batch.y_dos.view(-1, 400)
                target_desc = batch.y_desc.view(-1, 12)
                
                loss_form = mse_criterion(outputs["formation_energy"], target_form)
                loss_dos = mse_criterion(outputs["dos"], target_dos)
                loss_desc = mse_criterion(outputs["descriptors"], target_desc)
                
                loss = loss_form + 0.1 * loss_dos + 0.1 * loss_desc
                val_loss += loss.item() * batch.num_graphs
                val_form_mae += torch.abs(outputs["formation_energy"] - target_form).sum().item()
                val_dbc_mae += torch.abs(outputs["descriptors"][:, 0] - target_desc[:, 0]).sum().item()
        
        avg_val_loss = val_loss / len(test_dataset)
        avg_val_form_mae = val_form_mae / len(test_dataset)
        avg_val_dbc_mae = val_dbc_mae / len(test_dataset)
        
        # Step LR
        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        after_lr = optimizer.param_groups[0]['lr']
        
        if after_lr != before_lr:
            print(f"  -> LR Reduced to {after_lr:.6f}")
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (Val: {avg_val_loss:.4f}) | Form MAE: {avg_val_form_mae:.3f} eV | d-band MAE: {avg_val_dbc_mae:.3f} eV")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save BEST model
            torch.save(model.state_dict(), MODEL_Path)
            print(f"  -> Model saved (Val Loss: {best_loss:.4f})")
            
    print(f"Training Complete. Best Model Saved to {MODEL_Path}")

if __name__ == "__main__":
    train_theorist()
