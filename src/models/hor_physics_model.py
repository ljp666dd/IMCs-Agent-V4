import torch
import torch.nn as nn
import torch.nn.functional as F

class HORPhysicsInformedNet(nn.Module):
    """
    A unified ML hub for HOR Catalysis integrating Chemical Physics:
    - Sabatier Volcano Penalties
    - Epistemic Uncertainty Quantification via MC Dropout
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # We output a single value (j0 or overpotential)
        self.out = nn.Linear(hidden_dim, 1)
        
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
    def forward(self, x):
        # We enforce dropout even during evaluation for MC Dropout Uncertainty estimate
        # Note: If regular inference without UQ is needed, standard eval() logic applies,
        # but for UQ, we will explicitly manage the dropout behavior.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)
        
    def predict_with_uncertainty(self, x, num_samples: int = 50):
        """
        Inference with Monte Carlo Dropout to capture epistemic uncertainty (lack of knowledge).
        Returns:
            mean_pred: The predicted activity
            std_pred: The uncertainty (variance) flag
        """
        self.train()  # Force dropout to remain active
        
        with torch.no_grad():
            predictions = torch.stack([self.forward(x) for _ in range(num_samples)])
            
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        self.eval() # Return to eval mode
        return mean_pred, std_pred

class HORPhysicsLoss(nn.Module):
    """
    A custom loss function that knows chemistry.
    Loss = MSE(Data) + lambda * Physics_Penalty
    """
    def __init__(self, 
                 d_band_idx: int, 
                 formation_energy_idx: int,
                 optimal_d_band: float = -2.0, 
                 physics_weight: float = 0.5):
        super().__init__()
        self.d_band_idx = d_band_idx
        self.formation_energy_idx = formation_energy_idx
        self.optimal_d_band = optimal_d_band
        self.physics_weight = physics_weight
        
    def forward(self, pred, target, features):
        """
        features shape: (batch_size, input_dim)
        """
        # 1. Data Fidelity Loss
        mse = F.mse_loss(pred, target)
        
        # 2. Physics Constraints (The Sabatier Volcano Rule)
        # If the model predicts a very high activity (let's say low overpotential or high j0),
        # but the d_band_center is far from the optimal peak, we penalize it.
        # Here we assume pred is high when it's good (e.g. log(j0))
        
        d_band_centers = features[:, self.d_band_idx]
        d_band_deviation = torch.abs(d_band_centers - self.optimal_d_band)
        
        # A simple penalty heuristic: 
        # Large deviation should NOT yield high predicted absolute activity.
        # We penalize cases where (Pred is high) AND (Deviation is high)
        # Assuming pred is appropriately normalized. 
        # You can adjust this to your exact metric orientation
        volcano_violation = torch.relu(pred.squeeze() * d_band_deviation - 1.0)
        
        # 3. Stability Constraint
        # High activity prediction is penalized if formation energy > 0 (thermodynamically unstable)
        formation_energies = features[:, self.formation_energy_idx]
        stability_violation = torch.relu(pred.squeeze() * formation_energies) 
        
        physics_penalty = volcano_violation.mean() + 0.5 * stability_violation.mean()
        
        return mse + self.physics_weight * physics_penalty

