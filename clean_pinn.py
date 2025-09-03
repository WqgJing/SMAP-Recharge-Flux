import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Set random seed for reproducibility
torch.manual_seed(44)
np.random.seed(44)



class VanGenuchten:
    def __init__(self, theta_s, theta_r, alpha, n, Ks, l):
        self.theta_s, self.theta_r = theta_s, theta_r
        self.alpha, self.n, self.m = alpha, n, 1.0 - 1.0/n
        self.Ks, self.l = Ks, l
        self._tiny = 1e-12  # minimal guard

    def theta(self, h):
        abs_h = torch.abs(h)
        denom = (1.0 + (self.alpha * abs_h).pow(self.n)).pow(self.m)
        denom = denom + self._tiny                      # avoid 0
        theta_unsat = self.theta_r + (self.theta_s - self.theta_r) / denom
        return torch.where(h >= 0.0,
                           torch.as_tensor(self.theta_s, device=h.device, dtype=h.dtype),
                           theta_unsat)

    def Se(self, h):
        th = self.theta(h)
        Se = (th - self.theta_r) / (self.theta_s - self.theta_r + self._tiny)
        # keep in (0,1) to avoid fractional-power NaNs
        return torch.clamp(Se, 1e-6, 1.0 - 1e-6)

    def K(self, h):
        Se = self.Se(h)
        Se_1m = Se.pow(1.0 / self.m)
        K_unsat = self.Ks * (Se.pow(self.l)) * (1.0 - (1.0 - Se_1m).pow(self.m)).pow(2)
        return torch.where(h >= 0.0,
                           torch.as_tensor(self.Ks, device=h.device, dtype=h.dtype),
                           torch.clamp(K_unsat, 0.0, self.Ks))  # keep nonnegative & ≤ Ks

class PressureHeadNet(nn.Module):
    """Neural network for pressure head h(z,t)"""
    def __init__(self, hidden_dim, num_layers):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, z, t):
        inputs = torch.cat([z, t], dim=1)
        h = self.net(inputs)
        # Constrain pressure head to reasonable range
        return torch.clamp(h, -50.0, 5.0)

class WaterTableNet(nn.Module):
    """Neural network for water table depth z_b(t) with constraints"""
    def __init__(self, hidden_dim, num_layers):
        super().__init__()

        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, t):
        zb = self.net(t)
        # Ensure water table depth is positive and within reasonable bounds
        return zb  # or zb + 0.5 if you want to keep the offset




