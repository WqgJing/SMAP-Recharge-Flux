# Minimal PINN for Richards Equation
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(44)
np.random.seed(44)

# Van Genuchten Soil Model
class VanGenuchten:
    def __init__(self, theta_s=0.4, theta_r=0.1, alpha=0.01, n=1.3, Ks=5e-6):
        self.theta_s, self.theta_r = theta_s, theta_r
        self.alpha, self.n, self.m = alpha, n, 1.0 - 1.0/n
        self.Ks = Ks
        self._tiny = 1e-12

    def theta(self, h):
        abs_h = torch.abs(h)
        denom = (1.0 + (self.alpha * abs_h).pow(self.n)).pow(self.m) + self._tiny
        theta_unsat = self.theta_r + (self.theta_s - self.theta_r) / denom
        return torch.where(h >= 0.0, torch.tensor(self.theta_s, device=h.device), theta_unsat)

    def K(self, h):
        theta_val = self.theta(h)
        Se = torch.clamp((theta_val - self.theta_r)/(self.theta_s - self.theta_r + self._tiny), 1e-6, 1-1e-6)
        K_unsat = self.Ks * (Se**0.5) * (1.0 - (1.0 - Se**(1.0/self.m))**self.m)**2
        return torch.where(h >= 0.0, torch.tensor(self.Ks, device=h.device), torch.clamp(K_unsat, 0.0, self.Ks))

# Neural Networks
class PressureHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, z, t):
        return torch.clamp(self.net(torch.cat([z, t], dim=1)), -50.0, 5.0)

class WaterTableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, t):
        return torch.clamp(torch.abs(self.net(t)) + 0.5, 0.5, 5.0)

# PINN Model
class RichardsPINN(nn.Module):
    def __init__(self, soil_params, q0_data):
        super().__init__()
        self.h_net = PressureHeadNet()
        self.zb_net = WaterTableNet()
        self.soil = VanGenuchten(**soil_params)
        self.q0_times = torch.tensor(q0_data[0], device=device)
        self.q0_values = torch.tensor(q0_data[1], device=device)

    def forward(self, z, t):
        return self.h_net(z, t), self.zb_net(t)

    def surface_flux(self, t):
        # Interpolate q0
        q0 = torch.zeros_like(t)
        for i, t_val in enumerate(t.flatten()):
            if t_val <= self.q0_times[0]:
                q0.flat[i] = self.q0_values[0]
            elif t_val >= self.q0_times[-1]:
                q0.flat[i] = self.q0_values[-1]
            else:
                idx = torch.searchsorted(self.q0_times, t_val)
                alpha = (t_val - self.q0_times[idx-1])/(self.q0_times[idx] - self.q0_times[idx-1])
                q0.flat[i] = self.q0_values[idx-1] + alpha*(self.q0_values[idx] - self.q0_values[idx-1])
        return q0.reshape_as(t)

    def compute_losses(self, z_col, t_col, t_bc, z_ic):
        # PDE loss
        z_col.requires_grad_(True)
        t_col.requires_grad_(True)
        h, _ = self(z_col, t_col)
        theta = self.soil.theta(h)
        dtheta_dt = torch.autograd.grad(theta.sum(), t_col, create_graph=True)[0]
        dh_dz = torch.autograd.grad(h.sum(), z_col, create_graph=True)[0]
        q = -self.soil.K(h) * (dh_dz + 1.0)
        dq_dz = torch.autograd.grad(q.sum(), z_col, create_graph=True)[0]
        loss_pde = ((dtheta_dt - dq_dz)**2).mean()
        
        # BC loss
        z0 = torch.zeros_like(t_bc, requires_grad=True)
        h0, _ = self(z0, t_bc)
        dh_dz0 = torch.autograd.grad(h0.sum(), z0, create_graph=True)[0]
        q_surf = -self.soil.K(h0) * (dh_dz0 + 1.0)
        loss_bc = ((q_surf - self.surface_flux(t_bc))**2).mean()
        
        # Water table
        _, zb = self(torch.zeros_like(t_bc), t_bc)
        h_wt, _ = self(-zb, t_bc)
        loss_wt = (h_wt**2).mean()
        
        # IC
        t0 = torch.zeros_like(z_ic)
        h_ic, zb_ic = self(z_ic, t0)
        loss_ic = ((h_ic - (z_ic + 1.5))**2).mean() + ((zb_ic - 1.5)**2).mean()
        
        return loss_pde + loss_bc + loss_wt + loss_ic

# Training
def train_pinn(q0_data, soil_params, n_epochs=1000):
    model = RichardsPINN(soil_params, q0_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        # Sample points
        z_col = torch.rand(500, 1, device=device) * (-3.0)
        t_col = torch.rand(500, 1, device=device) * 10.0
        t_bc = torch.rand(100, 1, device=device) * 10.0
        z_ic = torch.rand(50, 1, device=device) * (-3.0)
        
        optimizer.zero_grad()
        loss = model.compute_losses(z_col, t_col, t_bc, z_ic)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 200 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.3e}")
    
    return model

# Visualization
def plot_results(model):
    model.eval()
    with torch.no_grad():
        # Water table
        t = torch.linspace(0, 10, 100, device=device).unsqueeze(1)
        _, zb = model(torch.zeros_like(t), t)
        
        # Pressure profiles
        z = torch.linspace(-3, 0, 50, device=device).unsqueeze(1)
        times = [0, 2.5, 5, 7.5, 10]
        profiles = []
        for t_val in times:
            h, _ = model(z, torch.full_like(z, t_val))
            profiles.append(h.cpu())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(t.cpu(), -zb.cpu(), 'r-', linewidth=2)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Water table depth (m)')
    ax1.set_title('Water Table Evolution')
    ax1.grid(True)
    
    for i, (t_val, h) in enumerate(zip(times, profiles)):
        ax2.plot(h, z.cpu(), label=f't={t_val}d')
    ax2.set_xlabel('Pressure head h (m)')
    ax2.set_ylabel('Depth z (m)')
    ax2.set_title('Pressure Head Profiles')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run simulation
if __name__ == "__main__":
    # Create sinusoidal flux
    t_data = np.linspace(0, 10, 20)
    q_data = -5e-7 - 4e-7 * np.sin(2*np.pi*t_data/10)
    q0_data = (t_data.tolist(), q_data.tolist())
    
    # Soil parameters
    soil_params = {'theta_s': 0.4, 'theta_r': 0.1, 'alpha': 0.01, 'n': 1.3, 'Ks': 5e-6}
    
    # Train and plot
    print("Training PINN...")
    model = train_pinn(q0_data, soil_params, n_epochs=1000)
    print("Plotting results...")
    plot_results(model)
