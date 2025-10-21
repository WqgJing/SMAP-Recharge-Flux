import torch
import torch.nn as nn
import torch.nn.functional as F


class PressureHeadNet(nn.Module):
    """Neural network for dimensionless pressure head h̃(z̃,t̃)"""

    def __init__(self, hidden_dim, num_layers, t_ref_tilde=None, z_max_tilde=1.0):
        """
        Args:
            hidden_dim: Number of hidden units
            num_layers: Number of hidden layers
            t_ref_tilde: Fixed reference time for scaling (default: 15 days / T)
            z_max_tilde: Maximum expected dimensionless depth for scaling (typically 1.0)
        """
        super().__init__()

        # Use fixed reference time of 15 days if not specified
        if t_ref_tilde is None:
            t_ref_tilde = 15.0 * 86400  # 15 days in seconds, will be normalized by T

        # Store dimensionless normalization parameters for NN input scaling
        self.t_ref_tilde = t_ref_tilde
        self.z_max_tilde = z_max_tilde

        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, z_tilde, t_tilde):
        """
        Args:
            z_tilde: Dimensionless spatial coordinate (typically in [-1, 0])
            t_tilde: Dimensionless time (can exceed t_ref_tilde during extrapolation)

        Returns:
            h_tilde: Dimensionless pressure head (O(1) values)
        """
        # Scale inputs using FIXED reference time (duration-independent)
        t_scaled = t_tilde / self.t_ref_tilde  # [0, t_ref_tilde] -> [0, 1] (can go beyond [0,1])
        z_scaled = (z_tilde + self.z_max_tilde) / self.z_max_tilde  # [-z_max_tilde, 0] -> [0, 1]

        inputs = torch.cat([z_scaled, t_scaled], dim=1)
        h_tilde = self.net(inputs)

        # Output is already dimensionless, no additional scaling needed
        return h_tilde


class WaterTableNet(nn.Module):
    """Neural network for dimensionless water table depth z̃_b(t̃) with z̃_b > 0"""

    def __init__(self, hidden_dim, num_layers, t_ref_tilde=None):
        """
        Args:
            hidden_dim: Number of hidden units
            num_layers: Number of hidden layers
            t_ref_tilde: Fixed reference time for scaling (default: 15 days / T)
        """
        super().__init__()

        # Use fixed reference time of 15 days if not specified
        if t_ref_tilde is None:
            t_ref_tilde = 15.0 * 86400  # 15 days in seconds, will be normalized by T

        # Store dimensionless time normalization parameter
        self.t_ref_tilde = t_ref_tilde

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t_tilde):
        """
        Args:
            t_tilde: Dimensionless time (can exceed t_ref_tilde during extrapolation)

        Returns:
            zb_tilde: Dimensionless water table depth (O(1) values, > 0)
        """
        # Scale time input using FIXED reference time (duration-independent)
        t_scaled = t_tilde / self.t_ref_tilde  # [0, t_ref_tilde] -> [0, 1] (can go beyond [0,1])

        raw = self.net(t_scaled)
        zb_tilde = F.softplus(raw)  # Ensures zb_tilde > 0 smoothly

        # Output is already dimensionless, no additional scaling needed
        return zb_tilde


class RichardsPINN(nn.Module):
    """PINN for Richards equation with moving boundary - NORMALIZED VERSION"""

    def __init__(
        self,
        soil_params,
        theta0_data,
        Sy=0.3,
        zr=0.5,
        h_net_config=None,
        zb_net_config=None,
        normalizer=None,
        zb_initial=1.5,
        t_max=86400,
        z_max_tilde=1.0,
        t_ref_days=15.0,
        ic_profile=None,
        ic_type='obs',
        device='cpu'
    ):
        """
        Args:
            soil_params: Soil parameters dict (for normalizer)
            theta0_data: Tuple of (times, soil_moisture) in DIMENSIONAL form - REQUIRED
            Sy: Specific yield (dimensional)
            zr: Root zone depth (dimensional) [m]
            h_net_config: Config dict for pressure head network
            zb_net_config: Config dict for water table network
            normalizer: NormalizationHelper instance
            zb_initial: Initial water table depth (dimensional) [m]
            t_max: Maximum time (dimensional) [s]
            z_max_tilde: Max dimensionless depth for network scaling
            t_ref_days: Fixed reference time in days (default: 15 days)
            ic_profile: Optional dict with 'depths' and 'theta' for measured IC profile
                       {'depths': [0.02, 0.15, 0.30, ...], 'theta': [0.3, 0.28, ...]}
                       Depths in meters (positive, below surface), theta in m³/m³
            ic_type: Type of initial condition:
                    'obs' - Use measured profile with linear extrapolation to water table
                    'linear' - Linear from surface h_obs to water table h=0
                    'hydrostatic' - Hydrostatic profile (h = -zb - z)
            device: Device for computation
        """
        super().__init__()

        # Validate moisture BC data
        if theta0_data is None:
            raise ValueError("theta0_data is required for moisture BC")

        # Store normalizer
        self.normalizer = normalizer

        # Compute fixed reference time in dimensionless units
        t_ref_dim = t_ref_days * 86400  # Convert days to seconds
        t_ref_tilde = t_ref_dim / normalizer.T

        # Store scaling parameters (needed for fine-tuning)
        self.t_ref_tilde = t_ref_tilde
        self.z_max_tilde = z_max_tilde

        # Initialize networks with FIXED reference time (duration-independent)
        self.h_net = PressureHeadNet(
            h_net_config["hidden_dim"],
            h_net_config["num_layers"],
            t_ref_tilde=t_ref_tilde,
            z_max_tilde=z_max_tilde
        )
        self.zb_net = WaterTableNet(
            zb_net_config["hidden_dim"],
            zb_net_config["num_layers"],
            t_ref_tilde=t_ref_tilde
        )

        # Store dimensionless parameters
        self.Sy_tilde = normalizer.normalize_Sy(Sy)
        self.zr_tilde = normalizer.normalize_z(zr)
        self.zb_initial_tilde = normalizer.normalize_zb(zb_initial)

        # Store and normalize moisture BC data
        self.theta0_times_dim = torch.tensor(theta0_data[0], dtype=torch.float32, device=device)
        self.theta0_values_dim = torch.tensor(theta0_data[1], dtype=torch.float32, device=device)

        # Normalize moisture: θ̃ = (θ - θ_r) / θ_* = S_e
        # θ_* = θ_s - θ_r (from normalizer)
        self.theta0_values_tilde = (self.theta0_values_dim - normalizer.theta_r) / normalizer.theta_star
        self.theta0_times_tilde = normalizer.normalize_t(self.theta0_times_dim)

        # Store S_max_tilde from normalizer
        self.S_max_tilde = normalizer.S_max_tilde

        # Store IC type
        self.ic_type = ic_type

        # Store measured IC profile if provided
        self.ic_profile = None
        if ic_profile is not None:
            # Convert depths to z-coordinates (negative, z=0 at surface)
            depths = torch.tensor(ic_profile['depths'], dtype=torch.float32, device=device)
            z_ic = -depths  # Convert positive depth to negative z
            theta_ic = torch.tensor(ic_profile['theta'], dtype=torch.float32, device=device)

            # Convert theta to effective saturation
            Se_ic = (theta_ic - normalizer.theta_r) / normalizer.theta_star
            Se_ic = torch.clamp(Se_ic, 1e-6, 1.0 - 1e-6)

            # Convert Se to pressure head using van Genuchten inversion
            m = 1.0 - 1.0 / normalizer.n
            inv_term = torch.pow(Se_ic, -1.0 / m) - 1.0
            h_ic = -torch.pow(torch.clamp(inv_term, min=0.0), 1.0 / normalizer.n) / normalizer.alpha

            self.ic_profile = {
                'z': z_ic,  # Dimensional z-coordinates [m]
                'h': h_ic   # Dimensional pressure head [m]
            }

    def forward(self, z_tilde, t_tilde):
        """
        Internal method - works with dimensionless quantities
        
        Args:
            z_tilde: Dimensionless spatial coordinate
            t_tilde: Dimensionless time
        
        Returns:
            h_tilde: Dimensionless pressure head
            zb_tilde: Dimensionless water table depth
        """
        h_tilde = self.h_net(z_tilde, t_tilde)
        zb_tilde = self.zb_net(t_tilde)
        return h_tilde, zb_tilde


    def surface_moisture_tilde(self, t_tilde):
        """Prescribed dimensionless surface moisture θ̃0(t̃) = S_e(t̃) - interpolated from input data (GPU-optimized)"""
        t_flat = t_tilde.flatten()

        # Vectorized searchsorted for all points at once (GPU-efficient)
        indices = torch.searchsorted(self.theta0_times_tilde, t_flat)
        indices = torch.clamp(indices, 1, len(self.theta0_times_tilde) - 1)

        # Get surrounding time and moisture values (vectorized)
        t1 = self.theta0_times_tilde[indices - 1]
        t2 = self.theta0_times_tilde[indices]
        theta1 = self.theta0_values_tilde[indices - 1]
        theta2 = self.theta0_values_tilde[indices]

        # Vectorized linear interpolation
        alpha = (t_flat - t1) / (t2 - t1 + 1e-12)
        theta0_tilde_interp = theta1 + alpha * (theta2 - theta1)

        return theta0_tilde_interp.reshape_as(t_tilde)

    def root_uptake_tilde(self, z_tilde, t_tilde):
        """Dimensionless root uptake S̃(z̃,t̃) in the root zone"""
        return torch.where(
            z_tilde >= -self.zr_tilde, 
            torch.full_like(z_tilde, self.S_max_tilde), 
            torch.zeros_like(z_tilde)
        )

    def compute_derivatives_tilde(self, h_tilde, z_tilde, t_tilde):
        """Compute dimensionless derivatives"""
        # Effective saturation
        Se = self.normalizer.Se_tilde(h_tilde)
        
        # ∂S_e/∂t̃
        dSe_dt_tilde = torch.autograd.grad(Se.sum(), t_tilde, create_graph=True)[0]
        
        # ∂h̃/∂z̃
        dh_dz_tilde = torch.autograd.grad(h_tilde.sum(), z_tilde, create_graph=True)[0]
        
        # Dimensionless conductivity
        K_tilde = self.normalizer.K_tilde(h_tilde)
        
        # Dimensionless flux: q̃ = -K̃(∂h̃/∂z̃ + 1)
        q_tilde = -K_tilde * (dh_dz_tilde + 1.0)
        
        # ∂q̃/∂z̃
        dq_dz_tilde = torch.autograd.grad(q_tilde.sum(), z_tilde, create_graph=True)[0]
        
        return dSe_dt_tilde, dq_dz_tilde, q_tilde, K_tilde

    # ==================== PUBLIC API - ACCEPTS DIMENSIONAL INPUTS ====================
    
    def pde_residual(self, z, t):
        """
        Dimensionless Richards equation residual
        
        Args:
            z: DIMENSIONAL spatial coordinate [m]
            t: DIMENSIONAL time [s]
        
        Returns:
            Dimensionless PDE residual (O(1))
        """
        # Normalize inputs
        z_tilde = self.normalizer.normalize_z(z).requires_grad_(True)
        t_tilde = self.normalizer.normalize_t(t).requires_grad_(True)
        
        # Compute in dimensionless space
        h_tilde, _ = self(z_tilde, t_tilde)
        dSe_dt_tilde, dq_dz_tilde, _, _ = self.compute_derivatives_tilde(h_tilde, z_tilde, t_tilde)
        S_tilde = self.root_uptake_tilde(z_tilde, t_tilde)
        
        # Dimensionless PDE: ∂S_e/∂t̃ + ∂q̃/∂z̃ + S̃ = 0
        return dSe_dt_tilde + dq_dz_tilde + S_tilde


    def surface_moisture_bc_residual(self, t):
        """
        Dimensionless soil moisture BC at surface

        Args:
            t: DIMENSIONAL time [s]

        Returns:
            Dimensionless BC residual (O(1))
        """
        # Normalize input
        t_tilde = self.normalizer.normalize_t(t).requires_grad_(True)
        z0_tilde = torch.zeros_like(t_tilde, requires_grad=True, device=t_tilde.device)

        # Compute in dimensionless space
        h0_tilde, _ = self(z0_tilde, t_tilde)

        # Compute effective saturation (= dimensionless moisture) from predicted head
        Se_pred = self.normalizer.Se_tilde(h0_tilde)

        # Get observed dimensionless moisture
        theta_obs_tilde = self.surface_moisture_tilde(t_tilde)

        # Residual: predicted moisture - observed moisture
        return Se_pred - theta_obs_tilde

    def water_table_head_residual(self, t):
        """
        Dimensionless head BC at water table
        
        Args:
            t: DIMENSIONAL time [s]
        
        Returns:
            Dimensionless BC residual (O(1))
        """
        # Normalize input
        t_tilde = self.normalizer.normalize_t(t)
        z_dummy = torch.zeros_like(t_tilde)
        
        # Compute in dimensionless space
        _, zb_tilde = self(z_dummy, t_tilde)
        h_wt_tilde, _ = self(-zb_tilde, t_tilde)
        
        return h_wt_tilde

    def water_table_kinematic_residual(self, t):
        """
        Dimensionless water table kinematic condition
        
        Args:
            t: DIMENSIONAL time [s]
        
        Returns:
            Dimensionless kinematic residual (O(1))
        """
        # Normalize input
        t_tilde = self.normalizer.normalize_t(t).requires_grad_(True)
        
        # Compute in dimensionless space
        _, zb_tilde = self(torch.zeros_like(t_tilde), t_tilde)
        dzb_dt_tilde = torch.autograd.grad(zb_tilde.sum(), t_tilde, create_graph=True)[0]
        z_wt_tilde = (-zb_tilde).requires_grad_(True)
        h_wt_tilde, _ = self(z_wt_tilde, t_tilde)
        K_wt_tilde = self.normalizer.K_tilde(h_wt_tilde)
        dh_dz_wt_tilde = torch.autograd.grad(h_wt_tilde.sum(), z_wt_tilde, create_graph=True)[0]
        q_wt_tilde = -K_wt_tilde * (dh_dz_wt_tilde + 1.0)
        
        return dzb_dt_tilde - q_wt_tilde / self.Sy_tilde

    def _torch_interp_1d(self, x, xp, fp):
        """
        GPU-optimized 1D linear interpolation (replacement for np.interp)

        Args:
            x: Query points (1D tensor)
            xp: Data point x-coordinates (1D tensor, must be sorted)
            fp: Data point y-coordinates (1D tensor, same length as xp)

        Returns:
            Interpolated values at query points x
        """
        # Handle edge cases: extrapolate with boundary values (like np.interp)
        x_clamped = torch.clamp(x, xp[0], xp[-1])

        # Use searchsorted for GPU-accelerated binary search
        indices = torch.searchsorted(xp, x_clamped)
        indices = torch.clamp(indices, 1, len(xp) - 1)

        # Get surrounding points
        x0 = xp[indices - 1]
        x1 = xp[indices]
        y0 = fp[indices - 1]
        y1 = fp[indices]

        # Linear interpolation: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        alpha = (x_clamped - x0) / (x1 - x0 + 1e-12)
        y_interp = y0 + alpha * (y1 - y0)

        return y_interp

    def initial_conditions_residual(self, z, t0):
        """
        Dimensionless initial conditions - uses measured profile if provided, else hydrostatic
        GPU-OPTIMIZED: All operations stay on GPU (no NumPy/CPU transfers)

        Args:
            z: DIMENSIONAL spatial coordinate [m]
            t0: DIMENSIONAL initial time [s]

        Returns:
            Tuple of (h residual, zb residual), both dimensionless (O(1))
        """
        # Normalize inputs
        z_tilde = self.normalizer.normalize_z(z)
        t0_tilde = self.normalizer.normalize_t(t0)

        # Compute in dimensionless space
        h_tilde, zb_tilde = self(z_tilde, t0_tilde)

        # Determine IC based on ic_type
        if self.ic_type == 'obs' and self.ic_profile is not None:
            # Option 1: Use measured profile with linear extrapolation to water table
            # ✅ GPU-OPTIMIZED: All operations stay on GPU
            z_measured = self.ic_profile['z']  # Already on GPU
            h_measured = self.ic_profile['h']  # Already on GPU

            # Get initial water table depth (dimensional) - stay on GPU
            zb_ic_dim = self.normalizer.denormalize_zb(
                torch.tensor(self.zb_initial_tilde, device=z.device)
            )

            # Extend measured profile to water table
            # Water table point: h(z=-zb) = 0
            z_wt = -zb_ic_dim.view(-1)
            h_wt = torch.zeros_like(z_wt)

            # Extended profile: measurements + water table point (GPU tensors)
            z_extended = torch.cat([z_measured, z_wt])
            h_extended = torch.cat([h_measured, h_wt])

            # Sort by z (most negative to least negative) - GPU operation
            sort_idx = torch.argsort(z_extended)
            z_extended_sorted = z_extended[sort_idx]
            h_extended_sorted = h_extended[sort_idx]

            # ✅ GPU-native linear interpolation (replacement for np.interp)
            z_flat = z.flatten()
            h_ic_interp = self._torch_interp_1d(z_flat, z_extended_sorted, h_extended_sorted)
            h_ic = h_ic_interp.view_as(z)

            # Normalize to dimensionless
            h_ic_tilde = self.normalizer.normalize_h(h_ic)

        elif self.ic_type == 'linear' and self.ic_profile is not None:
            # Option 2: Linear from surface h_obs to water table h=0
            # ✅ GPU-OPTIMIZED: All operations stay on GPU

            # Get surface h value (shallowest measurement, z closest to 0)
            h_surface = self.ic_profile['h'][self.ic_profile['z'].argmax()]

            # Get initial water table depth (dimensional) - stay on GPU
            zb_ic_dim = self.normalizer.denormalize_zb(
                torch.tensor(self.zb_initial_tilde, device=z.device)
            )

            # Linear profile: h(z) = h_surface * (1 + z/zb_ic)
            # At z=0: h = h_surface
            # At z=-zb_ic: h = 0
            h_ic = h_surface * (1.0 + z / zb_ic_dim)

            # Normalize to dimensionless
            h_ic_tilde = self.normalizer.normalize_h(h_ic)

        else:
            # Hydrostatic initial condition (dimensionless)
            h_ic_tilde = (-zb_tilde - z_tilde)

        # Initial water table depth (dimensionless)
        zb_ic_tilde = torch.tensor(self.zb_initial_tilde, device=z_tilde.device).expand_as(zb_tilde)

        return (h_tilde - h_ic_tilde), (zb_tilde - zb_ic_tilde)

    def predict_water_table(self, t):
        """
        Predict dimensional water table depth from dimensional time

        Args:
            t: dimensional time [s]

        Returns:
            zb: dimensional water table depth [m]
        """
        t_tilde = self.normalizer.normalize_t(t)
        zb_tilde = self.zb_net(t_tilde)
        zb = self.normalizer.denormalize_zb(zb_tilde)
        return zb

    def predict_head(self, z, t):
        """
        Predict dimensional pressure head from dimensional inputs
        
        Args:
            z: dimensional spatial coordinate [m]
            t: dimensional time [s]
        
        Returns:
            h: dimensional pressure head [m]
            zb: dimensional water table depth [m]
        """
        z_tilde = self.normalizer.normalize_z(z)
        t_tilde = self.normalizer.normalize_t(t)
        
        with torch.no_grad():
            h_tilde, zb_tilde = self(z_tilde, t_tilde)
        
        h = self.normalizer.denormalize_h(h_tilde)
        zb = self.normalizer.denormalize_zb(zb_tilde)
        
        return h, zb

    def predict_conductivity(self, h):
        """
        Predict dimensional conductivity from dimensional head
        
        Args:
            h: dimensional pressure head [m]
        
        Returns:
            K: dimensional conductivity [m/s]
        """
        h_tilde = self.normalizer.normalize_h(h)
        K_tilde = self.normalizer.K_tilde(h_tilde)
        K = K_tilde * self.normalizer.K_star  # Denormalize
        
        return K
