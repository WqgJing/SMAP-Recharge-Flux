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
        et_data=None,
        Sy=0.3,
        zr=0.5,
        h_net_config=None,
        zb_net_config=None,
        normalizer=None,
        zb_initial=1.5,
        t_max=86400,
        z_max_tilde=1.0,
        t_ref_days=15.0,
        device='cpu'
    ):
        """
        Args:
            soil_params: Soil parameters dict (for normalizer)
            theta0_data: Tuple of (times, soil_moisture) in DIMENSIONAL form - REQUIRED
            et_data: Tuple of (times, evaporation_rate) in DIMENSIONAL form [s, 1/s] - OPTIONAL
                     If None, uses S_max from normalizer as constant rate
            Sy: Specific yield (dimensional)
            zr: Root zone depth (dimensional) [m]
            h_net_config: Config dict for pressure head network
            zb_net_config: Config dict for water table network
            normalizer: NormalizationHelper instance
            zb_initial: Initial water table depth (dimensional) [m]
            t_max: Maximum time (dimensional) [s]
            z_max_tilde: Max dimensionless depth for network scaling
            t_ref_days: Fixed reference time in days (default: 15 days)
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

        # Store and normalize ET data (if provided)
        if et_data is not None:
            self.et_times_dim = torch.tensor(et_data[0], dtype=torch.float32, device=device)
            self.et_values_dim = torch.tensor(et_data[1], dtype=torch.float32, device=device)

            # Normalize ET: S̃ = S × L / K_*
            self.et_values_tilde = normalizer.normalize_S(self.et_values_dim)
            self.et_times_tilde = normalizer.normalize_t(self.et_times_dim)
            self.use_et_data = True
        else:
            # Fallback to constant S_max from normalizer
            self.S_max_tilde = normalizer.S_max_tilde
            self.use_et_data = False

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

    def et_rate_tilde(self, t_tilde):
        """Prescribed dimensionless ET rate S̃(t̃) - interpolated from input data (GPU-optimized)"""
        t_flat = t_tilde.flatten()

        # Vectorized searchsorted for all points at once (GPU-efficient)
        indices = torch.searchsorted(self.et_times_tilde, t_flat)
        indices = torch.clamp(indices, 1, len(self.et_times_tilde) - 1)

        # Get surrounding time and ET rate values (vectorized)
        t1 = self.et_times_tilde[indices - 1]
        t2 = self.et_times_tilde[indices]
        et1 = self.et_values_tilde[indices - 1]
        et2 = self.et_values_tilde[indices]

        # Vectorized linear interpolation
        alpha = (t_flat - t1) / (t2 - t1 + 1e-12)
        et_tilde_interp = et1 + alpha * (et2 - et1)

        return et_tilde_interp.reshape_as(t_tilde)

    def root_uptake_tilde(self, z_tilde, t_tilde):
        """
        Dimensionless root uptake S̃(z̃,t̃) = β(z̃) × T̃_p(t̃)

        Where:
          - T̃_p(t̃) is potential transpiration rate [dimensionless]
          - β(z̃) is root density distribution (simple exponential form)

        Exponential distribution:
          β(z̃) = (1/z̃_r) exp(z̃/z̃_r)  for z̃ ∈ [-z̃_r, 0]
          Note: ∫_{-z̃_r}^{0} β(z̃) dz̃ = 1 - e^{-1} ≈ 0.632 (not exactly 1)
        """
        # Get potential transpiration rate T̃_p(t̃)
        if self.use_et_data:
            Tp_tilde = self.et_rate_tilde(t_tilde)
        else:
            Tp_tilde = self.S_max_tilde

        # Root distribution β(z̃) = (1/z̃_r) exp(z̃/z̃_r)
        beta = (1.0 / self.zr_tilde) * torch.exp(z_tilde / self.zr_tilde)

        # Total uptake: S̃(z̃,t̃) = β(z̃) × T̃_p(t̃)
        S_uptake = beta * Tp_tilde

        # Apply only in root zone (z̃ >= -z̃_r), zero below
        return torch.where(
            z_tilde >= -self.zr_tilde,
            S_uptake,
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

    def initial_conditions_residual(self, z, t0):
        """
        Dimensionless IC residual using unified parabolic profile

        The parabolic profile satisfies:
        1. h(0) = h_surface (matches observed surface moisture at t=0)
        2. h(-zb) = 0 (water table boundary condition)
        3. dh/dz|_{z=-zb} = -1 (zero flux at water table → equilibrium)

        Profile: h(z) = a*z² + b*z + c where:
            a = (zb + h_surface) / zb²
            b = (zb + 2*h_surface) / zb
            c = h_surface

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

        # Get h_surface from observed surface moisture at t=0
        Se_surf_0 = self.theta0_values_tilde[0]  # First observation (dimensionless)

        # Invert van Genuchten: Se → h (dimensional)
        m = 1.0 - 1.0 / self.normalizer.n
        inv_term = torch.pow(Se_surf_0, -1.0 / m) - 1.0
        h_surface = -torch.pow(torch.clamp(inv_term, min=0.0), 1.0 / self.normalizer.n) / self.normalizer.alpha

        # Get initial water table depth (dimensional)
        zb_ic_dim = self.normalizer.denormalize_zb(
            torch.tensor(self.zb_initial_tilde, device=z.device)
        )

        # Parabolic profile coefficients: h(z) = a*z² + b*z + c
        a = (zb_ic_dim + h_surface) / (zb_ic_dim * zb_ic_dim)
        b = (zb_ic_dim + 2.0 * h_surface) / zb_ic_dim
        c = h_surface

        # Compute IC profile (dimensional)
        h_ic = a * z * z + b * z + c

        # Normalize to dimensionless
        h_ic_tilde = self.normalizer.normalize_h(h_ic)

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
