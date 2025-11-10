import torch
import numpy as np


class NormalizationHelper:
    """
    Handles all dimensional ↔ dimensionless conversions for Richards equation PINN.
    Based on the corrected normalization recipe with T = θ_* L / K_*
    """
    
    def __init__(self, soil_params, L, S_max=0.0):
        """
        Args:
            soil_params: dict with keys 'theta_s', 'theta_r', 'alpha', 'n', 'Ks', 'l'
            L: characteristic length scale (domain depth) [m]
            S_max: maximum sink term [1/s]
        """
        # Store original soil parameters (ensure float conversion)
        self.theta_s = float(soil_params['theta_s'])
        self.theta_r = float(soil_params['theta_r'])
        self.alpha = float(soil_params['alpha'])  # [1/m]
        self.n = float(soil_params['n'])
        self.m = 1.0 - 1.0 / self.n
        self.Ks = float(soil_params['Ks'])  # [m/s]
        self.l = float(soil_params['l'])

        # Characteristic scales
        self.L = float(L)  # Length scale [m]
        self.H_star = L  # Head scale (H_* = L) [m]
        self.K_star = self.Ks  # Conductivity scale [m/s]
        self.Q_star = self.Ks  # Flux scale [m/s]
        self.theta_star = self.theta_s - self.theta_r  # Water content span [-]
        
        # ✅ CORRECTED TIME SCALE: T = θ_* L / K_*
        self.T = self.theta_star * self.L / self.K_star  # [s]
        
        # Dimensionless parameters
        self.alpha_tilde = self.alpha * self.H_star  # α̃ = α × L [-]
        self.S_max_tilde = S_max * self.L / self.K_star if S_max > 0 else 0.0  # S̃_max [-]
        
        # Small constant for numerical stability
        self._tiny = 1e-12
        
        # Print normalization info
        self._print_scales()
    
    def _print_scales(self):
        """Print characteristic scales for verification"""
        print("\n" + "="*60)
        print("NORMALIZATION SCALES")
        print("="*60)
        print(f"Length scale (L):           {self.L:.3f} m")
        print(f"Head scale (H_*):           {self.H_star:.3f} m")
        print(f"Conductivity scale (K_*):   {self.K_star:.2e} m/s")
        print(f"Flux scale (Q_*):           {self.Q_star:.2e} m/s")
        print(f"Water content span (θ_*):   {self.theta_star:.3f}")
        print(f"Time scale (T):             {self.T:.2e} s ({self.T/3600:.2f} hours)")
        print(f"Dimensionless α̃:            {self.alpha_tilde:.4f}")
        print(f"Dimensionless S̃_max:        {self.S_max_tilde:.4f}")
        print("="*60 + "\n")
    
    # ==================== NORMALIZATION METHODS ====================
    
    def normalize_z(self, z):
        """z̃ = z / L"""
        return z / self.L
    
    def normalize_t(self, t):
        """t̃ = t / T"""
        return t / self.T
    
    def normalize_h(self, h):
        """h̃ = h / L"""
        return h / self.L
    
    def normalize_q(self, q):
        """q̃ = q / K_*"""
        return q / self.Q_star
    
    def normalize_S(self, S):
        """S̃ = S × L / K_*"""
        return S * self.L / self.K_star
    
    def normalize_zb(self, zb):
        """z̃_b = z_b / L"""
        return zb / self.L
    
    def normalize_Sy(self, Sy):
        """S̃_y = S_y / θ_*"""
        return Sy / self.theta_star
    
    # ==================== DENORMALIZATION METHODS ====================
    
    def denormalize_z(self, z_tilde):
        """z = z̃ × L"""
        return z_tilde * self.L
    
    def denormalize_t(self, t_tilde):
        """t = t̃ × T"""
        return t_tilde * self.T
    
    def denormalize_h(self, h_tilde):
        """h = h̃ × L"""
        return h_tilde * self.L
    
    def denormalize_q(self, q_tilde):
        """q = q̃ × K_*"""
        return q_tilde * self.Q_star
    
    def denormalize_zb(self, zb_tilde):
        """z_b = z̃_b × L"""
        return zb_tilde * self.L
    
    # ==================== DIMENSIONLESS VAN GENUCHTEN FUNCTIONS ====================
    
    def Se_tilde(self, h_tilde):
        """
        Effective saturation from dimensionless head h̃
        S_e(h̃) = [1 + (α̃|h̃|)^n]^(-m)
        """
        abs_h_tilde = torch.abs(h_tilde)
        denom = (1.0 + (self.alpha_tilde * abs_h_tilde).pow(self.n)).pow(self.m)
        denom = denom + self._tiny
        Se = 1.0 / denom

        # Se = 1 when h̃ >= 0 (saturated)
        Se = torch.where(
            h_tilde >= 0.0,
            torch.ones_like(h_tilde),
            Se
        )

        # Clamp to valid range
        return torch.clamp(Se, 1e-6, 1.0 - 1e-6)
    
    def K_tilde(self, h_tilde):
        """
        Dimensionless relative conductivity K̃ = K / K_*
        K̃(h̃) = k_r(S_e) using Mualem model
        """
        Se = self.Se_tilde(h_tilde)

        # Mualem model: k_r = Se^l × [1 - (1 - Se^(1/m))^m]^2
        Se_1m = Se.pow(1.0 / self.m)
        kr = (Se.pow(self.l)) * (1.0 - (1.0 - Se_1m).pow(self.m)).pow(2)

        # K̃ = 1 when h̃ >= 0 (saturated)
        K_tilde = torch.where(
            h_tilde >= 0.0,
            torch.ones_like(h_tilde),
            torch.clamp(kr, 0.0, 1.0)
        )

        return K_tilde
    
    def C_tilde(self, h_tilde):
        """
        Dimensionless capacity C̃ = dS_e/dh̃
        Useful for head-based form of Richards equation
        """
        h_tilde_req = h_tilde.requires_grad_(True)
        Se = self.Se_tilde(h_tilde_req)
        C = torch.autograd.grad(Se.sum(), h_tilde_req, create_graph=True)[0]
        return C
    
    # ==================== UTILITY METHODS ====================
    
    def get_scales_dict(self):
        """Return dictionary of all scales for logging/plotting"""
        return {
            'L': self.L,
            'H_star': self.H_star,
            'K_star': self.K_star,
            'Q_star': self.Q_star,
            'theta_star': self.theta_star,
            'T': self.T,
            'alpha_tilde': self.alpha_tilde,
            'S_max_tilde': self.S_max_tilde
        }
