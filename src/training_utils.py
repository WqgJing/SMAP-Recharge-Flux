import torch
import math
import numpy as np


class SamplingHelpers:
    """Helper functions for sampling training points."""

    @staticmethod
    def sample_initial_condition_points(model, batch_size, t_min, z_max, device):
        """Sample initial condition points."""
        ic_batch_size = min(batch_size // 4, 50)
        t_ic = torch.full((ic_batch_size, 1), t_min, device=device)

        with torch.no_grad():
            zb0 = model.predict_water_table(t_ic)

        u_ic = torch.rand_like(t_ic)
        # IC z samples in [-zb(t_min), 0]
        z_ic = -zb0 * (1.0 - u_ic) + z_max * u_ic

        return z_ic, t_ic

    @staticmethod
    def sample_pde_points_direct(model, batch_size, t_max, device, boundary_ratio=0.7):
        """
        Generate fresh random samples each epoch (Option 1: Direct Random Sampling).
        No cache, no residuals - just sample and go.

        Args:
            model: PINN model for predicting water table depth
            batch_size: Number of collocation points to sample
            t_max: Maximum time value
            device: torch device
            boundary_ratio: Fraction of points near boundaries (default: 0.7)

        Returns:
            z_col: Physical depth coordinates (z ∈ [-zb(t), 0])
            t_col: Time coordinates
        """
        # Sample time uniformly
        t_col = torch.rand(batch_size, 1, device=device) * t_max
        t_col.requires_grad_(True)

        # Sample normalized depth u ∈ [0,1]
        n_boundary = int(batch_size * boundary_ratio)
        n_interior = batch_size - n_boundary

        # Boundary points: concentrate near u=0 (surface) and u=1 (water table)
        n_surface = n_boundary // 2
        n_bottom = n_boundary - n_surface

        # Sample directly on GPU (avoid CPU→GPU transfer)
        u_surface = torch.distributions.Beta(
            torch.tensor(1.0, device=device),
            torch.tensor(3.0, device=device)
        ).sample((n_surface,))
        u_bottom = torch.distributions.Beta(
            torch.tensor(3.0, device=device),
            torch.tensor(1.0, device=device)
        ).sample((n_bottom,))
        u_interior = torch.rand(n_interior, device=device)

        u_col = torch.cat([u_surface, u_bottom, u_interior]).reshape(-1, 1)

        # Map to physical depth using predicted water table
        with torch.no_grad():
            zb_col = model.predict_water_table(t_col)

        z_col = (-u_col * zb_col).requires_grad_(True)

        return z_col, t_col


def compute_losses(model, z_col, t_col, t_bc, z_ic, t_ic):
    """Compute all loss components with optional importance weighting."""
    # PDE residual on interior points
    res_pde = model.pde_residual(z_col, t_col)
    loss_pde = (res_pde**2).mean()

    # Surface BC at z=0 - moisture BC (Dirichlet)
    res_surf = model.surface_moisture_bc_residual(t_bc)
    loss_surf = (res_surf**2).mean()

    # Water-table BCs
    res_wt_head = model.water_table_head_residual(t_bc)
    loss_wt_head = (res_wt_head**2).mean()

    res_wt_kin = model.water_table_kinematic_residual(t_bc)
    loss_wt_kin = (res_wt_kin**2).mean()

    # Initial conditions
    res_ic_h, res_ic_zb = model.initial_conditions_residual(z_ic, t_ic)
    loss_ic_h = (res_ic_h**2).mean()
    loss_ic_zb = (res_ic_zb**2).mean()

    return {
        "pde": loss_pde,
        "surf": loss_surf,
        "wt_head": loss_wt_head,
        "wt_kin": loss_wt_kin,
        "ic_h": loss_ic_h,
        "ic_zb": loss_ic_zb,
    }


def apply_weights_and_compute_gradients(losses, weights, model):
    """
    Apply weights to losses and compute gradients.
    GPU-OPTIMIZED: Gradient norms stay on GPU, sync deferred to weight update.
    """
    # Apply weights
    weighted_losses = {key: weights[key] * losses[key] for key in losses}

    # ✅ GPU-OPTIMIZED: Compute gradients (returns GPU tensors, no sync)
    gradients = {}
    for key in losses:
        gradients[key] = compute_grad_norm(weighted_losses[key], model)

    # Compute total weighted loss
    total_loss = sum(weighted_losses.values())

    return weighted_losses, gradients, total_loss


def apply_weights_fixed_mode(losses, weights):
    """Apply weights to losses without computing gradients (for fixed weight mode)."""
    # Apply weights
    weighted_losses = {key: weights[key] * losses[key] for key in losses}
    
    # Compute total weighted loss
    total_loss = sum(weighted_losses.values())
    
    # Return dummy gradients (will not be used)
    gradients = {key: 0.0 for key in losses}
    gradients["total"] = 0.0
    
    return weighted_losses, gradients, total_loss


class WeightManager:
    """Adaptive weights for PINN loss components with EMA + log-space updates."""

    def __init__(
        self,
        use_initial_scales: bool,
        weight_lr: float = 0.2,
        ema_alpha: float = 0.9,
        min_w: float = 1e-10,
        max_w: float = 1e10,
        max_step_factor: float = 2.0,  # per-update ×/÷ cap
        eps: float = 1e-12,
        # Custom initial weights
        initial_pde_weight: float = None,
        initial_surf_weight: float = None,
        initial_wt_head_weight: float = None,
        initial_wt_kin_weight: float = None,
        initial_ic_h_weight: float = None,
        initial_ic_zb_weight: float = None,
        # Fixed weights mode
        use_fixed_weights: bool = False,
    ):
        self.weight_lr = float(weight_lr)
        self.ema_alpha = float(ema_alpha)
        self.min_w, self.max_w = float(min_w), float(max_w)
        self.max_step_factor = float(max_step_factor)
        self.eps = float(eps)
        self.use_fixed_weights = use_fixed_weights

        # Use custom weights if provided, otherwise use default logic
        if initial_pde_weight is not None:
            base = {
                "pde": initial_pde_weight,
                "surf": initial_surf_weight,
                "wt_head": initial_wt_head_weight,
                "wt_kin": initial_wt_kin_weight,
                "ic_h": initial_ic_h_weight,
                "ic_zb": initial_ic_zb_weight,
            }
        else:
            # Default weights for moisture BC (Dirichlet)
            # Moisture BC is more stable than flux BC, use lower weight (20)
            default_surf_weight = 20

            base = {
                "pde": 1 if use_initial_scales else 1.0,
                "surf": default_surf_weight if use_initial_scales else 1.0,
                "wt_head": 1,
                "wt_kin": 1,
                "ic_h": 1,
                "ic_zb": 1,
            }
        self.weights = {k: float(v) for k, v in base.items()}
        self.weight_history = {k: [self.weights[k]] for k in self.weights}
        self.grad_ema = {k: None for k in self.weights}

    def _to_float(self, x):
        """
        Convert to float, handling both CPU and GPU tensors.
        GPU-OPTIMIZED: Batch sync deferred - only converts when absolutely necessary.
        """
        if isinstance(x, torch.Tensor):
            # ✅ GPU-OPTIMIZED: Defer sync by returning tensor directly
            # Conversion to scalar happens only when needed for weight computation
            return x.detach()
        try:
            # works for python float, numpy scalar
            return float(x)
        except Exception:
            # last resort
            return np.float64(x).item()

    def update(self, current_grads, counts=None, already_weighted: bool = True):
        """
        current_grads: dict term-> grad L2 norm (weighted if already_weighted=True)
        counts: optional dict term-> number of samples contributing to that term
        already_weighted: if True, unweight using current self.weights
        GPU-OPTIMIZED: Single batched sync at end instead of per-gradient syncs.
        """
        # Skip update if using fixed weights
        if self.use_fixed_weights:
            return

        # 1) Build unweighted, per-sample grad estimates (keep on GPU)
        g_unw_tensors = {}
        for k in self.weights:
            if k not in current_grads:
                continue
            g = self._to_float(current_grads[k])  # Returns tensor if input is tensor

            # Handle tensor operations
            if isinstance(g, torch.Tensor):
                if already_weighted:
                    g = g / max(self.weights[k], self.eps)  # unweight
                if counts is not None and k in counts and counts[k] and counts[k] > 0:
                    g = g / math.sqrt(float(counts[k]))  # per-sample
                g_unw_tensors[k] = torch.clamp(g, min=0.0)
            else:
                if already_weighted:
                    g = g / max(self.weights[k], self.eps)  # unweight
                if counts is not None and k in counts and counts[k] and counts[k] > 0:
                    g = g / math.sqrt(float(counts[k]))  # per-sample
                g_unw_tensors[k] = max(g, 0.0)

        # ✅ GPU-OPTIMIZED: Single batched CPU sync for all gradients
        g_unw = {}
        for k, v in g_unw_tensors.items():
            if isinstance(v, torch.Tensor):
                g_unw[k] = v.cpu().item()  # Single sync per key
            else:
                g_unw[k] = v

        # 2) EMA smoothing
        for k in self.weights:
            if k in g_unw:
                if self.grad_ema[k] is None:
                    self.grad_ema[k] = g_unw[k]
                else:
                    self.grad_ema[k] = (
                        self.ema_alpha * self.grad_ema[k]
                        + (1.0 - self.ema_alpha) * g_unw[k]
                    )

        # 3) Robust target = median of valid EMA grads
        valid = [
            v
            for v in (self.grad_ema[k] for k in self.weights)
            if v is not None and v > self.eps
        ]
        if not valid:
            return  # nothing to do yet
        g_target = float(np.median(valid))

        # 4) Log-space, capped multiplicative update
        log_cap = math.log(self.max_step_factor)
        for k in self.weights:
            v = self.grad_ema[k]
            if v is None or v <= self.eps:
                continue
            ratio = (
                v / g_target
            )  # >1 means this term is too large, so we reduce its weight
            log_w = math.log(self.weights[k])
            # move opposite to log(ratio); minus sign equalizes grads
            log_w_new = log_w - self.weight_lr * math.log(ratio)
            # cap per-update jump
            log_w_new = max(log_w - log_cap, min(log_w + log_cap, log_w_new))
            w_new = math.exp(log_w_new)
            # clamp absolute bounds
            self.weights[k] = float(min(self.max_w, max(self.min_w, w_new)))

    def is_using_fixed_weights(self):
        """Check if using fixed weights mode."""
        return self.use_fixed_weights

    def record_history(self):
        for k in self.weights:
            self.weight_history[k].append(self.weights[k])

    def get_weights(self):
        return dict(self.weights)

    def print_update(self, epoch):
        print(f"\n[Epoch {epoch+1}] Updated weights:")
        for k in self.weights:
            g = self.grad_ema[k]
            gtxt = f"{g:.3e}" if (g is not None) else "nan"
            print(f"  {k}: {self.weights[k]:.3e} (grad_unw_ema: {gtxt})")


# Gradient computation utilities
def compute_grad_norm(loss, model):
    """
    Compute the L2 norm of gradients for a specific loss term.
    GPU-OPTIMIZED: Returns GPU tensor, defers synchronization to caller.
    """
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    # ✅ GPU-OPTIMIZED: Keep everything on GPU, no .item() calls
    # Collect non-None gradients and compute norms on GPU
    grad_norms_squared = []
    for grad in grads:
        if grad is not None:
            grad_norms_squared.append(grad.norm(2) ** 2)

    if len(grad_norms_squared) == 0:
        # No gradients - return scalar 0 on same device as model
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    # ✅ Single GPU operation to sum all squared norms
    total_norm_squared = torch.stack(grad_norms_squared).sum()

    # ✅ Return GPU tensor (caller decides when to sync with .item())
    return total_norm_squared.sqrt()


def compute_total_grad_norm(model):
    """
    Compute total gradient norm after backward pass.
    GPU-OPTIMIZED: Returns GPU tensor, defers synchronization to caller.
    """
    # ✅ GPU-OPTIMIZED: Keep everything on GPU, no .item() calls
    grad_norms_squared = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms_squared.append(param.grad.norm(2) ** 2)

    if len(grad_norms_squared) == 0:
        # No gradients - return scalar 0 on same device as model
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    # ✅ Single GPU operation to sum all squared norms
    total_norm_squared = torch.stack(grad_norms_squared).sum()

    # ✅ Return GPU tensor (caller decides when to sync with .item())
    return total_norm_squared.sqrt()


def compute_full_sample_loss(model, q0_times_t, t_min, z_max, device, t_max,
                              batch_size_bc=100, chunk_size=1000,
                              boundary_ratio=0.7, sample_size=5000):
    """
    Compute loss over all samples (entire dataset) for smooth loss tracking.
    GPU-OPTIMIZED: Accumulates on GPU, single sync at end.

    Args:
        model: The PINN model
        q0_times_t: Boundary condition time points
        t_min: Minimum time for initial conditions
        z_max: Maximum z (surface) for initial conditions
        device: Device to use
        t_max: Maximum time
        batch_size_bc: Batch size for boundary condition points
        chunk_size: Chunk size for processing large datasets
        boundary_ratio: Boundary sampling ratio
        sample_size: Number of samples to generate

    Returns:
        Dictionary containing total loss and loss components computed over full dataset
    """
    model.eval()

    # ✅ GPU-OPTIMIZED: Initialize accumulators as GPU tensors
    loss_accum_gpu = {
        "pde": torch.tensor(0.0, device=device),
        "surf": torch.tensor(0.0, device=device),
        "wt_head": torch.tensor(0.0, device=device),
        "wt_kin": torch.tensor(0.0, device=device),
        "ic_h": torch.tensor(0.0, device=device),
        "ic_zb": torch.tensor(0.0, device=device),
    }

    # 1. Compute PDE loss over sampled points using direct sampling
    n_pde_points = 0

    for i in range(0, sample_size, chunk_size):
        chunk_batch_size = min(chunk_size, sample_size - i)

        # Sample time uniformly
        t_chunk = torch.rand(chunk_batch_size, 1, device=device) * t_max

        # Sample normalized depth u ∈ [0,1]
        n_boundary = int(chunk_batch_size * boundary_ratio)
        n_interior = chunk_batch_size - n_boundary
        n_surface = n_boundary // 2
        n_bottom = n_boundary - n_surface

        # Sample directly on GPU (avoid CPU→GPU transfer)
        u_surface = torch.distributions.Beta(
            torch.tensor(1.0, device=device),
            torch.tensor(3.0, device=device)
        ).sample((n_surface,))
        u_bottom = torch.distributions.Beta(
            torch.tensor(3.0, device=device),
            torch.tensor(1.0, device=device)
        ).sample((n_bottom,))
        u_interior = torch.rand(n_interior, device=device)
        u_chunk = torch.cat([u_surface, u_bottom, u_interior]).reshape(-1, 1)

        # Map u to z using predicted water table depth
        zb_chunk = model.predict_water_table(t_chunk)

        # Create z_chunk and enable gradients for PDE residual computation
        z_chunk = (-u_chunk * zb_chunk).detach().requires_grad_(True)
        t_chunk_grad = t_chunk.detach().requires_grad_(True)

        # Compute PDE residual (needs gradients enabled for physics derivatives)
        res_pde = model.pde_residual(z_chunk, t_chunk_grad)

        # ✅ GPU-OPTIMIZED: Accumulate on GPU (no .item() call)
        loss_accum_gpu["pde"] += (res_pde**2).sum().detach()
        n_pde_points += len(z_chunk)

    loss_accum_gpu["pde"] /= n_pde_points

    # 2. Compute boundary condition losses over all q0 time points
    n_bc_points = 0
    for i in range(0, len(q0_times_t), batch_size_bc):
        j = min(i + batch_size_bc, len(q0_times_t))
        t_bc = q0_times_t[i:j]

        # BC residuals need gradients for physics derivatives
        # Moisture BC (Dirichlet)
        res_surf = model.surface_moisture_bc_residual(t_bc)
        loss_accum_gpu["surf"] += (res_surf**2).sum().detach()

        res_wt_head = model.water_table_head_residual(t_bc)
        loss_accum_gpu["wt_head"] += (res_wt_head**2).sum().detach()

        res_wt_kin = model.water_table_kinematic_residual(t_bc)
        loss_accum_gpu["wt_kin"] += (res_wt_kin**2).sum().detach()

        n_bc_points += len(t_bc)

    loss_accum_gpu["surf"] /= n_bc_points
    loss_accum_gpu["wt_head"] /= n_bc_points
    loss_accum_gpu["wt_kin"] /= n_bc_points

    # 3. Compute initial condition losses over a representative set of points
    ic_batch_size = 200  # Use a reasonable number of IC points
    sampling_helper = SamplingHelpers()
    z_ic, t_ic = sampling_helper.sample_initial_condition_points(
        model, ic_batch_size, t_min, z_max, device
    )

    # IC residuals computation
    res_ic_h, res_ic_zb = model.initial_conditions_residual(z_ic, t_ic)
    loss_accum_gpu["ic_h"] = (res_ic_h**2).mean().detach()
    loss_accum_gpu["ic_zb"] = (res_ic_zb**2).mean().detach()

    model.train()

    # ✅ GPU-OPTIMIZED: Single batched CPU sync at the very end
    loss_accum = {k: v.cpu().item() for k, v in loss_accum_gpu.items()}

    return loss_accum
