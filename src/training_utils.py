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
    """Apply weights to losses and compute gradients."""
    # Apply weights
    weighted_losses = {key: weights[key] * losses[key] for key in losses}

    # Compute gradients for each component
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
                "pde": 0 if use_initial_scales else 1.0,
                "surf": default_surf_weight if use_initial_scales else 1.0,
                "wt_head": 0,
                "wt_kin": 0,
                "ic_h": 0,
                "ic_zb": 0,
            }
        self.weights = {k: float(v) for k, v in base.items()}
        self.weight_history = {k: [self.weights[k]] for k in self.weights}
        self.grad_ema = {k: None for k in self.weights}

    def _to_float(self, x):
        try:
            # works for python float, numpy scalar, or torch tensor on CPU
            return float(x)
        except Exception:
            # last resort
            return np.float64(x).item()

    def update(self, current_grads, counts=None, already_weighted: bool = True):
        """
        current_grads: dict term-> grad L2 norm (weighted if already_weighted=True)
        counts: optional dict term-> number of samples contributing to that term
        already_weighted: if True, unweight using current self.weights
        """
        # Skip update if using fixed weights
        if self.use_fixed_weights:
            return
            
        # 1) Build unweighted, per-sample grad estimates
        g_unw = {}
        for k in self.weights:
            if k not in current_grads:
                continue
            g = self._to_float(current_grads[k])
            if already_weighted:
                g = g / max(self.weights[k], self.eps)  # unweight
            if counts is not None and k in counts and counts[k] and counts[k] > 0:
                g = g / math.sqrt(float(counts[k]))  # per-sample
            g_unw[k] = max(g, 0.0)

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


class CachePoolManager:
    """Manages cache pool for adaptive sampling in PINN training."""

    def __init__(
        self,
        cache_size,
        batch_size,
        device,
        q0_times_t,
        t_max,
        boundary_ratio=0.7,
        high_residual_ratio=0.6,
        temperature=1.0,
    ):
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.device = device
        self.q0_times_t = q0_times_t
        self.t_max = t_max
        self.boundary_ratio = boundary_ratio
        self.high_residual_ratio = high_residual_ratio
        self.temperature = temperature

        # Initialize cache pool
        self.u_cache, self.t_cache = self._generate_cache_pool(cache_size)
        self.cache_residuals = torch.zeros(cache_size, device=device)
        self.sampling_probs = None

        # Cache statistics tracking
        self.cache_stats = {
            "resample_epochs": [],
            "mean_residual": [],
            "max_residual": [],
            "std_residual": [],
        }

    def _generate_cache_pool(self, size):
        """Generate cache pool with mixed temporal and normalized spatial sampling."""
        # Time sampling: half from q0_times, half random uniform
        n_q0_sample = min(size // 2, len(self.q0_times_t))
        q0_indices = torch.randint(
            0, len(self.q0_times_t), (n_q0_sample,), device=self.device
        )
        t_from_q0 = self.q0_times_t[q0_indices].flatten()
        n_random = size - n_q0_sample
        t_random = torch.rand(n_random, device=self.device) * self.t_max
        t_cache = torch.cat([t_from_q0, t_random]).reshape(-1, 1)

        # Normalized spatial sampling: u ∈ [0,1] where 0=surface, 1=water table
        n_boundary = int(size * self.boundary_ratio)
        n_interior = size - n_boundary

        # Split boundary points between surface and bottom
        n_surface = n_boundary // 2
        n_bottom = n_boundary - n_surface

        # Surface region: Beta(1,3) to concentrate near u=0
        u_surface = (
            torch.distributions.Beta(1.0, 3.0).sample((n_surface,)).to(self.device)
        )

        # Bottom region: Beta(3,1) to concentrate near u=1
        u_bottom = (
            torch.distributions.Beta(3.0, 1.0).sample((n_bottom,)).to(self.device)
        )

        # Interior points: uniform distribution
        u_interior = torch.rand(n_interior, device=self.device)

        # Combine all normalized spatial points
        u_cache = torch.cat([u_surface, u_bottom, u_interior]).reshape(-1, 1)

        # Shuffle paired u/t together
        perm = torch.randperm(size, device=self.device)
        return u_cache[perm], t_cache[perm]

    def update_residuals(self, model, epoch, resample_freq):
        """Update cache residuals and sampling probabilities."""
        if epoch % resample_freq != 0:
            return

        model.eval()

        # Evaluate residuals over cache in chunks
        residual_vals = []
        chunk_size = 500

        for i in range(0, self.cache_size, chunk_size):
            j = min(i + chunk_size, self.cache_size)
            u_chunk = self.u_cache[i:j].clone()
            t_chunk = self.t_cache[i:j].clone().requires_grad_(True)

            # Map u to z using predicted water table depth
            with torch.no_grad():
                zb_chunk = model.predict_water_table(t_chunk)
            z_chunk = (-u_chunk * zb_chunk).requires_grad_(True)

            res = model.pde_residual(z_chunk, t_chunk)
            residual_vals.append(res.detach().abs().squeeze())

        self.cache_residuals[:] = torch.cat(residual_vals)

        # Track statistics
        self.cache_stats["resample_epochs"].append(epoch)
        self.cache_stats["mean_residual"].append(self.cache_residuals.mean().item())
        self.cache_stats["max_residual"].append(self.cache_residuals.max().item())
        self.cache_stats["std_residual"].append(self.cache_residuals.std().item())

        # Compute sampling probabilities with safety checks
        tau = self.temperature if self.temperature > 0 else 1.0

        # Check for invalid residuals
        if torch.isnan(self.cache_residuals).any() or torch.isinf(self.cache_residuals).any():
            print(f"Warning: Invalid residuals detected at epoch {epoch}, using uniform probabilities")
            self.sampling_probs = torch.ones(self.cache_size, device=self.device) / self.cache_size
        elif self.cache_residuals.abs().max() < 1e-12:
            # All residuals are essentially zero - use uniform sampling
            self.sampling_probs = torch.ones(self.cache_size, device=self.device) / self.cache_size
        else:
            # Clamp residuals to prevent overflow in softmax
            residuals_clamped = torch.clamp(self.cache_residuals, min=0.0, max=1e10)
            scaled_residuals = residuals_clamped / (tau + 1e-12)
            self.sampling_probs = torch.softmax(scaled_residuals, dim=0)

            # Final safety check
            if torch.isnan(self.sampling_probs).any() or torch.isinf(self.sampling_probs).any():
                print(f"Warning: Invalid probabilities after softmax at epoch {epoch}, using uniform")
                self.sampling_probs = torch.ones(self.cache_size, device=self.device) / self.cache_size
        
        # Optionally refresh part of the cache pool
        if epoch > 0 and epoch % (resample_freq * 5) == 0:
            refresh_size = self.cache_size // 10
            refresh_idx = torch.randperm(self.cache_size, device=self.device)[
                :refresh_size
            ]
            u_new, t_new = self._generate_cache_pool(refresh_size)
            self.u_cache[refresh_idx] = u_new
            self.t_cache[refresh_idx] = t_new

        model.train()

    def sample_batch(self, model, epoch):
        """Sample batch from cache based on PDE residuals."""
        # Sample batch indices
        if self.sampling_probs is None or epoch == 0:
            # Initial warm-up: random sampling
            batch_idx = torch.randint(
                0, self.cache_size, (self.batch_size,), device=self.device
            )
        else:
            # Split batch: high-residual points + random points
            n_high = int(self.batch_size * self.high_residual_ratio)
            n_rand = self.batch_size - n_high

            # Sample high-residual points
            if n_high > 0:
                high_idx = torch.multinomial(
                    self.sampling_probs, n_high, replacement=(n_high > self.cache_size)
                )
            else:
                high_idx = torch.tensor([], dtype=torch.long, device=self.device)

            # Sample random points
            rand_idx = torch.randint(0, self.cache_size, (n_rand,), device=self.device)

            # Combine indices
            batch_idx = torch.cat([high_idx, rand_idx])

        # Get batch points from cache
        u_batch = self.u_cache[batch_idx].clone()
        t_batch = self.t_cache[batch_idx].clone().requires_grad_(True)

        # Map u to z using predicted water table depth
        zb_batch = model.predict_water_table(t_batch)
        z_batch = (-u_batch * zb_batch).requires_grad_(True)

        return z_batch, t_batch

    def print_stats(self, epoch):
        """Print cache statistics if available."""
        if (
            self.cache_stats["resample_epochs"]
            and self.cache_stats["resample_epochs"][-1] == epoch
        ):
            print(f"  Cache stats (epoch {epoch}):")
            print(f"    Mean residual: {self.cache_stats['mean_residual'][-1]:.3e}")
            print(f"    Max residual: {self.cache_stats['max_residual'][-1]:.3e}")
            print(f"    Std residual: {self.cache_stats['std_residual'][-1]:.3e}")


# Gradient computation utilities
def compute_grad_norm(loss, model):
    """Compute the L2 norm of gradients for a specific loss term."""
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grad_norm = 0.0
    for grad in grads:
        if grad is not None:
            grad_norm += grad.norm(2).item() ** 2
    return grad_norm**0.5


def compute_total_grad_norm(model):
    """Compute total gradient norm after backward pass."""
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm(2).item() ** 2
    return total_grad_norm**0.5


def compute_full_sample_loss(model, cache_manager, q0_times_t, t_min, z_max, device,
                              batch_size_bc=100, chunk_size=1000):
    """
    Compute loss over all samples (entire dataset) for smooth loss tracking.

    Args:
        model: The PINN model
        cache_manager: Cache pool manager containing all spatial-temporal points
        q0_times_t: Boundary condition time points
        t_min: Minimum time for initial conditions
        z_max: Maximum z (surface) for initial conditions
        device: Device to use
        batch_size_bc: Batch size for boundary condition points
        chunk_size: Chunk size for processing large datasets

    Returns:
        Dictionary containing total loss and loss components computed over full dataset
    """
    model.eval()

    # Initialize accumulators for losses
    loss_accum = {
        "pde": 0.0,
        "surf": 0.0,
        "wt_head": 0.0,
        "wt_kin": 0.0,
        "ic_h": 0.0,
        "ic_zb": 0.0,
    }

    # 1. Compute PDE loss over entire cache pool (in chunks to avoid memory issues)
    n_pde_points = 0
    for i in range(0, cache_manager.cache_size, chunk_size):
        j = min(i + chunk_size, cache_manager.cache_size)
        u_chunk = cache_manager.u_cache[i:j].clone()
        t_chunk = cache_manager.t_cache[i:j].clone()

        # Map u to z using predicted water table depth
        # Note: predict_water_table uses no_grad internally, so zb_chunk has no gradient
        zb_chunk = model.predict_water_table(t_chunk)

        # Create z_chunk and enable gradients for PDE residual computation
        z_chunk = (-u_chunk * zb_chunk).detach().requires_grad_(True)
        t_chunk_grad = t_chunk.detach().requires_grad_(True)

        # Compute PDE residual (needs gradients enabled for physics derivatives)
        res_pde = model.pde_residual(z_chunk, t_chunk_grad)

        # Detach before accumulating to avoid building computation graph
        loss_accum["pde"] += (res_pde**2).sum().detach().item()
        n_pde_points += len(z_chunk)

    loss_accum["pde"] /= n_pde_points

    # 2. Compute boundary condition losses over all q0 time points
    n_bc_points = 0
    for i in range(0, len(q0_times_t), batch_size_bc):
        j = min(i + batch_size_bc, len(q0_times_t))
        t_bc = q0_times_t[i:j]

        # BC residuals need gradients for physics derivatives
        # Moisture BC (Dirichlet)
        res_surf = model.surface_moisture_bc_residual(t_bc)
        loss_accum["surf"] += (res_surf**2).sum().detach().item()

        res_wt_head = model.water_table_head_residual(t_bc)
        loss_accum["wt_head"] += (res_wt_head**2).sum().detach().item()

        res_wt_kin = model.water_table_kinematic_residual(t_bc)
        loss_accum["wt_kin"] += (res_wt_kin**2).sum().detach().item()

        n_bc_points += len(t_bc)

    loss_accum["surf"] /= n_bc_points
    loss_accum["wt_head"] /= n_bc_points
    loss_accum["wt_kin"] /= n_bc_points

    # 3. Compute initial condition losses over a representative set of points
    ic_batch_size = 200  # Use a reasonable number of IC points
    sampling_helper = SamplingHelpers()
    z_ic, t_ic = sampling_helper.sample_initial_condition_points(
        model, ic_batch_size, t_min, z_max, device
    )

    # IC residuals computation
    res_ic_h, res_ic_zb = model.initial_conditions_residual(z_ic, t_ic)
    loss_accum["ic_h"] = (res_ic_h**2).mean().detach().item()
    loss_accum["ic_zb"] = (res_ic_zb**2).mean().detach().item()

    model.train()

    return loss_accum
