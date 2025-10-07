import torch
from torch.optim import Adam
from normalization_helper import NormalizationHelper
from pinn_models import RichardsPINN
from training_utils import (
    SamplingHelpers, 
    compute_losses, 
    apply_weights_and_compute_gradients, 
    apply_weights_fixed_mode,
    WeightManager,
    CachePoolManager,
    compute_grad_norm,
    compute_total_grad_norm
)
from training_logger import TrainingLogger
from boundary_sampling import sample_boundary_points


def train_pinn_pool_batch_autoweight(
    soil_params,
    q0_data,
    Sy,
    zr,
    h_net_config,
    zb_net_config,
    L=5.0,  # ← Add characteristic length parameter
    S_max=1e-7,  # ← Add sink term parameter
    n_epochs=1000,
    learning_rate=1e-3,
    zb_initial=1.5,
    weight_update_freq=100,
    weight_lr=0.5,
    use_initial_scales=True,
    cache_size=5000,
    batch_size=500,
    resample_freq=100,
    boundary_ratio=0.7,
    high_residual_ratio=0.6,
    temperature=1.0,
    n_events=3,
    batch_size_bc=100,
    device='cpu',
    spike_events=None
):
    """
    Training with pool + small batch sampling approach.
    """
    
    # --- Derive time bounds from q0_data ---
    q0_times_t = torch.tensor(q0_data[0], dtype=torch.float32, device=device).view(-1, 1)
    t_min = float(q0_times_t.min().item())
    t_max = float(q0_times_t.max().item())
    z_max = 0.0  # surface at z = 0

    # --- CREATE NORMALIZER (NEW) ---
    normalizer = NormalizationHelper(soil_params, L=L, S_max=S_max)

    # --- Model & optimizer ---
    model = RichardsPINN(
        soil_params=soil_params,
        q0_data=q0_data,
        Sy=Sy,
        zr=zr,
        h_net_config=h_net_config,
        zb_net_config=zb_net_config,
        normalizer=normalizer,  # ← Add normalizer
        zb_initial=zb_initial,
        t_max=t_max,  # ← Add t_max
        z_max_tilde=1.0,  # ← Optional, for network scaling
        device=device,  # ← Add device parameter
    ).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # --- Initialize managers and helpers ---
    # Auto-detect fixed weights mode based on weight_update_freq
    use_fixed_weights = weight_update_freq >= n_epochs
    if use_fixed_weights:
        print(f"Fixed weights mode enabled (weight_update_freq={weight_update_freq} >= n_epochs={n_epochs})")
    
    weight_manager = WeightManager(use_initial_scales, weight_lr, use_fixed_weights=use_fixed_weights)
    logger = TrainingLogger()
    sampling = SamplingHelpers()

    # --- Initialize cache pool manager ---
    cache_manager = CachePoolManager(
        cache_size=cache_size,
        batch_size=batch_size,
        device=device,
        q0_times_t=q0_times_t,
        t_max=t_max,
        boundary_ratio=boundary_ratio,
        high_residual_ratio=high_residual_ratio,
        temperature=temperature,
    )

    # --- Print initial information ---
    print(f"Training for {n_epochs} epochs | lr={learning_rate}")
    print(f"Adaptive weighting: updating every {weight_update_freq} epochs")
    print(
        f"Pool + Batch: cache_size={cache_size}, batch_size={batch_size}, "
        f"resample_freq={resample_freq}"
    )
    print(
        f"Boundary ratio: {boundary_ratio:.1%}, "
        f"High residual ratio: {high_residual_ratio:.1%}"
    )
    print(
        f"Time domain from q0: t∈[{t_min:.3f}, {t_max:.3f}], "
        f"z adaptive in [-z_b(t), 0]"
    )
    print(f"Initial weights: {weight_manager.get_weights()}")

    # --- Main training loop (UNCHANGED) ---
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Update cache residuals and sampling probabilities
        cache_manager.update_residuals(model, epoch, resample_freq)

        # Sample training points
        z_col, t_col = cache_manager.sample_batch(model, epoch)
        t_bc = sample_boundary_points(q0_times_t, spike_events, n_events, batch_size_bc, device)
        z_ic, t_ic = sampling.sample_initial_condition_points(
            model, batch_size, t_min, z_max, device
        )

        # Compute losses
        losses = compute_losses(model, z_col, t_col, t_bc, z_ic, t_ic)

        # Apply weights and compute gradients (conditionally)
        weights = weight_manager.get_weights()
        if weight_manager.is_using_fixed_weights():
            # Use lightweight computation for fixed weights
            weighted_losses, gradients, total_loss = apply_weights_fixed_mode(losses, weights)
        else:
            # Use full gradient computation for adaptive weights
            weighted_losses, gradients, total_loss = apply_weights_and_compute_gradients(
                losses, weights, model
            )

        # Backpropagation
        total_loss.backward()

        # Compute total gradient norm (only if not using fixed weights)
        if not weight_manager.is_using_fixed_weights():
            total_grad_norm = compute_total_grad_norm(model)
            gradients["total"] = total_grad_norm
        else:
            # Skip expensive gradient norm computation in fixed weight mode
            gradients["total"] = 0.0

        # Optimizer step
        optimizer.step()

        # Update weights periodically
        if (epoch + 1) % weight_update_freq == 0 and epoch > 0:
            weight_manager.update(gradients)
            weight_manager.print_update(epoch)

        # Record metrics
        logger.record_losses(total_loss, weighted_losses)
        logger.record_gradients(gradients)
        weight_manager.record_history()

        # Print progress
        if (epoch + 1) % 200 == 0 or epoch == 0:
            logger.print_progress(
                epoch, n_epochs, total_loss, weighted_losses, gradients, weights, cache_manager
            )

    # Print final summary
    final_grad_norm = gradients.get("total", 0.0)
    logger.print_final_summary(final_grad_norm, weights, cache_manager)

    return model, logger.losses, logger.comps


