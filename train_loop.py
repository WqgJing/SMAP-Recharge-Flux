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
    compute_total_grad_norm,
    compute_full_sample_loss
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
    spike_events=None,
    # HPC GPU optimizations (backward compatible, default: OFF)
    use_multi_gpu=True,  # Auto-detect and use DataParallel if multiple GPUs available
    use_amp=False,  # Mixed precision training (fp16) - reduces memory, may affect numerics
    grad_accumulation_steps=1,  # Gradient accumulation for larger effective batch size
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

    # --- HPC GPU Optimizations ---
    # Multi-GPU support (auto-detect)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_data_parallel = use_multi_gpu and n_gpus > 1

    if use_data_parallel:
        print(f"Multi-GPU mode: Using {n_gpus} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
        model_core = model.module  # Access underlying model for direct method calls
    else:
        model_core = model  # Single GPU or CPU

    # Mixed precision training setup
    scaler = None
    if use_amp:
        if not torch.cuda.is_available():
            print("Warning: AMP requested but CUDA not available. Disabling AMP.")
            use_amp = False
        else:
            scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training (AMP) enabled")

    # Gradient accumulation
    if grad_accumulation_steps > 1:
        print(f"Gradient accumulation: effective batch size = {batch_size * grad_accumulation_steps}")

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
    print(f"Device: {device} | GPUs available: {n_gpus}")
    if use_data_parallel:
        print(f"Multi-GPU: Enabled ({n_gpus} GPUs)")
    if use_amp:
        print(f"Mixed Precision (AMP): Enabled")
    if grad_accumulation_steps > 1:
        print(f"Gradient Accumulation: {grad_accumulation_steps} steps")
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

    # --- Main training loop (GPU-optimized) ---
    for epoch in range(n_epochs):
        # Gradient accumulation: only zero grad at start of accumulation cycle
        if epoch % grad_accumulation_steps == 0:
            optimizer.zero_grad()

        # Update cache residuals and sampling probabilities (use model_core for direct method access)
        cache_manager.update_residuals(model_core, epoch, resample_freq)

        # Sample training points (use model_core for direct method access)
        z_col, t_col = cache_manager.sample_batch(model_core, epoch)
        t_bc = sample_boundary_points(q0_times_t, spike_events, n_events, batch_size_bc, device)
        z_ic, t_ic = sampling.sample_initial_condition_points(
            model_core, batch_size, t_min, z_max, device
        )

        # Compute losses with optional mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                losses = compute_losses(model, z_col, t_col, t_bc, z_ic, t_ic)
        else:
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

        # Scale loss for gradient accumulation
        total_loss = total_loss / grad_accumulation_steps

        # Backpropagation with optional mixed precision
        if use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # Optimizer step (only at end of accumulation cycle)
        if (epoch + 1) % grad_accumulation_steps == 0:
            # Compute total gradient norm (only if not using fixed weights)
            if not weight_manager.is_using_fixed_weights():
                total_grad_norm = compute_total_grad_norm(model)
                gradients["total"] = total_grad_norm
            else:
                # Skip expensive gradient norm computation in fixed weight mode
                gradients["total"] = 0.0

            # Optimizer step with optional gradient scaling
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        else:
            # Not at accumulation boundary, set dummy gradient norm
            gradients["total"] = 0.0

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

        # Compute and record sample loss (over full dataset) every 500 epochs
        if (epoch + 1) % 500 == 0 or epoch == 0:
            sample_losses_dict = compute_full_sample_loss(
                model_core, cache_manager, q0_times_t, t_min, z_max, device
            )
            logger.record_sample_losses(epoch, sample_losses_dict, weights)

            # Print sample loss info
            total_sample_loss = sum(weights[key] * sample_losses_dict[key] for key in sample_losses_dict)
            print(f"\n  [Sample Loss at epoch {epoch+1}] Total={total_sample_loss:.3e}")
            print(f"    PDE={sample_losses_dict['pde']:.3e}, Surf={sample_losses_dict['surf']:.3e}")
            print(f"    WT(h)={sample_losses_dict['wt_head']:.3e}, WT(kin)={sample_losses_dict['wt_kin']:.3e}")
            print(f"    IC(h)={sample_losses_dict['ic_h']:.3e}, IC(zb)={sample_losses_dict['ic_zb']:.3e}")

    # Print final summary
    final_grad_norm = gradients.get("total", 0.0)
    logger.print_final_summary(final_grad_norm, weights, cache_manager)

    # Return unwrapped model for backward compatibility (model_core is the original model)
    # To plot sample losses, use: plot_training_losses(logger.losses, logger.comps,
    #                                                  logger.sample_losses, logger.sample_comps, logger.sample_epochs)
    return model_core, logger.losses, logger.comps, logger.sample_losses, logger.sample_comps, logger.sample_epochs


