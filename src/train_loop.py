import torch
from torch.optim import Adam
from .normalization_helper import NormalizationHelper
from .pinn_models import RichardsPINN
from .training_utils import (
    SamplingHelpers,
    compute_losses,
    apply_weights_and_compute_gradients,
    apply_weights_fixed_mode,
    WeightManager,
    compute_grad_norm,
    compute_total_grad_norm,
    compute_full_sample_loss
)
from .training_logger import TrainingLogger
from .gradient_based_sampling import gradient_based_sampling
import os
import glob


def train_pinn_pool_batch_autoweight(
    soil_params,
    theta0_data,
    Sy=0.3,
    zr=0.5,
    h_net_config=None,
    zb_net_config=None,
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
    batch_size_bc=100,
    ic_profile=None,  # ← Add measured IC profile parameter
    ic_type='obs',  # ← IC type: 'obs', 'linear', or 'hydrostatic'
    device='cpu',
    # Gradient-based boundary sampling parameters (three-way sampling)
    interp_ratio=0.80,           # Fraction from gradient-interpolated points
    neighbor_ratio=0.05,         # Fraction from neighbors of high-gradient regions
    baseline_ratio=0.15,         # Fraction from uniform baseline
    gradient_neighbor_expansion=2,  # Neighbors around high-gradient intervals
    gradient_threshold=0.7,      # Percentile for "high gradient" (0.7 = top 30%)
    gradient_power=2.0,          # Gradient emphasis (>1 = more aggressive)
    # HPC GPU optimizations (backward compatible, default: OFF)
    use_multi_gpu=True,  # Auto-detect and use DataParallel if multiple GPUs available
    use_amp=False,  # Mixed precision training (fp16) - reduces memory, may affect numerics
    grad_accumulation_steps=1,  # Gradient accumulation for larger effective batch size
    # Checkpointing for HPC fault tolerance
    checkpoint_dir='checkpoints',  # Directory to save checkpoints
    checkpoint_freq=None,  # Save checkpoint every N epochs (None = no checkpointing)
    resume_from_checkpoint=None,  # Path to checkpoint file to resume from
    keep_last_n_checkpoints=3,  # Keep only last N checkpoints (None = keep all)
):
    """
    Training with pool + small batch sampling approach using moisture BC.
    """

    # Validate moisture BC data
    if theta0_data is None:
        raise ValueError("theta0_data is required for moisture BC")

    # --- Derive time bounds from moisture BC data ---
    bc_times = theta0_data[0]
    bc_values = theta0_data[1]

    bc_times_t = torch.tensor(bc_times, dtype=torch.float32, device=device).view(-1, 1)
    bc_values_t = torch.tensor(bc_values, dtype=torch.float32, device=device)  # For importance weighting
    # ✅ GPU-OPTIMIZED: Batch min/max computation on GPU, single sync
    t_min_gpu = bc_times_t.min()
    t_max_gpu = bc_times_t.max()
    t_min = float(t_min_gpu.item())
    t_max = float(t_max_gpu.item())
    z_max = 0.0  # surface at z = 0

    # --- CREATE NORMALIZER (NEW) ---
    normalizer = NormalizationHelper(soil_params, L=L, S_max=S_max)

    # --- Model & optimizer ---
    model = RichardsPINN(
        soil_params=soil_params,
        theta0_data=theta0_data,
        Sy=Sy,
        zr=zr,
        h_net_config=h_net_config,
        zb_net_config=zb_net_config,
        normalizer=normalizer,  # ← Add normalizer
        zb_initial=zb_initial,
        t_max=t_max,  # ← Add t_max
        z_max_tilde=1.0,  # ← Optional, for network scaling
        ic_profile=ic_profile,  # ← Add measured IC profile
        ic_type=ic_type,  # ← Add IC type
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

    # --- Checkpointing setup ---
    start_epoch = 0
    checkpoint_path_list = []

    # Create checkpoint directory if needed
    if checkpoint_freq is not None and checkpoint_freq > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpointing enabled: saving every {checkpoint_freq} epochs to {checkpoint_dir}/")

    # Resume from checkpoint if specified
    if resume_from_checkpoint is not None:
        if os.path.exists(resume_from_checkpoint):
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location=device)

            # Load model state
            model_core.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load training state
            start_epoch = checkpoint['epoch'] + 1

            # Load weight manager state
            if 'weight_manager_state' in checkpoint:
                weight_manager.weights = checkpoint['weight_manager_state']['weights']
                weight_manager.weight_history = checkpoint['weight_manager_state']['weight_history']
                weight_manager.grad_ema = checkpoint['weight_manager_state']['grad_ema']

            # Load logger state
            if 'logger_state' in checkpoint:
                logger.losses = checkpoint['logger_state']['losses']
                logger.comps = checkpoint['logger_state']['comps']
                logger.sample_losses = checkpoint['logger_state']['sample_losses']
                logger.sample_comps = checkpoint['logger_state']['sample_comps']
                logger.sample_epochs = checkpoint['logger_state']['sample_epochs']

            # Load AMP scaler state if using AMP
            if use_amp and scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # Load random states for reproducibility
            if 'rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['rng_state'])
            if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint file not found: {resume_from_checkpoint}")
            print("Starting training from scratch")

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
        f"Direct Sampling: batch_size={batch_size} (no cache, no adaptive residuals)"
    )
    print(
        f"Boundary ratio: {boundary_ratio:.1%}"
    )
    print(
        f"Time domain from theta0: t∈[{t_min:.3f}, {t_max:.3f}], "
        f"z adaptive in [-z_b(t), 0]"
    )
    print(f"Initial weights: {weight_manager.get_weights()}")

    # --- Main training loop (GPU-optimized) ---
    for epoch in range(start_epoch, n_epochs):
        # Gradient accumulation: only zero grad at start of accumulation cycle
        if epoch % grad_accumulation_steps == 0:
            optimizer.zero_grad()

        # Sample training points directly (no cache, no adaptive residuals)
        z_col, t_col = sampling.sample_pde_points_direct(
            model_core, batch_size, t_max, device, boundary_ratio=boundary_ratio
        )

        # Gradient-based boundary condition sampling (three-way: interp + neighbor + baseline)
        t_bc, _ = gradient_based_sampling(
            bc_times_t,
            bc_values_t,
            batch_size_bc,
            device=device,
            use_interpolation=True,
            interp_ratio=interp_ratio,
            neighbor_ratio=neighbor_ratio,
            baseline_ratio=baseline_ratio,
            neighbor_expansion=gradient_neighbor_expansion,
            gradient_threshold=gradient_threshold,
            power=gradient_power
        )
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

        # Save checkpoint periodically (minimal version without training history)
        if checkpoint_freq is not None and (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')

            # Prepare checkpoint dictionary (minimal - no training history)
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model_core.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'weight_manager_state': {
                    'weights': weight_manager.weights,  # Current weights only
                    'grad_ema': weight_manager.grad_ema,  # Current gradient EMA only
                },
                # Training history removed to reduce file size
                'training_config': {
                    'n_epochs': n_epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                },
                # Normalization parameters (critical for fine-tuning)
                'normalization_params': {
                    'soil_params': soil_params,
                    'L': L,
                    'S_max': S_max,
                    'Sy': Sy,
                    'zr': zr,
                },
                # Network scaling parameters (critical for fine-tuning)
                'network_scaling': {
                    't_ref_tilde': model_core.t_ref_tilde,
                    'z_max_tilde': model_core.z_max_tilde,
                },
                'rng_state': torch.get_rng_state(),
            }

            # Save AMP scaler state if using mixed precision
            if use_amp and scaler is not None:
                checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

            # Save CUDA RNG state if available
            if torch.cuda.is_available():
                checkpoint_dict['cuda_rng_state'] = torch.cuda.get_rng_state()

            # Save checkpoint
            torch.save(checkpoint_dict, checkpoint_path)
            checkpoint_path_list.append(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

            # Manage checkpoint retention (keep only last N checkpoints)
            if keep_last_n_checkpoints is not None and len(checkpoint_path_list) > keep_last_n_checkpoints:
                old_checkpoint = checkpoint_path_list.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    print(f"  Removed old checkpoint: {old_checkpoint}")

        # Print progress
        if (epoch + 1) % 200 == 0 or epoch == 0:
            logger.print_progress(
                epoch, n_epochs, total_loss, weighted_losses, gradients, weights, cache_manager=None
            )

        # Compute and record sample loss (over full dataset) every 500 epochs
        if (epoch + 1) % 500 == 0 or epoch == 0:
            sample_losses_dict = compute_full_sample_loss(
                model_core, None, bc_times_t, t_min, z_max, device,
                t_max=t_max, boundary_ratio=boundary_ratio, sample_size=5000
            )
            logger.record_sample_losses(epoch, sample_losses_dict, weights)

            # ✅ GPU-OPTIMIZED: sample_losses_dict already contains CPU scalars from compute_full_sample_loss
            # Print sample loss info
            total_sample_loss = sum(weights[key] * sample_losses_dict[key] for key in sample_losses_dict)
            print(f"\n  [Sample Loss at epoch {epoch+1}] Total={total_sample_loss:.3e}")
            print(f"    PDE={sample_losses_dict['pde']:.3e}, Surf={sample_losses_dict['surf']:.3e}")
            print(f"    WT(h)={sample_losses_dict['wt_head']:.3e}, WT(kin)={sample_losses_dict['wt_kin']:.3e}")
            print(f"    IC(h)={sample_losses_dict['ic_h']:.3e}, IC(zb)={sample_losses_dict['ic_zb']:.3e}")

    # Print final summary
    final_grad_norm = gradients.get("total", 0.0)
    logger.print_final_summary(final_grad_norm, weights, cache_manager=None)

    # Save final checkpoint (minimal version without training history)
    if checkpoint_freq is not None:
        final_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_final.pt')
        checkpoint_dict = {
            'epoch': n_epochs - 1,
            'model_state_dict': model_core.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'weight_manager_state': {
                'weights': weight_manager.weights,  # Current weights only (no history)
                'grad_ema': weight_manager.grad_ema,  # Current gradient EMA only
            },
            # Training history removed to reduce file size (saved separately in plots/logs)
            # If you need history, use intermediate checkpoints (checkpoint_epoch_XXXX.pt)
            'training_config': {
                'n_epochs': n_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'cache_size': cache_size,
            },
            # Normalization parameters (critical for fine-tuning)
            'normalization_params': {
                'soil_params': soil_params,
                'L': L,
                'S_max': S_max,
                'Sy': Sy,
                'zr': zr,
            },
            # Network scaling parameters (critical for fine-tuning)
            'network_scaling': {
                't_ref_tilde': model_core.t_ref_tilde,
                'z_max_tilde': model_core.z_max_tilde,
            },
            'rng_state': torch.get_rng_state(),
        }
        if use_amp and scaler is not None:
            checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
        if torch.cuda.is_available():
            checkpoint_dict['cuda_rng_state'] = torch.cuda.get_rng_state()

        torch.save(checkpoint_dict, final_checkpoint_path)
        print(f"\nFinal checkpoint saved: {final_checkpoint_path}")

    # Return unwrapped model for backward compatibility (model_core is the original model)
    # ✅ GPU-OPTIMIZED: Convert stored tensors to numpy arrays for backward compatibility
    losses_np = logger.get_losses_numpy()
    comps_np = logger.get_comps_numpy()

    # To plot sample losses, use: plot_training_losses(losses_np, comps_np,
    #                                                  logger.sample_losses, logger.sample_comps, logger.sample_epochs)
    return model_core, losses_np, comps_np, logger.sample_losses, logger.sample_comps, logger.sample_epochs


def load_pretrained_model(checkpoint_path, device='cpu'):
    """
    Load a pretrained model from checkpoint with all normalization parameters.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model on

    Returns:
        Tuple of (model, checkpoint_dict) where checkpoint_dict contains:
            - normalization_params: soil_params, L, S_max, Sy, zr
            - network_scaling: t_ref_tilde, z_max_tilde
            - h_net_config, zb_net_config (must be same as training)

    Example:
        model, ckpt = load_pretrained_model('checkpoints/checkpoint_final.pt')
        norm_params = ckpt['normalization_params']
        # Use these for fine-tuning with new q0_data
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading pretrained model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Validate checkpoint contains required fields
    required_fields = ['normalization_params', 'network_scaling', 'model_state_dict']
    for field in required_fields:
        if field not in checkpoint:
            raise ValueError(f"Checkpoint missing required field: {field}")

    print("Checkpoint loaded successfully")
    print(f"  Trained for {checkpoint['epoch']+1} epochs")
    print(f"  Normalization params: L={checkpoint['normalization_params']['L']:.2f}, "
          f"S_max={checkpoint['normalization_params']['S_max']:.2e}")

    # Support both old and new checkpoint formats
    scaling_info = checkpoint['network_scaling']
    if 't_ref_tilde' in scaling_info:
        print(f"  Network scaling: t_ref_tilde={scaling_info['t_ref_tilde']:.2f}, "
              f"z_max_tilde={scaling_info['z_max_tilde']:.2f}")
    elif 't_max_tilde' in scaling_info:
        print(f"  Network scaling: t_max_tilde={scaling_info['t_max_tilde']:.2f} (legacy), "
              f"z_max_tilde={scaling_info['z_max_tilde']:.2f}")

    return checkpoint


def finetune_pinn(
    checkpoint_path,
    new_theta0_data,
    zb_initial=None,
    h_net_config=None,
    zb_net_config=None,
    # Fine-tuning hyperparameters (lower learning rate, fewer epochs)
    n_epochs=10000,
    learning_rate=1e-4,
    # Training parameters (same as base training)
    cache_size=5000,
    batch_size=500,
    resample_freq=100,
    boundary_ratio=0.7,
    high_residual_ratio=0.6,
    temperature=1.0,
    batch_size_bc=100,
    # Initial condition parameters
    ic_profile=None,  # Measured IC profile parameter
    ic_type='obs',    # IC type: 'obs', 'linear', or 'hydrostatic'
    # Gradient-based boundary sampling parameters
    interp_ratio=0.80,
    neighbor_ratio=0.05,
    baseline_ratio=0.15,
    gradient_neighbor_expansion=2,
    gradient_threshold=0.7,
    gradient_power=2.0,
    # Weight management (default: fixed weights)
    weight_update_freq=1e10,  # Very high = fixed weights
    weight_lr=0.1,
    use_initial_scales=True,
    # Device and GPU settings
    device='cpu',
    use_multi_gpu=True,
    use_amp=False,
    grad_accumulation_steps=1,
    # Checkpointing (isolated from base training)
    checkpoint_dir='checkpoints_finetune',
    checkpoint_freq=None,
    keep_last_n_checkpoints=3,
):
    """
    Fine-tune a pretrained PINN model on new moisture boundary condition data.

    Works with any time duration - uses fixed reference time from base model.

    Args:
        checkpoint_path: Path to base model checkpoint (.pt file)
        new_theta0_data: NEW moisture BC data, tuple of (times, soil_moisture)
                        Can have different time duration than original training data
        zb_initial: Initial water table depth for new scenario (if None, use from checkpoint)
        h_net_config: Network config (if None, must be same as training)
        zb_net_config: Network config (if None, must be same as training)

        n_epochs: Number of fine-tuning epochs (default: 10k, much less than training)
        learning_rate: Learning rate (default: 1e-4, lower than training)

        ic_profile: Initial condition profile (dict with 'depths' and 'theta' keys)
                   If None, uses hydrostatic profile
        ic_type: Type of IC ('obs', 'linear', or 'hydrostatic')

        checkpoint_dir: Directory for fine-tuning checkpoints (isolated from base)
        checkpoint_freq: Save checkpoint every N epochs

        Other parameters: Same as train_pinn_pool_batch_autoweight()

    Returns:
        Same as train_pinn_pool_batch_autoweight():
        (model, losses, comps, sample_losses, sample_comps, sample_epochs)

    Example usage:
        # Load base model trained on Jan-Mar data
        # Fine-tune on May-Aug data with new rainfall
        model, losses, comps, sl, sc, se = finetune_pinn(
            checkpoint_path='checkpoints/checkpoint_final.pt',
            new_q0_data=may_aug_flux,
            zb_initial=1.8,  # Estimate from April observations
            n_epochs=10000,
            learning_rate=1e-4,
            checkpoint_freq=2000,
            device='cuda',
        )
    """

    # Load base checkpoint
    checkpoint = load_pretrained_model(checkpoint_path, device=device)

    # Extract normalization parameters (MUST preserve these!)
    norm_params = checkpoint['normalization_params']
    soil_params = norm_params['soil_params']
    L = norm_params['L']
    S_max = norm_params['S_max']
    Sy = norm_params['Sy']
    zr = norm_params['zr']

    # Extract network scaling (MUST preserve these!)
    network_scaling = checkpoint['network_scaling']

    # Support both old (t_max_tilde) and new (t_ref_tilde) checkpoint formats
    if 't_ref_tilde' in network_scaling:
        t_ref_tilde_base = network_scaling['t_ref_tilde']
    elif 't_max_tilde' in network_scaling:
        # Legacy checkpoint - convert to fixed reference
        print(f"WARNING: Old checkpoint format detected (using t_max_tilde)")
        print(f"  Converting to fixed reference time of 15 days...")
        normalizer = NormalizationHelper(soil_params, L=L, S_max=S_max)
        t_ref_tilde_base = (15.0 * 86400) / normalizer.T
    else:
        raise ValueError("Checkpoint missing both 't_ref_tilde' and 't_max_tilde'")

    z_max_tilde = network_scaling['z_max_tilde']

    # Get new data time range
    new_t_min = min(new_theta0_data[0])
    new_t_max = max(new_theta0_data[0])
    new_duration = new_t_max - new_t_min

    # Create normalizer (same as base)
    normalizer = NormalizationHelper(soil_params, L=L, S_max=S_max)

    print("\n" + "="*70)
    print("Fine-Tuning Configuration")
    print("="*70)
    print(f"Base checkpoint: {checkpoint_path}")
    print(f"Base training epochs: {checkpoint['epoch']+1}")
    print(f"\nPreserved normalization parameters:")
    print(f"  soil_params: θs={soil_params['theta_s']}, θr={soil_params['theta_r']}, "
          f"α={soil_params['alpha']}, n={soil_params['n']}")
    print(f"  L={L:.2f} m, S_max={S_max:.2e} 1/s")
    print(f"  Sy={Sy}, zr={zr:.2f} m")
    print(f"\nPreserved network scaling:")
    print(f"  t_ref_tilde={t_ref_tilde_base:.4f} (FIXED reference time from base model)")
    print(f"  t_ref_days={t_ref_tilde_base * normalizer.T / 86400:.2f} days")
    print(f"  z_max_tilde={z_max_tilde:.4f}")
    print(f"\nNew boundary condition data:")
    print(f"  Time range: [{new_t_min:.1f}, {new_t_max:.1f}] s")
    print(f"  Duration: {new_duration/86400:.2f} days (can differ from training!)")
    print(f"  {len(new_theta0_data[0])} data points")
    print(f"\nFine-tuning hyperparameters:")
    print(f"  n_epochs={n_epochs} (vs {checkpoint.get('training_config', {}).get('n_epochs', 'N/A')} base)")
    print(f"  learning_rate={learning_rate:.2e}")
    print(f"  Fixed weights: {weight_update_freq >= n_epochs}")
    print("="*70 + "\n")

    # Determine network configs
    if h_net_config is None or zb_net_config is None:
        # Try to infer from checkpoint (if saved) or raise error
        if 'h_net_config' in checkpoint and 'zb_net_config' in checkpoint:
            h_net_config = checkpoint['h_net_config']
            zb_net_config = checkpoint['zb_net_config']
            print(f"Using network configs from checkpoint")
        else:
            raise ValueError(
                "h_net_config and zb_net_config not provided and not found in checkpoint. "
                "Please provide them explicitly."
            )

    # Determine zb_initial
    if zb_initial is None:
        # Try to extract from checkpoint
        if 'zb_initial' in checkpoint:
            zb_initial = checkpoint['zb_initial']
            print(f"Using zb_initial={zb_initial:.3f} from checkpoint")
        else:
            raise ValueError("zb_initial not provided and not found in checkpoint")

    # Create model with SAME normalization as base
    # Convert t_ref_tilde back to dimensional for model initialization
    t_ref_days = t_ref_tilde_base * normalizer.T / 86400

    model = RichardsPINN(
        soil_params=soil_params,
        theta0_data=new_theta0_data,  # NEW boundary conditions
        Sy=Sy,
        zr=zr,
        h_net_config=h_net_config,
        zb_net_config=zb_net_config,
        normalizer=normalizer,  # Same normalizer as base
        zb_initial=zb_initial,
        t_max=new_t_max,  # Use NEW data's t_max (doesn't affect network scaling anymore)
        z_max_tilde=z_max_tilde,  # Preserve from base
        t_ref_days=t_ref_days,  # FIXED reference time from base model
        ic_profile=ic_profile,  # NEW initial condition profile
        ic_type=ic_type,        # IC type for fine-tuning
        device=device,
    ).to(device)

    # Load pretrained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded pretrained weights from base model")

    # CRITICAL: Update theta0 data to NEW boundary conditions
    # load_state_dict() overwrites the theta0 buffers with OLD training data
    # We must manually restore the NEW theta0 data after loading weights
    model.theta0_times_dim = torch.tensor(new_theta0_data[0], dtype=torch.float32, device=device)
    model.theta0_values_dim = torch.tensor(new_theta0_data[1], dtype=torch.float32, device=device)
    model.theta0_times_tilde = normalizer.normalize_t(model.theta0_times_dim)
    model.theta0_values_tilde = (model.theta0_values_dim - normalizer.theta_r) / normalizer.theta_star
    print(f"Updated boundary condition data to NEW theta0 (overriding checkpoint data)")

    # Multi-GPU setup (same as base training)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_data_parallel = use_multi_gpu and n_gpus > 1

    if use_data_parallel:
        print(f"Multi-GPU mode: Using {n_gpus} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
        model_core = model.module
    else:
        model_core = model

    # Mixed precision setup
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

    # Optimizer (fresh optimizer for fine-tuning)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Initialize managers and helpers
    use_fixed_weights = weight_update_freq >= n_epochs
    if use_fixed_weights:
        print(f"Fixed weights mode enabled (weight_update_freq={weight_update_freq} >= n_epochs={n_epochs})")

    # Initialize weights from checkpoint if available, otherwise use initial scales
    weight_manager = WeightManager(use_initial_scales, weight_lr, use_fixed_weights=use_fixed_weights)
    if 'weight_manager_state' in checkpoint:
        weight_manager.weights = checkpoint['weight_manager_state']['weights']
        print(f"Using weights from base model: {weight_manager.weights}")
    else:
        print(f"Using initial scale-based weights: {weight_manager.weights}")

    logger = TrainingLogger()
    sampling = SamplingHelpers()

    # Checkpointing setup (isolated directory)
    start_epoch = 0
    checkpoint_path_list = []

    if checkpoint_freq is not None and checkpoint_freq > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpointing enabled: saving every {checkpoint_freq} epochs to {checkpoint_dir}/")

    # Initialize cache pool manager
    theta0_times_t = torch.tensor(new_theta0_data[0], dtype=torch.float32, device=device).view(-1, 1)
    # ✅ GPU-OPTIMIZED: Batch min/max computation on GPU, single sync
    t_min_gpu = theta0_times_t.min()
    t_max_gpu = theta0_times_t.max()
    t_min = float(t_min_gpu.item())
    t_max = float(t_max_gpu.item())

    # Print training information
    print(f"\nStarting fine-tuning for {n_epochs} epochs | lr={learning_rate:.2e}")
    print(f"Device: {device} | GPUs available: {n_gpus}")
    if use_data_parallel:
        print(f"Multi-GPU: Enabled ({n_gpus} GPUs)")
    if use_amp:
        print(f"Mixed Precision (AMP): Enabled")
    if grad_accumulation_steps > 1:
        print(f"Gradient Accumulation: {grad_accumulation_steps} steps")
    print(f"Fixed weights: {use_fixed_weights}")
    print(
        f"Direct Sampling: batch_size={batch_size} (no cache, no adaptive residuals)"
    )
    print(
        f"Boundary ratio: {boundary_ratio:.1%}"
    )
    print(f"Initial weights: {weight_manager.get_weights()}\n")

    # ✅ GPU-OPTIMIZED: Cache BC values tensor BEFORE loop (not recreated every epoch)
    theta0_values_t = torch.tensor(new_theta0_data[1], dtype=torch.float32, device=device)

    # Main training loop (identical to base training)
    for epoch in range(start_epoch, n_epochs):
        if epoch % grad_accumulation_steps == 0:
            optimizer.zero_grad()

        # Sample training points directly (no cache, no adaptive residuals)
        z_col, t_col = sampling.sample_pde_points_direct(
            model_core, batch_size, t_max, device, boundary_ratio=boundary_ratio
        )

        # ✅ GPU-OPTIMIZED: Use cached theta0_values_t (no recreation)
        # Gradient-based boundary condition sampling
        t_bc, _ = gradient_based_sampling(
            theta0_times_t,
            theta0_values_t,  # ✅ Reused from cache
            batch_size_bc,
            device=device,
            use_interpolation=True,
            interp_ratio=interp_ratio,
            neighbor_ratio=neighbor_ratio,
            baseline_ratio=baseline_ratio,
            neighbor_expansion=gradient_neighbor_expansion,
            gradient_threshold=gradient_threshold,
            power=gradient_power
        )
        z_ic, t_ic = sampling.sample_initial_condition_points(
            model_core, batch_size, t_min, 0.0, device
        )

        if use_amp:
            with torch.cuda.amp.autocast():
                losses = compute_losses(model, z_col, t_col, t_bc, z_ic, t_ic)
        else:
            losses = compute_losses(model, z_col, t_col, t_bc, z_ic, t_ic)

        weights = weight_manager.get_weights()
        if weight_manager.is_using_fixed_weights():
            weighted_losses, gradients, total_loss = apply_weights_fixed_mode(losses, weights)
        else:
            weighted_losses, gradients, total_loss = apply_weights_and_compute_gradients(
                losses, weights, model
            )

        total_loss = total_loss / grad_accumulation_steps

        if use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (epoch + 1) % grad_accumulation_steps == 0:
            if not weight_manager.is_using_fixed_weights():
                total_grad_norm = compute_total_grad_norm(model)
                gradients["total"] = total_grad_norm
            else:
                gradients["total"] = 0.0

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        else:
            gradients["total"] = 0.0

        if (epoch + 1) % weight_update_freq == 0 and epoch > 0:
            weight_manager.update(gradients)
            weight_manager.print_update(epoch)

        logger.record_losses(total_loss, weighted_losses)
        logger.record_gradients(gradients)
        weight_manager.record_history()

        # Save checkpoint periodically (minimal version without training history)
        if checkpoint_freq is not None and (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path_ft = os.path.join(checkpoint_dir, f'finetune_epoch_{epoch+1}.pt')

            checkpoint_dict_ft = {
                'epoch': epoch,
                'model_state_dict': model_core.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'weight_manager_state': {
                    'weights': weight_manager.weights,  # Current weights only
                    'grad_ema': weight_manager.grad_ema,  # Current gradient EMA only
                },
                # Training history removed to reduce file size
                'training_config': {
                    'n_epochs': n_epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                },
                'normalization_params': {
                    'soil_params': soil_params,
                    'L': L,
                    'S_max': S_max,
                    'Sy': Sy,
                    'zr': zr,
                },
                'network_scaling': {
                    't_ref_tilde': model_core.t_ref_tilde,
                    'z_max_tilde': model_core.z_max_tilde,
                },
                'base_checkpoint': checkpoint_path,  # Track lineage
                'rng_state': torch.get_rng_state(),
            }

            if use_amp and scaler is not None:
                checkpoint_dict_ft['scaler_state_dict'] = scaler.state_dict()
            if torch.cuda.is_available():
                checkpoint_dict_ft['cuda_rng_state'] = torch.cuda.get_rng_state()

            torch.save(checkpoint_dict_ft, checkpoint_path_ft)
            checkpoint_path_list.append(checkpoint_path_ft)
            print(f"  Checkpoint saved: {checkpoint_path_ft}")

            if keep_last_n_checkpoints is not None and len(checkpoint_path_list) > keep_last_n_checkpoints:
                old_checkpoint = checkpoint_path_list.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    print(f"  Removed old checkpoint: {old_checkpoint}")

        # Print progress
        if (epoch + 1) % 200 == 0 or epoch == 0:
            logger.print_progress(
                epoch, n_epochs, total_loss, weighted_losses, gradients, weights, cache_manager=None
            )

        # Compute and record sample loss every 500 epochs
        if (epoch + 1) % 500 == 0 or epoch == 0:
            sample_losses_dict = compute_full_sample_loss(
                model_core, None, theta0_times_t, t_min, 0.0, device,
                t_max=t_max, boundary_ratio=boundary_ratio, sample_size=5000
            )
            logger.record_sample_losses(epoch, sample_losses_dict, weights)

            total_sample_loss = sum(weights[key] * sample_losses_dict[key] for key in sample_losses_dict)
            print(f"\n  [Sample Loss at epoch {epoch+1}] Total={total_sample_loss:.3e}")
            print(f"    PDE={sample_losses_dict['pde']:.3e}, Surf={sample_losses_dict['surf']:.3e}")
            print(f"    WT(h)={sample_losses_dict['wt_head']:.3e}, WT(kin)={sample_losses_dict['wt_kin']:.3e}")
            print(f"    IC(h)={sample_losses_dict['ic_h']:.3e}, IC(zb)={sample_losses_dict['ic_zb']:.3e}")

    # Print final summary
    final_grad_norm = gradients.get("total", 0.0)
    logger.print_final_summary(final_grad_norm, weights, cache_manager=None)

    # Save final checkpoint
    if checkpoint_freq is not None:
        final_checkpoint_path = os.path.join(checkpoint_dir, 'finetune_final.pt')
        checkpoint_dict_final = {
            'epoch': n_epochs - 1,
            'model_state_dict': model_core.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'weight_manager_state': {
                'weights': weight_manager.weights,  # Current weights only (no history)
                'grad_ema': weight_manager.grad_ema,  # Current gradient EMA only
            },
            # Training history removed to reduce file size (saved separately in plots/logs)
            # If you need history, use intermediate checkpoints (checkpoint_epoch_XXXX.pt)
            'training_config': {
                'n_epochs': n_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'cache_size': cache_size,
            },
            'normalization_params': {
                'soil_params': soil_params,
                'L': L,
                'S_max': S_max,
                'Sy': Sy,
                'zr': zr,
            },
            'network_scaling': {
                't_ref_tilde': model_core.t_ref_tilde,
                'z_max_tilde': model_core.z_max_tilde,
            },
            'base_checkpoint': checkpoint_path,  # Track lineage
            'rng_state': torch.get_rng_state(),
        }
        if use_amp and scaler is not None:
            checkpoint_dict_final['scaler_state_dict'] = scaler.state_dict()
        if torch.cuda.is_available():
            checkpoint_dict_final['cuda_rng_state'] = torch.cuda.get_rng_state()

        torch.save(checkpoint_dict_final, final_checkpoint_path)
        print(f"\nFinal fine-tuned checkpoint saved: {final_checkpoint_path}")

    # ✅ GPU-OPTIMIZED: Convert stored tensors to numpy arrays for backward compatibility
    losses_np = logger.get_losses_numpy()
    comps_np = logger.get_comps_numpy()

    return model_core, losses_np, comps_np, logger.sample_losses, logger.sample_comps, logger.sample_epochs


