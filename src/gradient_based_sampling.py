import torch
import numpy as np


def compute_gradient_weights(q0_times_t, q0_values_t, device='cpu'):
    """
    Compute sampling weights based on gradient magnitude |dq/dt|.

    High gradient → high weight → more samples
    Low gradient → low weight → fewer samples

    Args:
        q0_times_t: torch.Tensor - time points (N, 1) or (N,)
        q0_values_t: torch.Tensor - flux values (N,)
        device: str or torch.device

    Returns:
        gradient_weights: torch.Tensor - sampling weights proportional to |dq/dt| (N,)
        gradients: torch.Tensor - actual gradient values dq/dt (N-1,)
    """
    times_flat = q0_times_t.flatten()

    # Compute temporal gradients dq/dt
    dq = torch.diff(q0_values_t)
    dt = torch.diff(times_flat)
    gradients = dq / (dt + 1e-12)  # Avoid division by zero

    # Take absolute value (we care about magnitude of change)
    gradient_magnitude = torch.abs(gradients)

    # Extend to same length as original (use forward difference for last point)
    gradient_weights = torch.zeros(len(q0_values_t), device=device)
    gradient_weights[:-1] = gradient_magnitude
    gradient_weights[-1] = gradient_magnitude[-1]  # Repeat last gradient

    # Add small baseline so even zero-gradient regions get some samples
    baseline = gradient_weights.max() * 0.1  # 10% of max gradient
    gradient_weights = gradient_weights + baseline

    # Normalize to sum to 1 (for probability distribution)
    gradient_weights = gradient_weights / gradient_weights.sum()

    return gradient_weights, gradients


def gradient_based_sampling(
    q0_times_t,
    q0_values_t,
    batch_size_bc,
    device='cpu',
    power=1.0,              # Exponent for gradient weighting (>1 = more aggressive)
    use_interpolation=True, # Create interpolated points between data points
    # Three-way sampling ratios
    interp_ratio=0.8,       # Fraction from gradient-interpolated points
    neighbor_ratio=0.05,    # Fraction from neighbors of high-gradient regions
    baseline_ratio=0.15,    # Fraction from uniform baseline
    neighbor_expansion=2,   # How many neighbors around high-gradient regions
    gradient_threshold=0.5, # Percentile for "high gradient" (0.5 = top 50%)
):
    """
    Sample boundary condition points with THREE-WAY strategy:
    1. Gradient-interpolated points (dense where |dq/dt| is large)
    2. Neighbor points (around high-gradient regions)
    3. Baseline points (uniform sampling from rest)

    Args:
        q0_times_t: torch.Tensor - time points (N, 1) or (N,)
        q0_values_t: torch.Tensor - flux values (N,)
        batch_size_bc: int - number of samples to generate
        device: str or torch.device
        power: float - gradient weight exponent (power > 1 emphasizes high gradients more)
        use_interpolation: bool - if True, creates interpolated points
        interp_ratio: float - fraction from gradient-interpolated points (e.g., 0.8)
        neighbor_ratio: float - fraction from neighbors of high-gradient regions (e.g., 0.05)
        baseline_ratio: float - fraction from uniform baseline (e.g., 0.15)
        neighbor_expansion: int - how many neighbors around high-gradient intervals
        gradient_threshold: float - percentile for "high gradient" (0.5 = top 50%)

    Returns:
        t_bc: torch.Tensor - sampled time points (batch_size_bc, 1) with gradient tracking
        sample_info: dict - diagnostic info about sampling

    Example:
        # 80% interpolated, 5% neighbors, 15% baseline
        t_bc, info = gradient_based_sampling(
            times, fluxes, 500, 'cpu',
            interp_ratio=0.8, neighbor_ratio=0.05, baseline_ratio=0.15
        )
    """
    # Validate ratios sum to ~1
    total_ratio = interp_ratio + neighbor_ratio + baseline_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio:.3f}, normalizing to 1.0")
        interp_ratio = interp_ratio / total_ratio
        neighbor_ratio = neighbor_ratio / total_ratio
        baseline_ratio = baseline_ratio / total_ratio

    n_points = len(q0_values_t)
    times_flat = q0_times_t.flatten()

    # Compute gradient-based weights
    gradient_weights, gradients = compute_gradient_weights(q0_times_t, q0_values_t, device)

    # Apply power to emphasize high gradients (optional)
    if power != 1.0:
        gradient_weights = gradient_weights.pow(power)
        gradient_weights = gradient_weights / gradient_weights.sum()  # Re-normalize

    # Allocate samples across three categories
    n_interp_samples = int(batch_size_bc * interp_ratio)
    n_neighbor_samples = int(batch_size_bc * neighbor_ratio)
    n_baseline_samples = batch_size_bc - n_interp_samples - n_neighbor_samples

    all_samples = []

    # ===== PART 1: Gradient-Interpolated Samples (SPIKE-FOCUSED) =====
    # ✅ GPU-OPTIMIZED: Fully vectorized - no Python loops, no .item() calls
    if use_interpolation and n_interp_samples > 0:
        gradient_magnitudes = torch.abs(gradients)

        # Calculate threshold: only intervals above this percentile get interpolated
        high_grad_threshold_val = torch.quantile(gradient_magnitudes, gradient_threshold)

        # Filter to only high-gradient intervals
        high_grad_mask = gradient_magnitudes >= high_grad_threshold_val
        high_grad_indices = torch.where(high_grad_mask)[0]

        if len(high_grad_indices) > 0:
            # ✅ Vectorized: Sample intervals weighted by gradient magnitude
            high_grad_mags = gradient_magnitudes[high_grad_indices]
            interval_weights = high_grad_mags.pow(power)
            interval_weights = interval_weights / interval_weights.sum()

            # Sample intervals (with replacement to get n_interp_samples total points)
            # Each interval gets samples proportional to its gradient magnitude
            sampled_interval_indices = torch.multinomial(
                interval_weights,
                n_interp_samples,
                replacement=True
            )

            # ✅ Vectorized: Generate random interpolation positions within intervals
            # alpha ∈ [0, 1] for each sample
            alpha = torch.rand(n_interp_samples, device=device)

            # Get start and end times for sampled intervals
            interval_idx = high_grad_indices[sampled_interval_indices]
            t_start = times_flat[interval_idx]
            t_end = times_flat[interval_idx + 1]

            # ✅ Vectorized interpolation: t = t_start + alpha * (t_end - t_start)
            interp_samples = t_start + alpha * (t_end - t_start)
            all_samples.append(interp_samples.reshape(-1, 1))

    # ===== PART 2: Neighbor Samples =====
    # ✅ GPU-OPTIMIZED: Fully vectorized - no Python loops, sets, or .item() calls
    if n_neighbor_samples > 0:
        # Identify high-gradient intervals
        gradient_magnitudes = torch.abs(gradients)
        high_grad_threshold_val = torch.quantile(gradient_magnitudes, gradient_threshold)
        high_grad_mask = gradient_magnitudes >= high_grad_threshold_val
        high_grad_indices = torch.where(high_grad_mask)[0]

        # ✅ Vectorized neighbor collection using tensor broadcasting
        if len(high_grad_indices) > 0:
            # Create offset tensor [-expansion, ..., 0, ..., +expansion]
            offsets = torch.arange(-neighbor_expansion, neighbor_expansion + 1, device=device)

            # Broadcast: [num_high_grad, 1] + [1, num_offsets] = [num_high_grad, num_offsets]
            neighbor_candidates = high_grad_indices.unsqueeze(1) + offsets.unsqueeze(0)

            # Flatten and filter valid indices [0, n_points)
            neighbor_indices = neighbor_candidates.flatten()
            valid_mask = (neighbor_indices >= 0) & (neighbor_indices < n_points)
            neighbor_indices = neighbor_indices[valid_mask]

            # ✅ Remove duplicates using unique() - GPU operation, no Python sets
            neighbor_indices = torch.unique(neighbor_indices)

            if len(neighbor_indices) > 0:
                # Sample from neighbors
                if len(neighbor_indices) >= n_neighbor_samples:
                    sampled_neighbor_idx = neighbor_indices[
                        torch.randperm(len(neighbor_indices), device=device)[:n_neighbor_samples]
                    ]
                else:
                    # Sample with replacement if not enough neighbors
                    sampled_neighbor_idx = neighbor_indices[
                        torch.randint(0, len(neighbor_indices), (n_neighbor_samples,), device=device)
                    ]

                neighbor_samples = times_flat[sampled_neighbor_idx].reshape(-1, 1)
                all_samples.append(neighbor_samples)

    # ===== PART 3: Baseline Samples =====
    if n_baseline_samples > 0:
        # Sample uniformly from all points
        baseline_idx = torch.randperm(n_points, device=device)[:n_baseline_samples]
        baseline_samples = times_flat[baseline_idx].reshape(-1, 1)
        all_samples.append(baseline_samples)

    # Combine all samples
    if len(all_samples) > 0:
        t_bc = torch.cat(all_samples, dim=0)
    else:
        # Fallback: uniform sampling
        print("Warning: No samples generated, falling back to uniform sampling")
        random_idx = torch.randperm(n_points, device=device)[:batch_size_bc]
        t_bc = times_flat[random_idx].reshape(-1, 1)

    # Pad or truncate to exact batch size (handle rounding errors)
    if len(t_bc) < batch_size_bc:
        shortfall = batch_size_bc - len(t_bc)
        extra_idx = torch.randperm(n_points, device=device)[:shortfall]
        extra_samples = times_flat[extra_idx].reshape(-1, 1)
        t_bc = torch.cat([t_bc, extra_samples], dim=0)
    elif len(t_bc) > batch_size_bc:
        t_bc = t_bc[:batch_size_bc]

    # Shuffle and enable gradients
    perm = torch.randperm(len(t_bc), device=device)
    t_bc = t_bc[perm].clone().requires_grad_(True)

    # ✅ GPU-OPTIMIZED: Keep diagnostic tensors on GPU, defer .item() until needed
    sample_info = {
        'gradient_weights': gradient_weights,
        'gradients': gradients,
        'max_gradient': gradients.abs().max(),  # ✅ Keep as tensor
        'mean_gradient': gradients.abs().mean(),  # ✅ Keep as tensor
        'used_interpolation': use_interpolation,
        'n_interp_samples': n_interp_samples,
        'n_neighbor_samples': n_neighbor_samples,
        'n_baseline_samples': n_baseline_samples,
        'actual_ratios': {
            'interp': n_interp_samples / batch_size_bc,
            'neighbor': n_neighbor_samples / batch_size_bc,
            'baseline': n_baseline_samples / batch_size_bc,
        }
    }

    return t_bc, sample_info


def visualize_gradient_sampling(q0_times_t, q0_values_t, t_bc_samples, sample_info,
                                title="Gradient-Based Sampling"):
    """
    Visualize gradient-based sampling distribution.

    Shows:
    1. Flux and gradient magnitude over time
    2. Sampled points overlaid on flux
    """
    import matplotlib.pyplot as plt

    times_np = q0_times_t.cpu().numpy().flatten()
    values_np = q0_values_t.cpu().numpy().flatten()
    samples_np = t_bc_samples.detach().cpu().numpy().flatten()

    gradient_weights = sample_info['gradient_weights'].cpu().numpy()
    gradients = sample_info['gradients'].cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Flux and gradient magnitude
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ax1.plot(times_np, values_np * 1e6, 'b-', linewidth=1.5, label='Flux q (μm/s)', alpha=0.7)
    ax1.axhline(0, color='k', linewidth=0.8, alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Flux q (μm/s)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    # Plot gradient magnitude on right axis
    grad_times = (times_np[:-1] + times_np[1:]) / 2  # Midpoints
    ax1_twin.plot(grad_times, np.abs(gradients) * 1e6, 'r-', linewidth=1,
                 label='|dq/dt| (μm/s²)', alpha=0.6)
    ax1_twin.set_ylabel('|dq/dt| (μm/s²)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')

    ax1.set_title(f'{title} - Flux and Gradient')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # Plot 2: Flux with sampled points overlaid
    ax2 = axes[1]
    ax2.plot(times_np, values_np * 1e6, 'b-', linewidth=1.5, alpha=0.5, label='Flux q (μm/s)')
    ax2.scatter(samples_np, np.interp(samples_np, times_np, values_np) * 1e6,
               s=20, c='green', alpha=0.6, marker='x', label=f'Sampled points (n={len(samples_np)})', zorder=5)
    ax2.axhline(0, color='k', linewidth=0.8, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Flux (μm/s)')
    ax2.set_title('Sampled Points on Flux (density should be higher where |dq/dt| is large)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig


def compare_sampling_methods(q0_times_t, q0_values_t, batch_size_bc, device='cpu'):
    """
    Compare uniform vs gradient-based sampling side by side.
    """
    import matplotlib.pyplot as plt

    times_np = q0_times_t.cpu().numpy().flatten()
    values_np = q0_values_t.cpu().numpy().flatten()

    # Uniform sampling
    n_points = len(q0_times_t)
    uniform_idx = torch.randperm(n_points, device=device)[:batch_size_bc]
    uniform_samples = q0_times_t[uniform_idx].detach().cpu().numpy().flatten()

    # Gradient-based sampling
    gradient_samples, info = gradient_based_sampling(
        q0_times_t, q0_values_t, batch_size_bc, device
    )
    gradient_samples_np = gradient_samples.detach().cpu().numpy().flatten()

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    # Plot flux for both
    for ax in axes[:, 0]:
        ax.plot(times_np, values_np * 1e6, 'b-', linewidth=1.5, alpha=0.5)
        ax.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flux (μm/s)')
        ax.grid(True, alpha=0.3)

    # Uniform sampling scatter
    axes[0, 0].scatter(uniform_samples, np.interp(uniform_samples, times_np, values_np) * 1e6,
                      s=20, c='red', alpha=0.6, marker='x', label='Uniform samples')
    axes[0, 0].set_title('Uniform Sampling (baseline)')
    axes[0, 0].legend()

    # Gradient sampling scatter
    axes[1, 0].scatter(gradient_samples_np, np.interp(gradient_samples_np, times_np, values_np) * 1e6,
                      s=20, c='green', alpha=0.6, marker='x', label='Gradient samples')
    axes[1, 0].set_title('Gradient-Based Sampling (high |dq/dt| → more samples)')
    axes[1, 0].legend()

    # Histograms
    axes[0, 1].hist(uniform_samples, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_title('Uniform Sample Distribution')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].hist(gradient_samples_np, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('Gradient Sample Distribution\n(peaks should align with high |dq/dt|)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
