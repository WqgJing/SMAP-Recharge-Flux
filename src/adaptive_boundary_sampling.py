import torch
import numpy as np


def create_interpolated_spike_samples(
    q0_times_t,
    spike_events,
    interpolation_density=5,
    neighborhood_expansion=2,
    device='cpu'
):
    """
    Create additional interpolated sample points around detected spike events.

    This function enhances the boundary condition sampling by generating
    interpolated time points with higher density around spike regions.

    Args:
        q0_times_t: torch.Tensor - original time points (N, 1)
        spike_events: list of torch.Tensor - detected spike event indices
        interpolation_density: int - number of interpolated points to add between
                                     each pair of adjacent points in spike regions
        neighborhood_expansion: int - how many additional neighboring points to
                                     include beyond the spike event boundaries
        device: str or torch.device - device for computation

    Returns:
        enhanced_indices: torch.Tensor - original indices plus interpolated locations
        interpolation_weights: torch.Tensor - weights for non-uniform sampling
                                              (higher near spikes)
    """
    if len(spike_events) == 0:
        # No spikes detected, return uniform weights
        n_points = len(q0_times_t)
        return torch.arange(n_points, device=device), torch.ones(n_points, device=device)

    n_points = len(q0_times_t)
    all_indices = torch.arange(n_points, device=device)

    # Step 1: Identify expanded spike neighborhoods
    expanded_spike_indices = []
    for event in spike_events:
        if event.device != device:
            event = event.to(device)

        # Expand the spike region by including neighbors
        min_idx = max(0, event.min().item() - neighborhood_expansion)
        max_idx = min(n_points - 1, event.max().item() + neighborhood_expansion)

        expanded_region = torch.arange(min_idx, max_idx + 1, device=device)
        expanded_spike_indices.append(expanded_region)

    # Merge all spike regions and get unique indices
    all_spike_indices = torch.cat(expanded_spike_indices).unique()

    # Step 2: Create interpolated points within spike regions
    interpolated_times = []
    interpolated_weights = []

    for spike_region_idx in range(len(all_spike_indices) - 1):
        idx_start = all_spike_indices[spike_region_idx].item()
        idx_end = all_spike_indices[spike_region_idx + 1].item()

        # Only interpolate between adjacent or close points
        if idx_end - idx_start <= 3:
            t_start = q0_times_t[idx_start].item()
            t_end = q0_times_t[idx_end].item()

            # Create interpolated time points
            n_interp = interpolation_density
            interp_times = torch.linspace(t_start, t_end, n_interp + 2, device=device)[1:-1]

            interpolated_times.append(interp_times)

            # Assign higher weights to interpolated points in spike regions
            interp_weights = torch.ones_like(interp_times) * 2.0  # 2x weight for spike regions
            interpolated_weights.append(interp_weights)

    # Step 3: Combine original and interpolated times
    if len(interpolated_times) > 0:
        all_interp_times = torch.cat(interpolated_times)
        all_interp_weights = torch.cat(interpolated_weights)
    else:
        all_interp_times = torch.tensor([], device=device)
        all_interp_weights = torch.tensor([], device=device)

    # Create a combined set: original times with standard weights + interpolated times with higher weights
    original_weights = torch.ones(n_points, device=device)

    # Increase weights for original points that are in spike events
    spike_mask = torch.isin(all_indices, all_spike_indices)
    original_weights[spike_mask] = 3.0  # 3x weight for actual spike points

    # Return original indices (for direct sampling) and their weights
    # Also return interpolated times separately
    return {
        'original_indices': all_indices,
        'original_weights': original_weights,
        'interpolated_times': all_interp_times,
        'interpolated_weights': all_interp_weights,
        'spike_indices': all_spike_indices
    }


def adaptive_boundary_sampling(
    q0_times_t,
    spike_events,
    n_events,
    batch_size_bc,
    device='cpu',
    # Adaptive sampling parameters
    spike_ratio=0.7,          # Fraction of samples from spike regions
    interpolation_density=3,   # Number of interpolated points between spike neighbors
    neighborhood_expansion=2,  # Expand spike regions by N points
    use_weighted_sampling=True, # Use importance sampling based on weights
):
    """
    Enhanced boundary sampling with adaptive interpolation around spike events.

    This function implements non-uniform, interpolation-based sampling that:
    1. Detects spike neighborhoods (including single-point spikes)
    2. Creates interpolated sample points within spike regions
    3. Applies higher sampling density near detected spikes
    4. Maintains computational efficiency through weighted sampling

    Args:
        q0_times_t: torch.Tensor - boundary condition time points (N, 1)
        spike_events: list of torch.Tensor - detected spike events
        n_events: int - number of spike events to include
        batch_size_bc: int - total number of boundary samples per batch
        device: str or torch.device - device for computation
        spike_ratio: float - fraction of batch to sample from spike regions [0, 1]
        interpolation_density: int - number of interpolated points between neighbors
        neighborhood_expansion: int - expand spike regions by this many points
        use_weighted_sampling: bool - use importance weights for sampling

    Returns:
        t_bc: torch.Tensor - sampled time points with gradient tracking
    """
    # Handle case where no spike events were detected
    if len(spike_events) == 0:
        print("Warning: No spike events detected, using uniform random sampling")
        total_indices = len(q0_times_t)
        random_indices = torch.randperm(total_indices, device=device)[:batch_size_bc]
        t_bc = q0_times_t[random_indices].clone().requires_grad_(True)
        return t_bc

    # Ensure n_events doesn't exceed available events
    n_events = min(n_events, len(spike_events))

    # Randomly select n_events spike events (shuffle for variety across epochs)
    if n_events < len(spike_events):
        # Randomly select n_events from all available spike events
        selected_indices = torch.randperm(len(spike_events), device=device)[:n_events]
        spike_events_subset = [spike_events[i] for i in selected_indices.cpu().tolist()]
    else:
        spike_events_subset = spike_events

    # Step 1: Create enhanced sampling distribution with interpolation
    sampling_info = create_interpolated_spike_samples(
        q0_times_t,
        spike_events_subset,
        interpolation_density=interpolation_density,
        neighborhood_expansion=neighborhood_expansion,
        device=device
    )

    original_indices = sampling_info['original_indices']
    original_weights = sampling_info['original_weights']
    interpolated_times = sampling_info['interpolated_times']
    interpolated_weights = sampling_info['interpolated_weights']
    spike_indices = sampling_info['spike_indices']

    # Step 2: Determine sample allocation
    n_spike_samples = int(batch_size_bc * spike_ratio)
    n_baseline_samples = batch_size_bc - n_spike_samples

    # Step 3: Sample from spike regions (original + interpolated points)
    spike_samples = []

    if n_spike_samples > 0 and len(spike_indices) > 0:
        # 3a. Sample from original spike points
        n_original_spike = min(n_spike_samples // 2, len(spike_indices))

        if use_weighted_sampling and len(spike_indices) > 0:
            # Weighted sampling from spike points
            spike_weights = original_weights[spike_indices]
            spike_probs = spike_weights / spike_weights.sum()

            sampled_spike_idx = torch.multinomial(
                spike_probs,
                n_original_spike,
                replacement=(n_original_spike > len(spike_indices))
            )
            original_spike_samples = q0_times_t[spike_indices[sampled_spike_idx]].reshape(-1, 1)
        else:
            # Uniform sampling from spike points
            perm = torch.randperm(len(spike_indices), device=device)[:n_original_spike]
            original_spike_samples = q0_times_t[spike_indices[perm]].reshape(-1, 1)

        spike_samples.append(original_spike_samples)

        # 3b. Sample from interpolated points around spikes
        n_interpolated = n_spike_samples - n_original_spike

        if n_interpolated > 0 and len(interpolated_times) > 0:
            if use_weighted_sampling:
                # Weighted sampling from interpolated points
                interp_probs = interpolated_weights / interpolated_weights.sum()
                sampled_interp_idx = torch.multinomial(
                    interp_probs,
                    min(n_interpolated, len(interpolated_times)),
                    replacement=(n_interpolated > len(interpolated_times))
                )
                interpolated_samples = interpolated_times[sampled_interp_idx].reshape(-1, 1)
            else:
                # Uniform sampling from interpolated points
                perm = torch.randperm(len(interpolated_times), device=device)[:n_interpolated]
                interpolated_samples = interpolated_times[perm].reshape(-1, 1)

            spike_samples.append(interpolated_samples)

    # Step 4: Sample from baseline (non-spike) regions
    baseline_samples = []

    if n_baseline_samples > 0:
        # Identify baseline indices (not in spike regions)
        baseline_mask = ~torch.isin(original_indices, spike_indices)
        baseline_indices = original_indices[baseline_mask]

        if len(baseline_indices) > 0:
            n_baseline_actual = min(n_baseline_samples, len(baseline_indices))

            if use_weighted_sampling:
                # Can use uniform or slightly weighted for diversity
                baseline_weights = original_weights[baseline_indices]
                if baseline_weights.sum() > 0:
                    baseline_probs = baseline_weights / baseline_weights.sum()
                    sampled_baseline_idx = torch.multinomial(
                        baseline_probs,
                        n_baseline_actual,
                        replacement=(n_baseline_actual > len(baseline_indices))
                    )
                else:
                    sampled_baseline_idx = torch.randperm(len(baseline_indices), device=device)[:n_baseline_actual]
            else:
                sampled_baseline_idx = torch.randperm(len(baseline_indices), device=device)[:n_baseline_actual]

            baseline_samples.append(q0_times_t[baseline_indices[sampled_baseline_idx]].reshape(-1, 1))

    # Step 5: Combine all samples
    all_samples = spike_samples + baseline_samples

    if len(all_samples) > 0:
        t_bc = torch.cat(all_samples, dim=0)
    else:
        # Fallback: random sampling
        print("Warning: No samples generated, falling back to uniform sampling")
        random_indices = torch.randperm(len(q0_times_t), device=device)[:batch_size_bc]
        t_bc = q0_times_t[random_indices].clone().reshape(-1, 1)

    # Step 6: Ensure correct batch size (pad or truncate if needed)
    if len(t_bc) < batch_size_bc:
        # Pad with random samples
        shortfall = batch_size_bc - len(t_bc)
        random_indices = torch.randperm(len(q0_times_t), device=device)[:shortfall]
        extra_samples = q0_times_t[random_indices].reshape(-1, 1)
        t_bc = torch.cat([t_bc, extra_samples], dim=0)
    elif len(t_bc) > batch_size_bc:
        # Truncate
        t_bc = t_bc[:batch_size_bc]

    # Step 7: Shuffle and enable gradients (keep 2D shape for model compatibility)
    perm = torch.randperm(len(t_bc), device=device)
    t_bc = t_bc[perm].clone().requires_grad_(True)

    return t_bc


def sample_boundary_points_with_interpolation(
    q0_times_t,
    spike_events,
    n_events,
    batch_size_bc,
    device='cpu',
    # Configuration parameters
    enable_interpolation=True,
    spike_ratio=0.7,
    interpolation_density=3,
    neighborhood_expansion=2,
):
    """
    Wrapper function for backward compatibility.

    This function provides the same interface as the original sample_boundary_points
    but with enhanced interpolation-based sampling capabilities.

    Args:
        q0_times_t: torch.Tensor - boundary condition time points
        spike_events: list of torch.Tensor - detected spike events
        n_events: int - number of spike events to include
        batch_size_bc: int - total number of samples per batch
        device: str or torch.device - device for computation
        enable_interpolation: bool - whether to use interpolation enhancement
        spike_ratio: float - fraction of batch from spike regions
        interpolation_density: int - interpolation point density
        neighborhood_expansion: int - spike region expansion size

    Returns:
        t_bc: torch.Tensor - sampled time points with gradient tracking
    """
    if enable_interpolation:
        return adaptive_boundary_sampling(
            q0_times_t,
            spike_events,
            n_events,
            batch_size_bc,
            device=device,
            spike_ratio=spike_ratio,
            interpolation_density=interpolation_density,
            neighborhood_expansion=neighborhood_expansion,
            use_weighted_sampling=True
        )
    else:
        # Fall back to original implementation (import from boundary_sampling)
        from .boundary_sampling import sample_boundary_points
        return sample_boundary_points(
            q0_times_t,
            spike_events,
            n_events,
            batch_size_bc,
            device=device
        )


def visualize_sampling_distribution(
    q0_times_t,
    q0_values,
    spike_events,
    sampled_times,
    title="Boundary Sampling Distribution"
):
    """
    Visualize the sampling distribution to verify spike-focused sampling.

    Args:
        q0_times_t: torch.Tensor - all time points
        q0_values: torch.Tensor - corresponding flux values
        spike_events: list of torch.Tensor - detected spike event indices
        sampled_times: torch.Tensor - sampled time points
        title: str - plot title
    """
    import matplotlib.pyplot as plt

    # Convert to numpy for plotting
    times_np = q0_times_t.cpu().numpy().flatten()
    values_np = q0_values.cpu().numpy().flatten()
    sampled_np = sampled_times.detach().cpu().numpy().flatten()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Full time series with sampled points
    ax1.plot(times_np, values_np, 'b-', alpha=0.5, linewidth=1.5, label='Flux data')

    # Plot sampled points - interpolate to get their flux values
    import numpy as np
    sampled_values = np.interp(sampled_np, times_np, values_np)
    ax1.scatter(sampled_np, sampled_values, c='green', s=20, alpha=0.6,
               marker='x', label='Sampled BC points', zorder=4)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Flux (m/s)')
    ax1.set_title(f'{title} - Sampled Points on Flux Data')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of sampled points
    ax2.hist(sampled_np, bins=50, alpha=0.6, label=f'Sampled points (n={len(sampled_np)})', color='green', edgecolor='black')
    ax2.axvline(sampled_np.mean(), color='red', linestyle='--',
                label=f'Mean: {sampled_np.mean():.1f}', linewidth=2)

    # Mark spike regions
    for i, event in enumerate(spike_events):
        event_np = event.cpu().numpy()
        spike_times = times_np[event_np]
        ax2.axvspan(spike_times.min(), spike_times.max(),
                   alpha=0.2, color='red', label='Spike region' if i == 0 else '')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Sample count')
    ax2.set_title(f'{title} - Sample Distribution')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
