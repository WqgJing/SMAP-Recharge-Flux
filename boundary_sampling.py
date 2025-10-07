import torch


def sample_boundary_points(q0_times_t, spike_events, n_events, batch_size_bc, device='cpu'):
    """
    Sample boundary condition points with event-based sampling.
    
    Args:
        q0_times_t: torch.Tensor - the data values
        batch_size: int - total number of samples per batch
        spike_events: list of torch.Tensor - detected spike events (from detect_spike_events)
        n_events: int - number of spike events to include in each batch
        device: str or torch.device - device for computation
    
    Returns:
        t_bc: torch.Tensor - sampled values with gradient tracking
    """
    batch_size = batch_size_bc
    # Get total number of indices
    total_indices = len(q0_times_t)
    all_indices = torch.arange(total_indices, device=device)
    
    # Handle case where no spike events were detected
    if len(spike_events) == 0:
        print("Warning: No spike events detected, using uniform random sampling")
        random_indices = torch.randperm(total_indices, device=device)[:batch_size]
        t_bc = q0_times_t[random_indices].clone().requires_grad_(True)
        return t_bc
    
    # Ensure n_events doesn't exceed available events
    n_events = min(n_events, len(spike_events))
    
    # Step 1: Randomly select n spike events
    event_indices_to_sample = torch.randperm(len(spike_events), device=device)[:n_events]
    
    # Step 2: Collect all indices from selected events
    event_samples_list = []
    for event_idx in event_indices_to_sample:
        event = spike_events[event_idx.item()]
        # Ensure event is on the correct device
        if event.device != device:
            event = event.to(device)
        event_samples_list.append(event)
    
    # Combine all event indices
    if event_samples_list:
        event_samples = torch.cat(event_samples_list)  # Remove any duplicates
    else:
        event_samples = torch.tensor([], dtype=torch.long, device=device)
    
    num_event_samples = len(event_samples)
    
    # Step 3: Define baseline indices (all indices NOT in ANY spike event)
    # First, get all indices that belong to any spike event
    all_event_indices_list = []
    for event in spike_events:
        if event.device != device:
            event = event.to(device)
        all_event_indices_list.append(event)
    
    all_event_indices = torch.cat(all_event_indices_list).unique()
    
    # Baseline = complement of all event indices
    baseline_mask = ~torch.isin(all_indices, all_event_indices)
    baseline_indices = all_indices[baseline_mask]
    
    # Step 4: Calculate how many baseline samples we need
    baseline_sample_size = batch_size - num_event_samples
    
    # Handle edge case: if events are too large
    if baseline_sample_size < 0:
        print(f"Warning: {n_events} events contain {num_event_samples} indices, "
              f"exceeding batch_size {batch_size}. Truncating event samples.")
        # Randomly sample from event indices to fit batch_size
        event_perm = torch.randperm(num_event_samples, device=device)[:batch_size]
        final_indices = event_samples[event_perm]
    else:
        # Step 5: Sample from baseline
        if len(baseline_indices) > 0 and baseline_sample_size > 0:
            actual_baseline_sample = min(baseline_sample_size, len(baseline_indices))
            baseline_perm = torch.randperm(len(baseline_indices), device=device)[:actual_baseline_sample]
            baseline_samples = baseline_indices[baseline_perm]
            
            # Combine event samples and baseline samples
            final_indices = torch.cat([event_samples, baseline_samples])
            
            # Handle shortfall if not enough baseline indices
            if actual_baseline_sample < baseline_sample_size:
                shortfall = baseline_sample_size - actual_baseline_sample
                print(f"Warning: Not enough baseline indices. Short by {shortfall} samples. "
                      f"Sampling with replacement from baseline.")
                # Sample with replacement to fill the gap
                extra_samples = baseline_indices[
                    torch.randint(0, len(baseline_indices), (shortfall,), device=device)
                ]
                final_indices = torch.cat([final_indices, extra_samples])
        else:
            # No baseline available or not needed
            final_indices = event_samples
            if baseline_sample_size > 0:
                print(f"Warning: No baseline indices available. Using only event samples.")
    
    # Step 6: Shuffle the final indices to mix events and baseline
    final_indices = final_indices[torch.randperm(len(final_indices), device=device)]
    
    # Step 7: Ensure exactly batch_size samples (final safety check)
    if len(final_indices) > batch_size:
        final_indices = final_indices[:batch_size]
    elif len(final_indices) < batch_size:
        # This should rarely happen after above logic, but just in case
        shortfall = batch_size - len(final_indices)
        print(f"Warning: Final shortfall of {shortfall} samples. Padding with random samples.")
        available_mask = ~torch.isin(all_indices, final_indices)
        available_indices = all_indices[available_mask]
        if len(available_indices) > 0:
            extra_samples = available_indices[
                torch.randint(0, len(available_indices), (shortfall,), device=device)
            ]
            final_indices = torch.cat([final_indices, extra_samples])
        else:
            # Ultimate fallback: sample with replacement from what we have
            extra_samples = final_indices[
                torch.randint(0, len(final_indices), (shortfall,), device=device)
            ]
            final_indices = torch.cat([final_indices, extra_samples])
    
    # Step 8: Get the actual values and return with gradient tracking
    t_bc = q0_times_t[final_indices].clone().requires_grad_(True)
    
    return t_bc
