import torch


def detect_spike_events(q_actual_flux, threshold_method='std', threshold_value=3.0, 
                       expansion_window=5, merge_distance=3, min_event_size=2, device='cpu'):
    """
    Detect spike events (spike + surrounding shape) in time series data.
    
    Args:
        q_actual_flux: torch.Tensor, shape (N,) - the actual data values
        threshold_method: str - 'std' (z-score), 'percentile', or 'iqr'
        threshold_value: float - threshold for spike detection
            - For 'std': number of standard deviations (e.g., 3.0)
            - For 'percentile': percentile value (e.g., 99.0)
            - For 'iqr': IQR multiplier (e.g., 1.5)
        expansion_window: int - how many points to expand around each spike
        merge_distance: int - merge events if gaps between them <= this value
        min_event_size: int - minimum size of an event to keep
        device: str or torch.device - device for computation
    
    Returns:
        spike_events: list of torch.Tensor - each tensor contains indices of one spike event
        spike_cores: list of torch.Tensor - the original spike indices (before expansion)
    """
    n = len(q_actual_flux)
    
    # Step 1: Detect spike points using the chosen method
    if threshold_method == 'std':
        # Z-score method
        mean = q_actual_flux.mean()
        std = q_actual_flux.std()
        z_scores = torch.abs((q_actual_flux - mean) / (std + 1e-8))
        spike_mask = z_scores > threshold_value
        
    elif threshold_method == 'percentile':
        # Percentile method
        threshold = torch.quantile(q_actual_flux, threshold_value / 100.0)
        spike_mask = q_actual_flux > threshold
        
    elif threshold_method == 'iqr':
        # IQR method (outlier detection)
        q1 = torch.quantile(q_actual_flux, 0.25)
        q3 = torch.quantile(q_actual_flux, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold_value * iqr
        upper_bound = q3 + threshold_value * iqr
        spike_mask = (q_actual_flux < lower_bound) | (q_actual_flux > upper_bound)
    
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")
    
    # Get spike indices
    spike_core_indices = torch.where(spike_mask)[0]
    
    if len(spike_core_indices) == 0:
        print("No spikes detected!")
        return [], []
    
    print(f"Detected {len(spike_core_indices)} initial spike points")
    
    # Step 2: Expand around each spike to capture the event shape
    expanded_regions = []
    for spike_idx in spike_core_indices:
        start = max(0, spike_idx - expansion_window)
        end = min(n - 1, spike_idx + expansion_window)
        region = torch.arange(start, end + 1)
        expanded_regions.append(region)
    
    # Step 3: Merge overlapping or nearby regions
    if len(expanded_regions) == 0:
        return [], []
    
    # Sort regions by start index
    expanded_regions = sorted(expanded_regions, key=lambda x: x[0].item())
    
    merged_events = []
    spike_cores = []
    
    current_event = expanded_regions[0]
    current_cores = [spike_core_indices[0]]
    
    for i in range(1, len(expanded_regions)):
        next_region = expanded_regions[i]
        gap = next_region[0] - current_event[-1]
        
        if gap <= merge_distance:
            # Merge: combine regions
            all_indices = torch.cat([current_event, next_region])
            current_event = torch.unique(all_indices, sorted=True)
            current_cores.append(spike_core_indices[i])
        else:
            # Save current event and start new one
            if len(current_event) >= min_event_size:
                merged_events.append(current_event)
                spike_cores.append(torch.tensor(current_cores, device=device))
            current_event = next_region
            current_cores = [spike_core_indices[i]]
    
    # Don't forget the last event
    if len(current_event) >= min_event_size:
        merged_events.append(current_event)
        spike_cores.append(torch.tensor(current_cores, device=device))
    
    print(f"Merged into {len(merged_events)} spike events")
    total_spike_indices = torch.cat(merged_events) if merged_events else torch.tensor([], device=device, dtype=torch.long)
    print(f"Total indices in merged events: {total_spike_indices.numel()}")
    return merged_events, total_spike_indices
