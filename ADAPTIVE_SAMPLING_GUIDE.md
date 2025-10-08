# Adaptive Boundary Sampling with Spike Interpolation

## Overview

This document describes the enhanced boundary condition sampling strategy implemented for Physics-Informed Neural Networks (PINNs) that focuses on accurately capturing spike events through interpolation-based non-uniform sampling.

## Motivation

In PINN training for Richards equation with time-varying boundary conditions (e.g., rainfall events), spike events represent critical physical phenomena that require higher sampling density to capture accurately. Traditional uniform or simple event-based sampling may miss important dynamics, especially for:

- **Single-point spikes**: Isolated rainfall events captured at a single time step
- **Sparse spike events**: Rapid transitions in boundary conditions
- **Near-spike regions**: The critical transition zones around spike events

## Implementation

### Core Components

The adaptive sampling system consists of three main modules:

1. **Spike Detection** (`src/spike_detection.py`)
   - Detects spike events using statistical methods (z-score, percentile, IQR)
   - Expands spike regions to capture event neighborhoods
   - Merges nearby events to avoid fragmentation

2. **Interpolation Generation** (`src/adaptive_boundary_sampling.py::create_interpolated_spike_samples`)
   - Creates additional sample points through linear interpolation
   - Assigns importance weights based on proximity to spikes
   - Expands spike neighborhoods to include transition regions

3. **Adaptive Sampling** (`src/adaptive_boundary_sampling.py::adaptive_boundary_sampling`)
   - Implements weighted sampling strategy
   - Allocates samples between spike and baseline regions
   - Maintains computational efficiency through configurable parameters

### Key Features

#### 1. Spike Neighborhood Expansion

Detected spike events are expanded by `neighborhood_expansion` points on each side to capture the full event dynamics:

```python
# Example: Spike at index 100 with expansion=2
# Original: [100]
# Expanded: [98, 99, 100, 101, 102]
```

#### 2. Interpolation-Based Sample Generation

Between adjacent points in spike regions, additional interpolated time points are created:

```python
# Example: Adjacent spike indices [100, 101] with interpolation_density=3
# Original times: [t_100, t_101]
# Interpolated: [t_100, t_100.25, t_100.5, t_100.75, t_101]
```

#### 3. Non-Uniform Weighted Sampling

Sampling weights are assigned based on point classification:
- **Actual spike points**: 3x weight
- **Interpolated spike points**: 2x weight
- **Baseline points**: 1x weight

#### 4. Configurable Sample Allocation

The `spike_ratio` parameter controls the fraction of samples from spike regions:
- `spike_ratio=0.7`: 70% from spikes, 30% from baseline
- Ensures both critical events and background conditions are represented

## Usage

### Basic Usage

```python
from src.adaptive_boundary_sampling import adaptive_boundary_sampling
from src.spike_detection import detect_spike_events

# 1. Detect spike events
spike_events, _ = detect_spike_events(
    q_values,
    threshold_method='std',
    threshold_value=1.0,
    expansion_window=5,
    merge_distance=3,
    device=device
)

# 2. Sample with adaptive strategy
t_bc = adaptive_boundary_sampling(
    q0_times_t,              # Time points tensor (N, 1)
    spike_events,            # Detected spike events
    n_events=9,              # Number of events to include
    batch_size_bc=200,       # Batch size
    device=device,
    # Adaptive sampling parameters
    spike_ratio=0.7,         # 70% from spike regions
    interpolation_density=3, # 3 interpolated points between neighbors
    neighborhood_expansion=2,# Expand spike regions by 2 points
    use_weighted_sampling=True
)
```

### Integration with PINN Training

#### Option 1: Modify Training Loop Directly

Edit `src/train_loop.py`:

```python
# Before (line 223):
from .boundary_sampling import sample_boundary_points
t_bc = sample_boundary_points(q0_times_t, spike_events, n_events, batch_size_bc, device)

# After:
from .adaptive_boundary_sampling import adaptive_boundary_sampling
t_bc = adaptive_boundary_sampling(
    q0_times_t, spike_events, n_events, batch_size_bc, device,
    spike_ratio=0.7,
    interpolation_density=3,
    neighborhood_expansion=2
)
```

#### Option 2: Use Backward-Compatible Wrapper

```python
from src.adaptive_boundary_sampling import sample_boundary_points_with_interpolation

t_bc = sample_boundary_points_with_interpolation(
    q0_times_t,
    spike_events,
    n_events,
    batch_size_bc,
    device,
    enable_interpolation=True,  # Toggle on/off
    spike_ratio=0.7,
    interpolation_density=3,
    neighborhood_expansion=2
)
```

### Parameter Tuning Guide

| Parameter | Range | Recommended | Effect |
|-----------|-------|-------------|--------|
| `spike_ratio` | 0.5-0.9 | 0.7 | Higher values focus more on spikes but may miss baseline dynamics |
| `interpolation_density` | 2-5 | 3 | Higher values create more interpolated points but increase memory |
| `neighborhood_expansion` | 1-3 | 2 | Larger neighborhoods capture more context but may over-sample |
| `use_weighted_sampling` | True/False | True | Enables importance sampling for better spike coverage |

#### Tuning for Different Scenarios

**Scenario 1: Frequent, Short Spikes**
```python
spike_ratio=0.6
interpolation_density=2
neighborhood_expansion=1
```

**Scenario 2: Rare, Intense Spikes** (recommended)
```python
spike_ratio=0.7
interpolation_density=3
neighborhood_expansion=2
```

**Scenario 3: Very Sparse Spikes**
```python
spike_ratio=0.8
interpolation_density=5
neighborhood_expansion=3
```

## Validation and Testing

### Running Tests

Execute the test suite to validate the implementation:

```bash
python test_adaptive_sampling.py
```

Expected output:
```
✓ Interpolated Sample Generation: PASSED
✓ Adaptive Sampling Distribution: PASSED
✓ Enhanced vs Original Comparison: PASSED
✓ Edge Case Handling: PASSED

ALL TESTS PASSED!
```

### Visualization

Use the provided visualization function to verify sampling distribution:

```python
from src.adaptive_boundary_sampling import visualize_sampling_distribution

fig = visualize_sampling_distribution(
    q0_times_t,
    q0_values,
    spike_events,
    sampled_times,
    title="Adaptive Sampling Distribution"
)
plt.show()
```

This generates a two-panel plot showing:
1. **Top panel**: Full time series with spike events highlighted
2. **Bottom panel**: Histogram of sampled points overlaid with spike regions

### Quantitative Metrics

Compute spike coverage to quantify sampling effectiveness:

```python
# Count samples near spike regions
spike_times = q0_times_t[torch.cat(spike_events).unique()]
dt_mean = (q0_times_t[1:] - q0_times_t[:-1]).mean()
tolerance = 2.0 * dt_mean

n_near_spike = 0
for sample in sampled_times.flatten():
    if torch.abs(spike_times - sample).min() <= tolerance:
        n_near_spike += 1

spike_coverage = n_near_spike / len(sampled_times)
print(f"Spike coverage: {spike_coverage*100:.1f}%")
```

Expected coverage with `spike_ratio=0.7`: approximately 65-75%

## Performance Considerations

### Computational Complexity

- **Interpolation generation**: O(n_spike_points × interpolation_density)
- **Weighted sampling**: O(batch_size × log(n_points))
- **Memory overhead**: Minimal (temporary interpolated tensors)

### Efficiency Tips

1. **Cache interpolated samples** if using fixed spike events across epochs
2. **Adjust batch sizes** to balance between spike coverage and diversity
3. **Use GPU tensors** for large datasets (automatically supported)

### Memory Usage

For typical PINN training:
- Original time points: 500-1000 points
- Interpolated points: ~50-200 points (depends on density)
- Additional memory: < 1 MB for float32 tensors

## Implementation Details

### Spike Event Structure

Spike events are represented as a list of index tensors:

```python
spike_events = [
    torch.tensor([95, 96, 97, 98, 99, 100]),  # Event 1
    torch.tensor([250, 251, 252]),             # Event 2
    torch.tensor([420]),                       # Event 3 (single point)
]
```

### Interpolation Method

Linear interpolation is used between adjacent spike points:

```python
t_interp = t_start + alpha * (t_end - t_start)
# where alpha ∈ [0, 1] with (interpolation_density + 2) points total
```

The PINN model's `surface_flux_tilde` method automatically handles interpolation when evaluating boundary conditions at these intermediate time points.

### Weight Assignment Logic

```python
# Original points
original_weights = torch.ones(n_points)
original_weights[spike_indices] = 3.0  # 3x for actual spikes

# Interpolated points
interpolated_weights = torch.ones(n_interp) * 2.0  # 2x for interpolated

# Sampling probabilities
probs = weights / weights.sum()
sampled_idx = torch.multinomial(probs, batch_size, replacement=True)
```

## Examples

### Example 1: Compare Sampling Strategies

See `notebooks/adaptive_sampling_demo.ipynb` for a complete example comparing original vs enhanced sampling.

### Example 2: Integration in Training

```python
# In your training loop
for epoch in range(n_epochs):
    # ... PDE sampling ...

    # Enhanced boundary sampling
    t_bc = adaptive_boundary_sampling(
        q0_times_t, spike_events, n_events=9, batch_size_bc=200,
        device=device, spike_ratio=0.7, interpolation_density=3
    )

    # Compute boundary residuals (automatically interpolates)
    res_surf = model.surface_bc_residual(t_bc)
    loss_surf = (res_surf**2).mean()

    # ... rest of training ...
```

### Example 3: Sensitivity Analysis

```python
# Test different configurations
configs = [
    {'spike_ratio': 0.5, 'interpolation_density': 2},
    {'spike_ratio': 0.7, 'interpolation_density': 3},
    {'spike_ratio': 0.8, 'interpolation_density': 5},
]

for config in configs:
    t_bc = adaptive_boundary_sampling(
        q0_times_t, spike_events, n_events=9, batch_size_bc=200,
        device=device, **config
    )
    # Evaluate performance...
```

## Troubleshooting

### Issue: "No spike events detected"

**Cause**: Spike detection threshold too high or data doesn't contain spikes

**Solution**:
- Lower `threshold_value` in `detect_spike_events`
- Check data with visualization
- Verify data contains actual spike events

### Issue: Sampling coverage too low/high

**Cause**: Incorrect parameter tuning

**Solution**:
- Adjust `spike_ratio` to control allocation
- Increase `neighborhood_expansion` for broader coverage
- Verify spike detection is working correctly

### Issue: Out of memory errors

**Cause**: Too many interpolated points

**Solution**:
- Reduce `interpolation_density`
- Reduce `batch_size_bc`
- Process in smaller chunks

## Future Enhancements

Potential improvements for future versions:

1. **Adaptive interpolation density**: Vary interpolation based on spike intensity
2. **Multi-scale interpolation**: Different densities for different spike magnitudes
3. **Temporal correlation**: Account for temporal dependencies in sampling
4. **Dynamic adjustment**: Adapt parameters during training based on residuals

## References

- **Original boundary sampling**: `src/boundary_sampling.py`
- **Spike detection**: `src/spike_detection.py`
- **PINN model**: `src/pinn_models.py` (see `surface_flux_tilde` method)
- **Training loop**: `src/train_loop.py`

## Authors and Acknowledgments

Enhanced adaptive sampling implementation for SMAP Recharge Flux PINN project.

For questions or issues, please refer to the test suite (`test_adaptive_sampling.py`) or demonstration notebook (`notebooks/adaptive_sampling_demo.ipynb`).
