# Quick Start: Adaptive Boundary Sampling

## What It Does

Enhances PINN boundary condition sampling by focusing on spike events through interpolation and weighted sampling.

## Installation

No installation needed - files are already in the project.

## Quick Test

```bash
# Run test suite to verify everything works
python test_adaptive_sampling.py
```

Expected output: `ALL TESTS PASSED!`

## Minimal Usage Example

```python
import torch
from src.surf_flux import synth_surface_flux
from src.spike_detection import detect_spike_events
from src.adaptive_boundary_sampling import adaptive_boundary_sampling

# 1. Load your boundary condition data
t, q = synth_surface_flux(total_days=15, dt_minutes=30)
q_tensor = torch.tensor(q)
t_tensor = torch.tensor(t).view(-1, 1)

# 2. Detect spikes
spike_events, _ = detect_spike_events(q_tensor, threshold_value=1.0)

# 3. Sample adaptively
t_bc = adaptive_boundary_sampling(
    t_tensor,
    spike_events,
    n_events=9,
    batch_size_bc=200,
    device='cpu',
    spike_ratio=0.7,           # 70% from spikes
    interpolation_density=3,   # 3 interpolated points
    neighborhood_expansion=2   # ±2 point expansion
)

# 4. Use in PINN training
res_surf = model.surface_bc_residual(t_bc)
loss_surf = (res_surf**2).mean()
```

## Integration into Existing Training

### Quick Integration (Recommended)

Edit `src/train_loop.py` line 223:

```python
# BEFORE:
from .boundary_sampling import sample_boundary_points
t_bc = sample_boundary_points(q0_times_t, spike_events, n_events, batch_size_bc, device)

# AFTER:
from .adaptive_boundary_sampling import adaptive_boundary_sampling
t_bc = adaptive_boundary_sampling(
    q0_times_t, spike_events, n_events, batch_size_bc, device,
    spike_ratio=0.7, interpolation_density=3, neighborhood_expansion=2
)
```

### Alternative: Use Wrapper (Backward Compatible)

```python
from .adaptive_boundary_sampling import sample_boundary_points_with_interpolation

t_bc = sample_boundary_points_with_interpolation(
    q0_times_t, spike_events, n_events, batch_size_bc, device,
    enable_interpolation=True  # Set to False to disable
)
```

## Parameter Quick Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `spike_ratio` | 0.7 | 0.5-0.9 | Fraction of samples from spike regions |
| `interpolation_density` | 3 | 2-5 | Number of interpolated points between neighbors |
| `neighborhood_expansion` | 2 | 1-3 | Expand spike regions by N points |
| `use_weighted_sampling` | True | True/False | Enable importance weighting |

## Recommended Configurations

### Default (Most Cases)
```python
spike_ratio=0.7
interpolation_density=3
neighborhood_expansion=2
```

### Conservative (Fewer Spikes)
```python
spike_ratio=0.6
interpolation_density=2
neighborhood_expansion=1
```

### Aggressive (Rare, Critical Spikes)
```python
spike_ratio=0.8
interpolation_density=5
neighborhood_expansion=3
```

## Validation

### Quick Check
```python
from src.adaptive_boundary_sampling import visualize_sampling_distribution

# Visualize where samples are concentrated
fig = visualize_sampling_distribution(t_tensor, q_tensor, spike_events, t_bc)
plt.show()
```

### Quantitative Check
```python
# Compute spike coverage (should be ~70% with spike_ratio=0.7)
spike_times = t_tensor[torch.cat(spike_events).unique()]
coverage = sum(
    min(abs(spike_times - sample).min() for sample in t_bc.flatten())
    for _ in range(len(t_bc))
) / len(t_bc)
print(f"Spike coverage: {coverage*100:.1f}%")
```

## Files Reference

| File | Purpose |
|------|---------|
| `src/adaptive_boundary_sampling.py` | Core implementation |
| `test_adaptive_sampling.py` | Test suite |
| `notebooks/adaptive_sampling_demo.ipynb` | Interactive demo |
| `ADAPTIVE_SAMPLING_GUIDE.md` | Detailed documentation |
| `IMPLEMENTATION_SUMMARY.md` | Technical summary |

## Common Issues

### "No spike events detected"
- Lower `threshold_value` in `detect_spike_events`
- Check if data actually contains spikes

### Spike coverage too low
- Increase `spike_ratio`
- Increase `neighborhood_expansion`
- Verify spike detection is working

### Out of memory
- Reduce `interpolation_density`
- Reduce `batch_size_bc`

## Next Steps

1. ✅ Run `python test_adaptive_sampling.py` to verify installation
2. ✅ Open `notebooks/adaptive_sampling_demo.ipynb` for interactive demo
3. ✅ Read `ADAPTIVE_SAMPLING_GUIDE.md` for detailed documentation
4. ✅ Integrate into your training loop using examples above

## Questions?

- See `ADAPTIVE_SAMPLING_GUIDE.md` for comprehensive documentation
- Check `test_adaptive_sampling.py` for usage examples
- Run `notebooks/adaptive_sampling_demo.ipynb` for visualizations
