# Configuration Files

This directory contains YAML configuration files for PINN training. All hyperparameters are centralized here for easy management.

## Available Configurations

### `baseline.yaml`
Standard configuration for production training.
- **Use case**: Default training runs
- **Cache size**: 80,000 points
- **Batch size**: 4,000 points
- **BC sampling**: Balanced (30% gradient, 70% uniform)
- **Duration**: ~7 days of data

### `fast_test.yaml`
Quick configuration for prototyping and debugging.
- **Use case**: Fast iteration, debugging
- **Cache size**: 10,000 points (smaller)
- **Batch size**: 1,000 points
- **Epochs**: 10,000 (vs 500k)
- **Duration**: 2 days of data

### `aggressive_bc_focus.yaml`
Strong focus on boundary condition fitting.
- **Use case**: Spiky BC data, challenging surface conditions
- **Boundary ratio**: 85% (vs 70%)
- **BC batch size**: 400 (vs 250)
- **Gradient sampling**: 85% interpolated around spikes
- **Temperature**: 0.2 (sharper focus)

### `hpc_fast.yaml`
Optimized for HPC GPU clusters.
- **Use case**: Maximum throughput on multi-GPU systems
- **Mixed precision**: Enabled (30-50% faster)
- **Batch size**: 8,000 (larger for GPU)
- **Gradient accumulation**: 2 steps (effective batch = 16k)
- **Multi-GPU**: Enabled

## Usage

### Basic Usage

```python
from src.config_loader import load_config

# Load config
config = load_config('configs/baseline.yaml')

# Print summary
config.print_summary()

# Access parameters
print(config.cache_size)      # → 80000
print(config.learning_rate)   # → 0.0001
```

### Use in Training Script

```bash
# Run with specific config
python hpc/train_forward_pinn.py --config configs/baseline.yaml

# Quick test
python hpc/train_forward_pinn.py --config configs/fast_test.yaml

# Aggressive BC focus
python hpc/train_forward_pinn.py --config configs/aggressive_bc_focus.yaml
```

### Use in Jupyter Notebook

```python
from src.config_loader import load_config

# Load config
config = load_config('configs/baseline.yaml')

# Now use config properties instead of hardcoded values
model, losses, comps, sl, sc, se = train_pinn_pool_batch_autoweight(
    soil_params=config.soil_params,
    theta0_data=theta0_data,
    Sy=config.Sy,
    zr=config.zr,
    h_net_config=config.h_net_config,
    zb_net_config=config.zb_net_config,
    L=config.L,
    S_max=config.S_max,
    n_epochs=config.n_epochs,
    learning_rate=config.learning_rate,
    zb_initial=config.zb_initial,
    cache_size=config.cache_size,
    batch_size=config.batch_size,
    # ... use config.xxx for all parameters
    device=config.device,
)
```

## Creating Custom Configs

1. Copy an existing config:
   ```bash
   cp configs/baseline.yaml configs/my_experiment.yaml
   ```

2. Edit the parameters you want to change:
   ```yaml
   # configs/my_experiment.yaml
   sampling:
     cache_size: 100000  # Increase cache
     temperature: 0.1    # More aggressive

   boundary_sampling:
     batch_size_bc: 500  # More BC points
   ```

3. Run with your config:
   ```bash
   python hpc/train_forward_pinn.py --config configs/my_experiment.yaml
   ```

## Parameter Categories

### Cache Pool & Batch Sampling
Controls PDE collocation point sampling:
- `cache_size`: Total candidate points (typical: 10k-100k)
- `batch_size`: Points per iteration (typical: 1k-8k)
- `resample_freq`: Update frequency in epochs (typical: 50-200)
- `boundary_ratio`: Fraction near boundaries (0-1)
- `high_residual_ratio`: Fraction from high residuals (0-1)
- `temperature`: Softmax sharpness (lower = more aggressive)

### Boundary Sampling
Controls BC enforcement at surface:
- `batch_size_bc`: Dedicated BC points (typical: 100-500)
- `interp_ratio`: Fraction interpolated around spikes (0-1)
- `neighbor_ratio`: Fraction from spike neighbors (0-1)
- `baseline_ratio`: Fraction uniform (0-1)
- `gradient_threshold`: Percentile for "high gradient" (0-1)
- `power`: Gradient emphasis (>1 = more aggressive)

### Optimization
- `weight_update_freq`: Weight adaptation (1e10 = fixed)
- `use_amp`: Mixed precision (faster on GPU)
- `grad_accumulation_steps`: Effective batch scaling

## Tips

**For faster convergence:**
- Increase `boundary_ratio` (0.7 → 0.85)
- Increase `batch_size_bc` (250 → 400)
- Decrease `temperature` (0.3 → 0.2)

**For stability:**
- Decrease `high_residual_ratio` (0.6 → 0.5)
- Increase `temperature` (0.3 → 0.5)
- Use larger `baseline_ratio` in boundary sampling

**For speed:**
- Use `fast_test.yaml` or `hpc_fast.yaml`
- Enable `use_amp: true`
- Increase `batch_size` if GPU memory allows
