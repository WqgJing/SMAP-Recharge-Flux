# PINN Unified API Guide 🚀

## Overview

We've implemented a **unified, YAML-driven architecture** that simplifies PINN training and visualization. Everything is now controlled through configuration files - no more passing 30+ arguments to functions!

---

## Key Benefits

✅ **Ultra-simple**: 3 lines of code to train
✅ **YAML-driven**: All parameters in config files
✅ **Dataset-agnostic**: Works with ANY data format
✅ **Automatic depth handling**: 3, 5, or 10 observation depths
✅ **Optional WTD**: Adds validation automatically if available
✅ **Clean code**: No more argument hell
✅ **Type-safe**: Dataset validates your config

---

## Quick Start

### Old Way (100+ lines):
```python
# Load config
config = load_config('configs/baseline.yaml')

# Preprocess data
preprocessed = preprocess_soil_data(config)

# Extract 20+ variables
theta0_times = preprocessed['theta0_times']
theta0_values = preprocessed['theta0_values']
ic_profile = preprocessed['ic_profile']
# ... extract more data ...

# Train with 30+ arguments
model = train_pinn_pool_batch_autoweight(
    soil_params=config.soil_params,
    theta0_data=(theta0_times, theta0_values),
    Sy=config.Sy,
    zr=config.zr,
    # ... 25+ more arguments ...
)

# Manually prepare obs_data
obs_data = {
    'times': times_seconds,
    'depths': [-0.02, -0.15, -0.30, -0.40, -0.60, -0.80],
    'theta': [theta_2cm, theta_15cm, ...]
}

plot_comprehensive_results(model, theta0_data, soil_params, obs_data=obs_data)
```

### New Way (3 lines):
```python
# Load dataset (everything in one object!)
dataset = PINNDataset('configs/baseline.yaml')

# Train (one line!)
model, losses, comps, *_ = train_pinn(dataset)

# Visualize (automatic!)
plot_results(model, dataset)
```

---

## Architecture

### 1. **PINNDataset Class** (`src/dataset.py`)

Central data container that loads everything from YAML:

```python
from src.dataset import PINNDataset

# Initialize from config
dataset = PINNDataset('configs/baseline.yaml')

# Access any data
print(dataset.obs_depths)          # ['2cm', '15cm', '30cm', ...]
print(dataset.bc_times)            # Boundary condition times
print(dataset.soil_params)         # van Genuchten parameters
print(dataset.has_wtd())           # Check if WTD available
```

**Attributes:**
- `bc_times`, `bc_values`, `bc_type` - Boundary condition
- `ic_profile`, `ic_type` - Initial condition
- `obs_times`, `obs_depths`, `obs_theta` - Observations
- `wtd_times`, `wtd_values` - Water table depth (optional)
- `soil_params` - Soil parameters
- `Sy`, `zr`, `L`, `S_max`, `zb_initial` - Physics
- `n_epochs`, `learning_rate`, `device` - Training
- All sampling and optimization parameters

**Methods:**
- `get_bc_data()` → `(times, values)` tuple
- `get_obs_data_for_viz()` → dict for plotting
- `get_wtd_data_for_viz()` → WTD tuple or None
- `has_wtd()` → bool
- `has_obs_at_depth(depth)` → bool

### 2. **Unified Training API** (`src/train_loop.py`)

Simple training functions that accept `PINNDataset`:

```python
from src.train_loop import train_pinn, finetune_pinn_with_dataset

# Forward training
model, losses, comps, sample_losses, sample_comps, sample_epochs = train_pinn(
    dataset,
    device='auto',  # or 'cuda', 'cpu'
    checkpoint_dir=None  # optional override
)

# Fine-tuning
model_ft, *_ = finetune_pinn_with_dataset(
    base_checkpoint_path='checkpoint.pt',
    new_dataset=finetune_dataset,
    device='auto'
)
```

### 3. **Unified Visualization API** (`src/visualization.py`)

Simple plotting functions:

```python
from src.visualization import plot_results, plot_losses

# Plot comprehensive results (automatic depth handling, WTD overlay, etc.)
plot_results(model, dataset, device='cpu')

# Plot training losses
plot_losses(losses, comps, sample_losses, sample_comps, sample_epochs)
```

---

## Usage Examples

### Example 1: Train with Calhoun Data

```python
from src.dataset import PINNDataset
from src.train_loop import train_pinn
from src.visualization import plot_results

# Load Calhoun dataset (6 depths, no WTD)
dataset = PINNDataset('configs/baseline.yaml')

# Train
model, losses, comps, *_ = train_pinn(dataset, device='cuda')

# Visualize
plot_results(model, dataset, device='cuda')
```

### Example 2: Train with US-Uaf Data

```python
# Load US-Uaf dataset (5 depths, with WTD if configured)
dataset = PINNDataset('configs/us_uaf_2019.yaml')

# Everything else is THE SAME!
model, losses, comps, *_ = train_pinn(dataset, device='cuda')
plot_results(model, dataset, device='cuda')
```

### Example 3: Fine-Tuning

```python
# Base training
base_dataset = PINNDataset('configs/baseline.yaml')
model_base, *_ = train_pinn(base_dataset)

# Fine-tuning on different period
finetune_dataset = PINNDataset('configs/finetune_calhoun.yaml')
model_ft, *_ = finetune_pinn_with_dataset(
    'checkpoint_final.pt',
    finetune_dataset
)
```

### Example 4: Command Line Training

```bash
# New simple script
python hpc/train_simple.py --config configs/baseline.yaml

# Works with ANY config!
python hpc/train_simple.py --config configs/us_uaf_2019.yaml
```

### Example 5: Jupyter Notebook

See `notebooks/train_simple.ipynb` for the new simplified notebook.

---

## YAML Configuration

Your existing configs already work! The `column_mapping` section defines the data format:

```yaml
data:
  path: 'data/US-Uaf_2019_May-Sep_LE_SWC_WTD.xlsx'
  start_date: '2019-07-30'
  end_date: '2019-08-13'
  interpolate: true
  max_gap_hours: 6

  column_mapping:
    datetime: 'datetime'
    multi_level_header: false
    unit_conversion: 0.01  # percentage to fraction

    # Observation depths (first = surface BC)
    depths:
      2cm: 'SWC_1_2_1'   # Surface BC
      15cm: 'SWC_1_1_1'
      30cm: 'SWC_2_1_1'
      40cm: 'SWC_2_2_1'
      60cm: 'SWC_2_3_1'

    # Optional: Water table depth
    wtd: 'WTD_1_1_1'
    wtd_sign: -1.0  # Sign conversion

soil:
  theta_s: 0.46
  theta_r: 0.05
  # ... etc
```

---

## File Structure

### New Files:
- `src/dataset.py` - PINNDataset class
- `hpc/train_simple.py` - Simple training script
- `notebooks/train_simple.ipynb` - Simple notebook
- `UNIFIED_API_GUIDE.md` - This file

### Modified Files:
- `src/train_loop.py` - Added `train_pinn()`, `finetune_pinn_with_dataset()`
- `src/visualization.py` - Added `plot_results()`, `plot_losses()`

### Unchanged (Backward Compatible):
- `hpc/train_with_config.py` - Still works!
- `notebooks/forward_pinn.ipynb` - Still works!
- `notebooks/finetune_example.ipynb` - Still works!
- `src/data_loader.py` - Still works!
- `src/data_preprocessing.py` - Still works!

---

## Migration Guide

### For Existing Scripts:

**Option 1: Keep using old API** (fully supported)
```python
# Your existing code continues to work!
config = load_config('configs/baseline.yaml')
preprocessed = preprocess_soil_data(config)
model = train_pinn_pool_batch_autoweight(...)
```

**Option 2: Migrate to new API** (recommended)
```python
# Replace everything with:
dataset = PINNDataset('configs/baseline.yaml')
model, *_ = train_pinn(dataset)
plot_results(model, dataset)
```

### For New Scripts:

Always use the new API! It's simpler and more maintainable.

---

## Testing

Both configs tested and working:

```bash
# Test Calhoun data (6 depths, no WTD)
python3 -c "
from src.dataset import PINNDataset
dataset = PINNDataset('configs/baseline.yaml', verbose=False)
print(f'✓ Loaded {len(dataset.obs_depths)} depths: {dataset.obs_depths}')
print(f'✓ BC points: {len(dataset.bc_times)}')
print(f'✓ Has WTD: {dataset.has_wtd()}')
"

# Test US-Uaf data (5 depths, WTD configurable)
python3 -c "
from src.dataset import PINNDataset
dataset = PINNDataset('configs/us_uaf_2019.yaml', verbose=False)
print(f'✓ Loaded {len(dataset.obs_depths)} depths: {dataset.obs_depths}')
print(f'✓ BC points: {len(dataset.bc_times)}')
"
```

---

## Benefits Summary

### Before:
- ❌ 100+ lines of boilerplate code
- ❌ 30+ function arguments to remember
- ❌ Manual data extraction and formatting
- ❌ Hardcoded depth assumptions
- ❌ Difficult to switch datasets
- ❌ Error-prone obs_data dict construction

### After:
- ✅ 3 lines of code
- ✅ Zero manual arguments (all from YAML)
- ✅ Automatic data handling
- ✅ Works with any number of depths
- ✅ Switch datasets by changing config file
- ✅ Type-safe, validated data container

---

## Next Steps

1. **Try the new notebook**: `notebooks/train_simple.ipynb`
2. **Run simple training**: `python hpc/train_simple.py --config configs/baseline.yaml`
3. **Create your own config**: Copy `baseline.yaml` and modify
4. **Migrate existing scripts**: Replace old API calls with new ones

---

## Troubleshooting

### FileNotFoundError when loading dataset

If you get `FileNotFoundError` when running from a notebook:

```python
# ✅ CORRECT - Use relative path from notebook
dataset = PINNDataset('../configs/baseline.yaml')

# ❌ WRONG - Don't use path from project root
dataset = PINNDataset('configs/baseline.yaml')  # Won't work from notebooks/
```

The `PINNDataset` class automatically adjusts data paths when running from notebooks, but you need to give it the correct config path relative to your notebook location.

---

## FAQ

**Q: Do I need to change my existing code?**
A: No! Old API is fully supported. But new code should use the new API.

**Q: What if my data has different columns?**
A: Just update the `column_mapping` section in your YAML config!

**Q: Can I still use WTD validation?**
A: Yes! Add `wtd: 'your_column'` and `wtd_sign: ±1` to column_mapping.

**Q: Does this work with any number of observation depths?**
A: Yes! 1, 3, 5, 10, doesn't matter. Automatic handling.

**Q: What about fine-tuning?**
A: Use `finetune_pinn_with_dataset()` - same simple API.

**Q: Can I override parameters?**
A: Yes! Most functions accept optional overrides (e.g., `device`, `checkpoint_dir`).

---

## Support

- **Documentation**: This file + docstrings in code
- **Examples**: `notebooks/train_simple.ipynb`
- **Issues**: Check function docstrings or ask questions

---

**Last Updated**: 2025-01-28
**Version**: 1.0.0
**Status**: Production Ready ✅
