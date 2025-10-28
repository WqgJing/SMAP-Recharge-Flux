# PINN System Overview - Complete Reference for AI Assistants & Developers

**Purpose:** This document provides complete context for Claude Code (or any AI assistant/developer) to understand the entire PINN training system and provide effective help.

**Last Updated:** 2025-01-28
**Status:** Production Ready ✅

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Key Design Decisions](#key-design-decisions)
3. [File Structure & Responsibilities](#file-structure--responsibilities)
4. [Data Flow](#data-flow)
5. [API Overview](#api-overview)
6. [Critical Implementation Details](#critical-implementation-details)
7. [Common User Tasks](#common-user-tasks)
8. [Troubleshooting Reference](#troubleshooting-reference)
9. [Future Extension Points](#future-extension-points)

---

## System Architecture

### **High-Level Overview**

This is a **Physics-Informed Neural Network (PINN)** system for solving the **Richards equation** (variably saturated subsurface flow). The system:

- Solves for pressure head h(z,t) and water table depth zb(t)
- Uses soil moisture observations as surface boundary condition
- Supports multiple observation depths for validation
- Works with heterogeneous datasets via YAML-driven configuration

### **Core Architecture Pattern: Unified Dataset Container**

```
┌─────────────────────────────────────────────────────────┐
│                    YAML Config File                     │
│  (All parameters, data columns, hyperparameters)        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   PINNDataset Class                     │
│  • Loads data from any Excel format                     │
│  • Extracts BC, IC, observations, WTD                   │
│  • Stores all parameters                                │
│  • Provides clean API for training/visualization        │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  train_pinn()   │    │  plot_results() │
│  (1 function)   │    │  (1 function)   │
└─────────────────┘    └─────────────────┘
```

**Key Innovation:** Everything flows through one `PINNDataset` object. No more passing 30+ arguments!

---

## Key Design Decisions

### **1. YAML-Driven Configuration (Why)**

**Problem:** Different research sites have different data formats (column names, units, header structures).

**Old Solution:** Hardcoded column names in Python → Required code changes for each site.

**New Solution:** YAML `column_mapping` section specifies data format → No code changes needed.

**Example:**
```yaml
column_mapping:
  datetime: 'Date'  # or ['Unnamed: 0', 'DateTime'] for multi-level
  depths:
    2cm: 'SWC_1_2_1'  # Maps generic name → actual column
    15cm: 'SWC_1_1_1'
```

### **2. First Depth = Surface BC (Convention)**

**Critical Convention:** The **first depth listed** in `column_mapping.depths` is automatically used as the surface boundary condition.

```yaml
depths:
  2cm: 'sensor_A'    # ← This becomes surface BC θ(z=0,t)
  15cm: 'sensor_B'   # ← Observation only (validation)
  30cm: 'sensor_C'   # ← Observation only (validation)
```

**Why:** Surface BC is required for training. Other depths are optional (validation only).

**Implementation:** `src/dataset.py:173` → `surface_depth = self.obs_depths[0]`

### **3. Automatic Depth Adaptation (Flexibility)**

**Problem:** Calhoun has 6 depths, US-Uaf has 5, custom sites have 3-10.

**Solution:** Code iterates over `dataset.obs_depths` (from YAML) instead of hardcoding 6 depths.

**Result:**
- Calhoun config → 12 lines in validation plot (6 obs + 6 modeled)
- US-Uaf config → 10 lines in validation plot (5 obs + 5 modeled)
- Custom config → Adapts automatically

### **4. Backward Compatibility (Migration Strategy)**

**New API exists alongside old API:**

```python
# Old API (still works!)
config = load_config('config.yaml')
preprocessed = preprocess_soil_data(config)
model = train_pinn_pool_batch_autoweight(
    soil_params=..., theta0_data=..., # ... 30 more args
)

# New API (recommended)
dataset = PINNDataset('config.yaml')
model, *_ = train_pinn(dataset)
```

**Why:** Allows gradual migration without breaking existing workflows.

---

## File Structure & Responsibilities

### **Core Files (Must Understand)**

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `src/dataset.py` | **Central data container** | `PINNDataset` class |
| `src/train_loop.py` | Training functions | `train_pinn()`, `train_pinn_pool_batch_autoweight()` |
| `src/visualization.py` | Plotting functions | `plot_results()`, `plot_comprehensive_results()` |
| `src/pinn_models.py` | Neural network models | `RichardsPINN`, `PressureHeadNet`, `WaterTableNet` |
| `src/data_loader.py` | Excel data loading | `load_soil_moisture()` |
| `src/config_loader.py` | YAML parsing | `load_config()`, `PINNConfig` |

### **Configuration Files**

| File | Purpose |
|------|---------|
| `configs/baseline.yaml` | Calhoun CCZO data (6 depths, multi-level headers) |
| `configs/us_uaf_2019.yaml` | US-Uaf data (5 depths, single-level headers) |
| `configs/finetune_calhoun.yaml` | Fine-tuning config (different time period) |

### **Entry Points (How Users Run Things)**

| File | Purpose | Usage |
|------|---------|-------|
| `hpc/train_simple.py` | **NEW simple training script** | `python hpc/train_simple.py --config configs/baseline.yaml` |
| `hpc/train_with_config.py` | Old training script (still works) | `python hpc/train_with_config.py --config configs/baseline.yaml` |
| `notebooks/train_simple.ipynb` | **NEW simple notebook** | Interactive training with 3-line API |
| `notebooks/forward_pinn.ipynb` | Old notebook (still works) | 100+ line cells, old API |

### **Documentation Files (For Users)**

| File | Purpose |
|------|---------|
| `UNIFIED_API_GUIDE.md` | Guide to new unified API |
| `YAML_CONFIG_GUIDE.md` | How to write YAML configs |
| `HANDLING_VARIABLE_DEPTHS.md` | Depth handling & missing data |
| `SYSTEM_OVERVIEW.md` | This file (for AI/developers) |

---

## Data Flow

### **Complete Flow: YAML → Training → Visualization**

```
1. YAML Configuration
   ├── Data: path, dates, column_mapping
   ├── Soil: θs, θr, α, n, Ks, l
   ├── Physics: Sy, zr, L, S_max, zb_initial
   ├── Training: n_epochs, learning_rate, device
   ├── Network: h_net, zb_net architectures
   └── Sampling: cache_size, batch_size, etc.
         ↓
2. PINNDataset.__init__()
   ├── Load Excel via load_soil_moisture()
   │   └── Uses column_mapping to find columns
   ├── Extract depths_names = ['2cm', '15cm', '30cm', ...]
   ├── Extract obs_theta = {'2cm': array, '15cm': array, ...}
   ├── BC: surface_depth = obs_depths[0], bc_times, bc_values
   ├── IC: ic_profile = {'depths': [...], 'theta': [...]}
   └── Store all config parameters as attributes
         ↓
3. train_pinn(dataset)
   ├── Get BC data: dataset.get_bc_data() → (times, values)
   ├── Get IC profile: dataset.ic_profile
   ├── Get soil params: dataset.soil_params
   ├── Initialize RichardsPINN model
   │   ├── Store theta0_times_dim, theta0_values_dim
   │   └── Normalize for training
   ├── Training loop:
   │   ├── Sample collocation points (t, z)
   │   ├── Sample BC points (t_bc)
   │   ├── Compute losses:
   │   │   ├── PDE residual at (t, z)
   │   │   ├── Surface BC: θ_pred(z=0, t_bc) - θ_obs(t_bc)
   │   │   ├── Water table BC: h(z=-zb, t) ≈ 0
   │   │   ├── IC: h(z, t=0) ≈ h_IC(z)
   │   │   └── Total loss = weighted sum
   │   ├── Backprop, update weights
   │   └── Save checkpoints
   └── Return trained model, losses
         ↓
4. plot_results(model, dataset)
   ├── Get obs_data: dataset.get_obs_data_for_viz()
   │   └── Returns {'times': ..., 'depths': [...], 'theta': [...]}
   ├── Iterate over obs_data['depths']:
   │   ├── Predict θ_modeled(depth, times)
   │   ├── Get θ_observed(depth, times)
   │   └── Plot both on Subplot 5
   ├── Plot h(z,t), zb(t), IC, losses
   └── Compute RMSE/MAE statistics
```

---

## API Overview

### **New Unified API (Recommended)**

```python
from src.dataset import PINNDataset
from src.train_loop import train_pinn, finetune_pinn_with_dataset
from src.visualization import plot_results, plot_losses

# 1. Load dataset (everything in one object)
dataset = PINNDataset('configs/baseline.yaml')

# 2. Train (one function call)
model, losses, comps, sample_losses, sample_comps, sample_epochs = train_pinn(
    dataset,
    device='cuda'  # or 'auto', 'cpu'
)

# 3. Visualize (automatic depth handling)
plot_results(model, dataset, device='cuda')
plot_losses(losses, comps, sample_losses, sample_comps, sample_epochs)

# 4. Fine-tuning (also one function call)
finetune_dataset = PINNDataset('configs/finetune_calhoun.yaml')
model_ft, *_ = finetune_pinn_with_dataset(
    'checkpoint_final.pt',
    finetune_dataset
)
```

### **Old API (Still Works)**

```python
from src.config_loader import load_config
from src.data_preprocessing import preprocess_soil_data
from src.train_loop import train_pinn_pool_batch_autoweight
from src.visualization import plot_comprehensive_results

config = load_config('configs/baseline.yaml')
preprocessed = preprocess_soil_data(config)

model = train_pinn_pool_batch_autoweight(
    soil_params=config.soil_params,
    theta0_data=(preprocessed['theta0_times'], preprocessed['theta0_values']),
    ic_profile=preprocessed['ic_profile'],
    # ... 25 more arguments
)

obs_data = {
    'times': preprocessed['times_seconds'],
    'depths': [-0.02, -0.15, -0.30, -0.40, -0.60, -0.80],
    'theta': [preprocessed['theta_2cm'], ...]
}
plot_comprehensive_results(model, theta0_data, config.soil_params, obs_data=obs_data)
```

---

## Critical Implementation Details

### **1. Boundary Condition Handling**

**Where BC data is stored:**
```python
# In PINNDataset (src/dataset.py:173-180)
surface_depth = self.obs_depths[0]  # First depth
theta_surface = self.obs_theta[surface_depth]
self.bc_times = times[valid_mask]
self.bc_values = theta_surface[valid_mask]

# In RichardsPINN model (src/pinn_models.py:180-186)
self.theta0_times_dim = torch.tensor(theta0_data[0])
self.theta0_values_dim = torch.tensor(theta0_data[1])
self.theta0_values_tilde = (theta0_values - θr) / (θs - θr)
```

**How BC residual is computed:**
```python
# src/pinn_models.py:surface_moisture_bc_residual()
def surface_moisture_bc_residual(self, t):
    # Predict θ at surface
    h_pred = self.h_net(z=0, t=t)
    Se_pred = Se_from_h(h_pred)

    # Get observed θ at time t (via linear interpolation)
    theta_obs = self.surface_moisture_tilde(t)

    # Residual (should be ~0)
    return Se_pred - theta_obs
```

### **2. Path Handling for Notebooks**

**Problem:** Notebooks run from `notebooks/`, configs specify paths relative to project root.

**Solution:** `PINNDataset._fix_data_path()` (src/dataset.py:104-122)
```python
if not os.path.exists(data_path):
    parent_path = os.path.join('..', data_path)
    if os.path.exists(parent_path):
        self.config._config['data']['path'] = parent_path
```

**Result:** Works from both project root and notebooks/ directory.

### **3. NaN Handling**

**Three levels of NaN handling:**

1. **Data Loading** (src/data_loader.py):
   - Replaces missing codes (-9999, -6999) with NaN
   - Interpolates small gaps (≤ max_gap_hours)
   - Large gaps remain as NaN

2. **Plotting** (src/visualization.py:263-269):
   ```python
   # Plots all points including NaN
   # Matplotlib shows gaps where data is missing
   axs[1, 2].plot(t_days, theta_obs)  # Gaps visible
   ```

3. **Statistics** (src/visualization.py:306-317):
   ```python
   # Removes NaN before computing RMSE/MAE
   valid_mask = ~np.isnan(theta_obs)
   theta_obs_valid = theta_obs[valid_mask]
   rmse = np.sqrt(np.mean((theta_pinn_valid - theta_obs_valid)**2))
   ```

### **4. Multi-Level Header Handling**

**Calhoun data has 2-row headers:**
```
| Unnamed: 0     | CR1000_2589 | CR1000_2589 |
| Date Time      | 2cm theta   | 15cm theta  |
```

**Config specifies as list:**
```yaml
column_mapping:
  datetime: ['Unnamed: 0_level_0', 'Date Time']
  multi_level_header: true
  depths:
    2cm: ['CR1000_2589', '2cm theta']
```

**Code converts to pandas tuple:**
```python
# src/config_loader.py:73-86
if isinstance(col_name, list):
    col_tuple = tuple(col_name)
df[col_tuple]  # Pandas multi-index access
```

---

## Common User Tasks

### **Task 1: Train on New Dataset**

**Steps:**
1. Create new YAML config (copy `configs/baseline.yaml`)
2. Update `data.path` to new Excel file
3. Update `column_mapping` to match new column names
4. Update `soil` parameters for new site
5. Run: `python hpc/train_simple.py --config configs/new_site.yaml`

**Critical:** Always list depths shallowest → deepest!

### **Task 2: Switch from Calhoun to US-Uaf Data**

**Old way:** Change code in notebooks/scripts
**New way:** Change one line:
```python
# dataset = PINNDataset('configs/baseline.yaml')
dataset = PINNDataset('configs/us_uaf_2019.yaml')
# Everything else stays the same!
```

### **Task 3: Fine-Tuning on New Time Period**

```python
# Base training
base_dataset = PINNDataset('configs/baseline.yaml')  # May 1-19
model_base, *_ = train_pinn(base_dataset)

# Fine-tuning
ft_dataset = PINNDataset('configs/finetune_calhoun.yaml')  # May 20-31
model_ft, *_ = finetune_pinn_with_dataset('checkpoint_final.pt', ft_dataset)
```

### **Task 4: Add Water Table Depth (WTD) Validation**

Add to YAML:
```yaml
column_mapping:
  wtd: 'WTD_column_name'
  wtd_sign: -1.0  # If your WTD is negative, use -1.0 to convert
```

Code automatically:
- Loads WTD if present
- Adds to Subplot 2 (red markers on blue zb(t) line)
- Skips if not present

### **Task 5: Adjust Training Speed vs Quality**

**Fast (for testing):**
```yaml
training:
  n_epochs: 1000
  learning_rate: 5.0e-4
network:
  h_net: {hidden_dim: 32, num_layers: 3}
```

**High Quality (for publication):**
```yaml
training:
  n_epochs: 50000
  learning_rate: 1.0e-4
network:
  h_net: {hidden_dim: 128, num_layers: 6}
optimization:
  use_amp: true  # Mixed precision for speed
```

---

## Troubleshooting Reference

### **Error: FileNotFoundError - No such file 'data/...'**

**Cause:** Path issue when running from notebooks/

**Check:**
```python
# If running from notebooks/, use:
dataset = PINNDataset('../configs/baseline.yaml')

# If running from project root, use:
dataset = PINNDataset('configs/baseline.yaml')
```

**Fix:** Already handled in `PINNDataset._fix_data_path()` but config path must be correct.

---

### **Error: KeyError - Column 'SWC_2cm' not found**

**Cause:** Column name mismatch between YAML and Excel.

**Debug:**
```python
import pandas as pd
df = pd.read_excel('data/your_file.xlsx')
print(df.columns.tolist())  # Check actual column names
```

**Fix:** Update YAML with exact column names (case-sensitive).

---

### **Error: AttributeError - 'dict' object has no attribute 'soil_params'**

**Cause:** `config` variable was overwritten as dict instead of PINNConfig object.

**Fix:** Reload config:
```python
from src.config_loader import load_config
config = load_config(config_file)  # Ensures it's PINNConfig object
```

---

### **Error: Training loss is NaN**

**Causes:**
1. Learning rate too high
2. Invalid soil parameters (n ≤ 1, negative Ks)
3. Bad data (all NaN, wrong units)

**Debug:**
```python
# Check data
print(dataset.bc_values.min(), dataset.bc_values.max())  # Should be 0-1
print(dataset.soil_params)  # Check n > 1, Ks > 0

# Lower learning rate
training.learning_rate = 1.0e-5
```

---

### **Issue: Training very slow**

**Solutions:**
1. Use GPU: `device='cuda'`
2. Enable mixed precision: `use_amp: true`
3. Reduce network size: `hidden_dim: 32, num_layers: 3`
4. Reduce cache: `cache_size: 5000`

---

### **Issue: Subplot 5 shows wrong number of depths**

**Cause:** YAML `depths` section doesn't match actual data.

**Fix:** Only include depths you actually have:
```yaml
depths:
  2cm: 'sensor_A'   # Have this ✅
  15cm: 'sensor_B'  # Have this ✅
  # Don't list 80cm if you don't have it
```

---

## Future Extension Points

### **Easy Extensions**

1. **Add new dataset:**
   - Create new YAML config
   - Specify column_mapping
   - Run training (no code changes!)

2. **Add WTD validation:**
   - Add `wtd: 'column_name'` to YAML
   - Automatic visualization

3. **Change hyperparameters:**
   - Edit YAML file
   - Rerun training

### **Medium Extensions**

1. **Flux-based BC (instead of moisture):**
   - Modify `RichardsPINN.surface_*_residual()`
   - Add flux column to YAML
   - Update `PINNDataset` to load flux

2. **Multiple sites/periods:**
   - Create multiple configs
   - Train on each
   - Compare results

3. **Ensemble training:**
   - Train multiple models with different seeds
   - Average predictions

### **Advanced Extensions**

1. **Heterogeneous soil layers:**
   - Modify PDE residual to include layer boundaries
   - Add layer parameters to YAML

2. **2D/3D flow:**
   - Extend networks to accept (x, y, z, t)
   - Modify PDE residual for 2D/3D Richards

3. **Inverse problem (parameter estimation):**
   - Make soil parameters trainable
   - Add regularization loss

---

## Quick Reference for AI Assistants

### **When User Asks About...**

| Topic | Key Files | Key Concepts |
|-------|-----------|--------------|
| "How to use new dataset" | YAML_CONFIG_GUIDE.md | column_mapping, depths ordering |
| "Training not working" | src/train_loop.py, SYSTEM_OVERVIEW.md | Check loss=NaN, learning rate, data quality |
| "Depths not showing" | src/dataset.py:173, src/visualization.py:241 | First depth = BC, others for validation |
| "Path errors" | src/dataset.py:104-122 | Notebooks vs root directory |
| "What's the new API" | UNIFIED_API_GUIDE.md | PINNDataset → train_pinn() → plot_results() |
| "YAML syntax" | YAML_CONFIG_GUIDE.md | Examples, templates, troubleshooting |

### **Critical Code Locations**

| Function/Class | File | Line(s) | Purpose |
|----------------|------|---------|---------|
| `PINNDataset.__init__()` | src/dataset.py | 72-102 | Main data loading |
| `surface_depth = obs_depths[0]` | src/dataset.py | 173 | **BC selection** |
| `train_pinn()` | src/train_loop.py | 960-1058 | New unified training API |
| `plot_results()` | src/visualization.py | 435-458 | New unified viz API |
| `surface_moisture_bc_residual()` | src/pinn_models.py | ~220-243 | BC loss computation |
| `get_obs_data_for_viz()` | src/dataset.py | 380-399 | Format obs data for plotting |

### **Architecture Decision Summary**

✅ **YAML-driven:** All config in YAML, no hardcoded assumptions
✅ **Unified container:** PINNDataset holds everything
✅ **First depth convention:** obs_depths[0] = surface BC
✅ **Automatic adaptation:** Works with any number of depths
✅ **Backward compatible:** Old API still works
✅ **Path-aware:** Works from notebooks/ or project root
✅ **NaN-tolerant:** Handles missing data gracefully

---

## Summary for Quick Onboarding

**If you're Claude Code helping a user:**

1. **Read this file first** - Understand system architecture
2. **Check UNIFIED_API_GUIDE.md** - Know the new 3-line API
3. **Reference YAML_CONFIG_GUIDE.md** - Help with configs
4. **Remember key conventions:**
   - First depth = surface BC
   - YAML drives everything
   - PINNDataset is central
   - Old API coexists with new

5. **Common user needs:**
   - "Train on my data" → YAML config creation
   - "Not working" → Check paths, column names, data quality
   - "Slow training" → GPU, mixed precision, smaller network
   - "Wrong depths" → Check YAML ordering and column_mapping

**Everything is designed to be YAML-driven and automatic. The user should rarely need to modify Python code.**

---

**End of System Overview**

**Last Updated:** 2025-01-28
**Maintainer:** System auto-generates from configs
**Version:** 1.0.0 - Unified API Release
