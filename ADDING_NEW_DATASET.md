# Adding a New Dataset - Quick Guide

## Overview
This guide shows you how to add a new Excel dataset and train a PINN model on it.

**Time estimate:** 10-15 minutes

---

## Step-by-Step Process

### Step 1: Add Your Excel File

```bash
# Copy your file to the data folder
cp /path/to/your/new_data.xlsx data/

# Example:
cp ~/Downloads/Site_XYZ_2020.xlsx data/
```

### Step 2: Inspect Your Excel File

Open your Excel file and identify:

1. **Datetime column name** (e.g., `datetime`, `Date Time`, `timestamp`)
2. **Soil moisture column names** at different depths
3. **Header structure**: Single-level or multi-level?
4. **Units**: Percentage (0-100) or fraction (0-1)?
5. **Optional**: Water table depth column

**Example inspection:**

```
Single-level header:
┌─────────────┬─────────┬──────────┬──────────┐
│ datetime    │ SWC_5cm │ SWC_15cm │ SWC_30cm │
├─────────────┼─────────┼──────────┼──────────┤
│ 2020-05-01  │ 25.3    │ 30.1     │ 35.8     │
└─────────────┴─────────┴──────────┴──────────┘
→ datetime: 'datetime'
→ multi_level_header: false
→ unit_conversion: 0.01 (it's percentage)

Multi-level header:
┌──────────────────┬─────────────┬─────────────┐
│ Unnamed: 0_level_0│ CR1000_2589 │ CR1000_2589 │
├──────────────────┼─────────────┼─────────────┤
│ Date Time        │ 5cm theta   │ 15cm theta  │
├──────────────────┼─────────────┼─────────────┤
│ 2020-05-01 00:00 │ 0.253       │ 0.301       │
└──────────────────┴─────────────┴─────────────┘
→ datetime: ['Unnamed: 0_level_0', 'Date Time']
→ multi_level_header: true
→ unit_conversion: 1.0 (already fraction)
```

### Step 3: Create YAML Config

```bash
# Copy the template
cp configs/new_site_template.yaml configs/my_site_2020.yaml

# Edit with your favorite editor
nano configs/my_site_2020.yaml
# or
code configs/my_site_2020.yaml
```

### Step 4: Customize the YAML

Fill in these critical sections:

#### 4.1 Data Path and Date Range
```yaml
data:
  path: 'data/Site_XYZ_2020.xlsx'  # ← Your file name
  start_date: '2020-05-01'         # ← Start date
  end_date: '2020-09-30'           # ← End date
```

#### 4.2 Column Mapping (MOST IMPORTANT!)

**For single-level headers:**
```yaml
column_mapping:
  datetime: 'datetime'  # ← Your datetime column name
  multi_level_header: false
  unit_conversion: 0.01  # ← 0.01 for %, 1.0 for fraction

  depths:
    5cm: 'SWC_5cm'    # ← FIRST = Surface BC (shallowest)
    15cm: 'SWC_15cm'  # ← Match your column names exactly!
    30cm: 'SWC_30cm'
    50cm: 'SWC_50cm'
```

**For multi-level headers:**
```yaml
column_mapping:
  datetime: ['Unnamed: 0_level_0', 'Date Time']
  multi_level_header: true
  unit_conversion: 1.0

  depths:
    5cm: ['CR1000_2589', '5cm theta']   # ← [Logger, Column]
    15cm: ['CR1000_2589', '15cm theta']
    30cm: ['CR1000_2590', '30cm theta']
```

⚠️ **CRITICAL**: The first depth listed = surface boundary condition!

#### 4.3 Soil Parameters

Update these for your site (get from soil surveys or measurements):
```yaml
soil:
  theta_s: 0.45    # Saturated water content
  theta_r: 0.05    # Residual water content
  alpha: 3.5       # van Genuchten α [1/m]
  n: 1.5           # van Genuchten n
  Ks: 1.0e-5       # Saturated conductivity [m/s]
  l: 0.5           # Pore-connectivity parameter
```

**Where to find soil parameters:**
- SSURGO database: https://websoilsurvey.nrcs.usda.gov/
- Site measurements (if available)
- Literature values for your soil type
- Pedotransfer functions (Rosetta, etc.)

#### 4.4 Physics Parameters

```yaml
physics:
  Sy: 0.3                     # Specific yield
  zr: 0.5                     # Root zone depth [m]
  L: 4.0                      # Characteristic length [m]
  S_max: 1.0e-7               # Maximum sink term [1/s]
  zb_initial: 5.0             # Initial water table depth [m]
  ic_type: 'obs'              # Use observed data for IC
```

### Step 5: Test Data Loading

Before training, verify your config loads correctly:

```python
from src.dataset import PINNDataset

# Load dataset
dataset = PINNDataset('configs/my_site_2020.yaml', verbose=True)

# This will print:
# - Number of observation depths found
# - Data completeness for each depth
# - Date range loaded
# - Boundary condition info
```

**Example output:**
```
Loading data from: data/Site_XYZ_2020.xlsx
Date range: 2020-05-01 to 2020-09-30 (153 days)

Raw data quality (after replacing missing codes with NaN):
----------------------------------------------------------------------
5cm   :   3650 valid,    18 NaN ( 0.49%)  ← Excellent!
15cm  :   3598 valid,    70 NaN ( 1.91%)  ← Good
30cm  :   3420 valid,   248 NaN ( 6.76%)  ← Acceptable
50cm  :   2890 valid,   778 NaN (21.2%)   ← May want to exclude

Boundary Condition (Surface θ):
  Using depth: 5cm (first in config)
  Valid BC points: 3650
  BC time range: 0.000 to 153.0 days
```

### Step 6: Run Training

**Option A: Command line**
```bash
python hpc/train_simple.py --config configs/my_site_2020.yaml
```

**Option B: Interactive notebook**
```python
# In notebooks/train_simple.ipynb

# Cell 1: Setup
import sys
sys.path.append('..')
from src.dataset import PINNDataset
from src.train_loop import train_pinn
from src.visualization import plot_results

# Cell 2: Load data
config_file = '../configs/my_site_2020.yaml'
dataset = PINNDataset(config_file, verbose=True)

# Cell 3: Train
model, losses, comps, _, _, _ = train_pinn(dataset, device='cuda')

# Cell 4: Visualize
plot_results(model, dataset, device='cuda')
```

---

## Complete Example

Let's say you have a file `US-Foo_2021.xlsx` that looks like:

```
| timestamp   | SWC_10  | SWC_20  | SWC_40  | WTD   |
|-------------|---------|---------|---------|-------|
| 2021-06-01  | 28.5    | 32.1    | 35.8    | 4.2   |
| 2021-06-02  | 27.9    | 31.8    | 35.5    | 4.3   |
```

**Your YAML would be:**

```yaml
data:
  path: 'data/US-Foo_2021.xlsx'
  start_date: '2021-06-01'
  end_date: '2021-09-01'
  interpolate: true
  max_gap_hours: 6

  column_mapping:
    datetime: 'timestamp'
    multi_level_header: false
    unit_conversion: 0.01  # Data is percentage

    depths:
      10cm: 'SWC_10'   # Surface BC
      20cm: 'SWC_20'
      40cm: 'SWC_40'

    wtd: 'WTD'         # Optional
    wtd_sign: 1.0      # Positive = depth below surface

soil:
  theta_s: 0.42
  theta_r: 0.06
  alpha: 2.5
  n: 1.6
  Ks: 8.0e-6
  l: 0.5

# ... rest of config (training params, etc.)
```

Then run:
```bash
python hpc/train_simple.py --config configs/us_foo_2021.yaml
```

---

## Troubleshooting

### ❌ FileNotFoundError: No such file
**Problem:** Path is wrong
**Solution:**
```yaml
# Make sure path is relative to PROJECT ROOT
data:
  path: 'data/your_file.xlsx'  # ✅ Correct
  # NOT: '../data/your_file.xlsx'  # ❌ Wrong
```

### ❌ KeyError: 'SWC_5cm'
**Problem:** Column name doesn't match
**Solution:**
- Check spelling and case (exact match required!)
- For multi-level headers, use list: `['Logger', 'Column']`
- Print your columns: `pd.read_excel('data/file.xlsx').columns`

### ❌ ValueError: theta_s must be > theta_r
**Problem:** Invalid soil parameters
**Solution:**
- Ensure 0 < theta_r < theta_s < 1
- Check you didn't swap theta_s and theta_r

### ❌ "No valid data" or "All NaN"
**Problem:** Unit conversion or date range issue
**Solution:**
- If data is 0-100 (percentage): use `unit_conversion: 0.01`
- If data is 0-1 (fraction): use `unit_conversion: 1.0`
- Check date range exists in your file

### ❌ Training crashes or NaN loss
**Problem:** Numerical issues
**Solution:**
- Check soil parameters are physically reasonable
- Ensure zb_initial is realistic (positive, in meters)
- Try reducing learning_rate to 5e-5 or 1e-5

---

## Quick Reference: Common Excel Formats

### Format 1: Simple single-level
```yaml
datetime: 'Date'
multi_level_header: false
depths:
  5cm: 'SWC_5'
  15cm: 'SWC_15'
```

### Format 2: Multi-level (Calhoun style)
```yaml
datetime: ['Unnamed: 0_level_0', 'Date Time']
multi_level_header: true
depths:
  2cm: ['CR1000_2589', '2cm theta']
  15cm: ['CR1000_2589', '15cm theta']
```

### Format 3: Different depth names
```yaml
datetime: 'timestamp'
multi_level_header: false
depths:
  0.1m: 'sensor_A'   # Any depth name works!
  0.3m: 'sensor_B'
  0.5m: 'sensor_C'
```

---

## Checklist

Before training, verify:

- [ ] Excel file is in `data/` folder
- [ ] YAML `data.path` points to correct file
- [ ] Date range (`start_date`, `end_date`) is correct
- [ ] Column names match EXACTLY (case-sensitive)
- [ ] `multi_level_header` is set correctly
- [ ] `unit_conversion` is correct (0.01 for %, 1.0 for fraction)
- [ ] First depth in `depths:` is your shallowest sensor (surface BC)
- [ ] Soil parameters are reasonable for your site
- [ ] `zb_initial` is realistic for your site
- [ ] Test loading with `PINNDataset(..., verbose=True)` first

---

## Next Steps

After successful training:

1. **Check results**: Look at the 6-panel plot
   - Subplot 5 (θ validation): Are PINN predictions close to observations?
   - Subplot 6 (h(z,t)): Does pressure head field look reasonable?

2. **Evaluate metrics**: Check printed RMSE values
   - RMSE < 0.02 m³/m³ is excellent
   - RMSE < 0.05 m³/m³ is good
   - RMSE > 0.10 m³/m³ needs investigation

3. **Iterate if needed**:
   - Adjust hyperparameters (learning_rate, n_epochs)
   - Try different network architecture
   - Refine soil parameters
   - Extend training duration

4. **Fine-tuning**: If results are close but not perfect:
   ```python
   from src.train_loop import finetune_pinn_with_dataset
   model = finetune_pinn_with_dataset(
       'checkpoints_train/checkpoint_epoch_5000.pt',
       dataset,
       n_epochs=2000,
       device='cuda'
   )
   ```

---

## Additional Resources

- **YAML_CONFIG_GUIDE.md** - Comprehensive YAML reference with all options
- **SYSTEM_OVERVIEW.md** - Complete system architecture reference
- **HANDLING_VARIABLE_DEPTHS.md** - How depth handling works
- **UNIFIED_API_GUIDE.md** - API usage examples

---

**Last Updated:** 2025-01-28
