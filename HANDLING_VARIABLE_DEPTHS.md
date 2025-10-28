# Handling Variable Depths and Missing Data

## Overview

The unified API automatically handles:
- ✅ **Variable number of depths** (3, 5, 6, 10... doesn't matter)
- ✅ **Different depth levels** (2cm, 15cm, 30cm vs 5cm, 10cm, 25cm)
- ✅ **Missing data (NaN)** at specific depths or time periods
- ✅ **Failed sensors** (entire depth has NaN)

---

## Example Scenarios

### **Scenario 1: Different Number of Depths**

#### Site A (Calhoun CCZO - 6 depths):
```yaml
# configs/baseline.yaml
depths:
  2cm: ['CR1000_2589', '2cm theta']
  15cm: ['CR1000_2589', '15cm theta']
  30cm: ['CR1000_2588', '30cm_theta']
  40cm: ['CR1000_2589', '40cm theta']
  60cm: ['CR1000_2588', '60cm_theta']
  80cm: ['CR1000_2588', '80cm_theta']
```

Result:
```python
dataset = PINNDataset('configs/baseline.yaml')
print(dataset.obs_depths)
# ['2cm', '15cm', '30cm', '40cm', '60cm', '80cm']  ← 6 depths

# Plotting: Shows 6 lines in Subplot 5 (θ validation)
```

#### Site B (US-Uaf - 5 depths):
```yaml
# configs/us_uaf_2019.yaml
depths:
  2cm: 'SWC_1_2_1'
  15cm: 'SWC_1_1_1'
  30cm: 'SWC_2_1_1'
  40cm: 'SWC_2_2_1'
  60cm: 'SWC_2_3_1'
  # No 80cm!
```

Result:
```python
dataset = PINNDataset('configs/us_uaf_2019.yaml')
print(dataset.obs_depths)
# ['2cm', '15cm', '30cm', '40cm', '60cm']  ← 5 depths

# Plotting: Shows 5 lines in Subplot 5 (θ validation)
```

**✅ Same code, different outputs! Automatic adaptation.**

---

### **Scenario 2: Different Depth Levels**

#### Site C (Custom depths):
```yaml
depths:
  5cm: 'sensor_A'
  10cm: 'sensor_B'
  25cm: 'sensor_C'
  50cm: 'sensor_D'
  100cm: 'sensor_E'
```

Result:
```python
dataset = PINNDataset('configs/custom_site.yaml')
print(dataset.obs_depths)
# ['5cm', '10cm', '25cm', '50cm', '100cm']

# Plotting: Shows "5cm obs", "10cm obs", etc. in legend
```

**✅ Works with ANY depth naming! Just specify in YAML.**

---

### **Scenario 3: Missing Data (NaN) During Time Period**

#### Example: 40cm sensor failed from day 5-10

Excel data:
```
| datetime   | SWC_40cm |
|------------|----------|
| Day 1      | 0.78     | ✅ Valid
| Day 2      | 0.79     | ✅ Valid
| Day 3      | 0.80     | ✅ Valid
| Day 4      | 0.81     | ✅ Valid
| Day 5      | NaN      | ❌ Missing
| Day 6      | NaN      | ❌ Missing
| Day 7      | NaN      | ❌ Missing
| Day 8      | 0.82     | ✅ Valid
| Day 9      | 0.83     | ✅ Valid
```

**What happens:**

1. **Data Loading** (src/data_loader.py):
   ```python
   # Interpolates small gaps (≤ 6 hours by default)
   # Large gaps (> 6 hours) remain as NaN
   ```

2. **Plotting** (Subplot 5):
   ```python
   # Matplotlib plots with gaps where NaN exists
   axs[1, 2].plot(t_days, theta_obs)  # ← Shows gap from day 5-7
   ```

   **Visual result:**
   ```
   θ (m³/m³)
   0.85 |     ●————●————●————●           ●————●
   0.80 |                      [gap]
        +————————————————————————————————————> Time
           1   2   3   4   5   6   7   8   9
   ```

3. **Statistics** (printed output):
   ```python
   # Only computes RMSE/MAE on valid data points
   valid_mask = ~np.isnan(theta_obs)
   rmse = np.sqrt(np.mean((theta_pinn[valid_mask] - theta_obs[valid_mask])**2))
   # ✅ Ignores NaN in statistics
   ```

**✅ Gaps are visible but don't break the code or statistics.**

---

### **Scenario 4: Entire Depth Has No Valid Data**

#### Example: 80cm sensor completely failed

Excel data:
```
| datetime   | SWC_80cm |
|------------|----------|
| Day 1      | NaN      |
| Day 2      | NaN      |
| Day 3      | NaN      |
| ...        | NaN      |
```

**What happens:**

1. **Config option 1: Don't include it**
   ```yaml
   depths:
     2cm: 'SWC_1_2_1'
     15cm: 'SWC_1_1_1'
     30cm: 'SWC_2_1_1'
     # Don't list 80cm if it's all NaN
   ```
   **✅ Dataset just has 3 depths, plots 3 lines**

2. **Config option 2: Include it anyway**
   ```yaml
   depths:
     2cm: 'SWC_1_2_1'
     15cm: 'SWC_1_1_1'
     30cm: 'SWC_2_1_1'
     80cm: 'SWC_80cm'  # All NaN
   ```

   **What happens:**
   - Dataset loads with 4 depths
   - Subplot 5 tries to plot 80cm
   - **All NaN → Empty line or warning**
   - Statistics: "80cm: RMSE = NaN (skipped - no valid data)"

**Recommendation: Don't include depths with all NaN in your YAML config.**

---

## How It Works (Technical)

### **1. PINNDataset.get_obs_data_for_viz()**

```python
def get_obs_data_for_viz(self):
    """Get observation data for plotting."""
    obs_depths_m = []
    obs_theta_list = []

    # Iterate over ONLY the depths specified in config
    for depth_name in self.obs_depths:  # ← From column_mapping.depths
        depth_m = -self._depth_name_to_meters(depth_name)
        obs_depths_m.append(depth_m)
        obs_theta_list.append(self.obs_theta[depth_name])  # ← Can have NaN

    return {
        'times': self.obs_times,
        'depths': obs_depths_m,      # List of depths that exist
        'theta': obs_theta_list      # List of arrays (can have NaN)
    }
```

### **2. plot_comprehensive_results() - Subplot 5**

```python
if obs_data is not None:
    obs_depths = obs_data['depths']      # Only depths from dataset
    obs_theta_series = obs_data['theta']

    # Iterate over actual depths (not hardcoded!)
    for i, (z_obs, theta_obs) in enumerate(zip(obs_depths, obs_theta_series)):
        # Get modeled theta at this depth
        theta_pinn = model.predict_theta(z_obs_tensor, t_obs_tensor)

        # Plot (NaN creates gaps - visually shows data quality)
        depth_cm = int(abs(z_obs) * 100)
        axs[1, 2].plot(t_days, theta_obs, label=f'{depth_cm}cm obs')
        axs[1, 2].plot(t_days, theta_pinn, label=f'{depth_cm}cm PINN')
```

### **3. Statistics Computation**

```python
# For each depth, compute RMSE/MAE
for i, (z_obs, theta_obs) in enumerate(zip(obs_data['depths'], obs_data['theta'])):
    theta_obs_np = np.array(theta_obs, dtype=np.float64)

    # Filter out NaN values
    valid_mask = ~np.isnan(theta_obs_np)
    theta_obs_valid = theta_obs_np[valid_mask]
    theta_pinn_valid = theta_pinn[valid_mask]

    # Only compute if valid data exists
    if len(theta_obs_valid) > 0:
        rmse = np.sqrt(np.mean((theta_pinn_valid - theta_obs_valid)**2))
        print(f"  {depth_cm}cm: RMSE = {rmse:.6f} m³/m³")
    else:
        print(f"  {depth_cm}cm: No valid data (all NaN)")
```

---

## Visual Comparison

### **Calhoun (6 depths):**
```
Subplot 5: θ Validation: Obs vs PINN
────────────────────────────────────────
0.5 |  ●————●  2cm obs
    |  - - -  2cm PINN
0.4 |  ●————●  15cm obs
    |  - - -  15cm PINN
0.3 |  ●————●  30cm obs
    |  - - -  30cm PINN
0.25|  ●————●  40cm obs
    |  - - -  40cm PINN
0.2 |  ●————●  60cm obs
    |  - - -  60cm PINN
0.15|  ●————●  80cm obs
    |  - - -  80cm PINN
    +────────────────────────> Time [days]
         5        10       15
```

### **US-Uaf (5 depths):**
```
Subplot 5: θ Validation: Obs vs PINN
────────────────────────────────────────
0.5 |  ●————●  2cm obs
    |  - - -  2cm PINN
0.4 |  ●————●  15cm obs
    |  - - -  15cm PINN
0.3 |  ●————●  30cm obs
    |  - - -  30cm PINN
0.25|  ●————●  40cm obs
    |  - - -  40cm PINN
0.2 |  ●————●  60cm obs
    |  - - -  60cm PINN
    +────────────────────────> Time [days]
         5        10       14
```

**✅ Automatically adapts! No 80cm line for US-Uaf.**

---

## Best Practices

### **1. Only include depths with sufficient data**
```yaml
# ✅ Good
depths:
  2cm: 'sensor_A'   # 95% valid data
  15cm: 'sensor_B'  # 90% valid data
  30cm: 'sensor_C'  # 85% valid data

# ❌ Avoid
depths:
  40cm: 'sensor_D'  # 10% valid data (mostly NaN)
  80cm: 'sensor_E'  # 0% valid data (all NaN)
```

### **2. Check data quality before training**
```python
dataset = PINNDataset('configs/my_site.yaml', verbose=True)
# Will print data completeness for each depth
```

Output example:
```
Raw data quality (after replacing missing codes with NaN):
----------------------------------------------------------------------
2cm   :   630 valid,    43 NaN ( 6.39%)  ← Good!
15cm  :   673 valid,     0 NaN ( 0.00%)  ← Perfect!
40cm  :   495 valid,   178 NaN (26.45%)  ← Questionable...
80cm  :     0 valid,   673 NaN (100.0%)  ← Remove from config!
```

### **3. Interpolation handles small gaps**
```yaml
data:
  interpolate: true
  max_gap_hours: 6  # Interpolate gaps ≤ 6 hours
```

This fills:
- ✅ Single missing points
- ✅ Small sensor glitches (< 6 hours)
- ❌ Large gaps (sensor failure for days)

---

## Summary

### **What's Automatic:**
✅ Detects number of depths from YAML
✅ Adapts plotting to show only available depths
✅ Handles NaN by creating gaps in plots
✅ Skips NaN in statistics (RMSE/MAE)
✅ Works with ANY depth naming (2cm, 5cm, 100cm, etc.)

### **What You Control:**
📝 Which depths to include in `column_mapping.depths`
📝 Interpolation settings (`max_gap_hours`)
📝 Whether to include depths with lots of NaN

### **Key Insight:**
**The unified API makes depth handling completely flexible.** You just specify what you have in YAML, and everything adapts automatically!

---

**Last Updated**: 2025-01-28
