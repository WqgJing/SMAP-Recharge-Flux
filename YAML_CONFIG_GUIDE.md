# YAML Configuration Guide for PINN Training

## Overview

This guide shows you how to write YAML configuration files for training PINNs on your soil moisture data. Each config file specifies:
- **Data source** (Excel file, columns, dates)
- **Soil parameters** (van Genuchten-Mualem)
- **Physics parameters** (specific yield, root zone depth, etc.)
- **Network architecture** (layers, neurons)
- **Training hyperparameters** (epochs, learning rate, sampling)

---

## Table of Contents

1. [Quick Start Template](#quick-start-template)
2. [Section-by-Section Guide](#section-by-section-guide)
3. [Complete Examples](#complete-examples)
4. [Common Patterns](#common-patterns)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start Template

Copy this template and fill in your values:

```yaml
# ============================================================================
# MY SITE CONFIGURATION
# ============================================================================

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
data:
  path: 'data/my_site_data.xlsx'
  start_date: '2020-06-01'
  end_date: '2020-06-30'
  interpolate: true
  max_gap_hours: 6

  # Column mapping - CUSTOMIZE THIS FOR YOUR DATA!
  column_mapping:
    # Datetime column
    datetime: 'Date'  # Replace with your datetime column name

    # Header structure
    multi_level_header: false  # true if your Excel has 2 header rows

    # Unit conversion (1.0 if already m³/m³, 0.01 if percentage)
    unit_conversion: 1.0

    # Soil moisture sensors (ALWAYS list shallowest first!)
    depths:
      2cm: 'Sensor_A'    # Replace with your column names
      15cm: 'Sensor_B'
      30cm: 'Sensor_C'
      # Add more depths as needed

    # Optional: Water table depth
    # wtd: 'WaterTable'
    # wtd_sign: 1.0  # Use -1.0 if your WTD is negative

# ============================================================================
# SOIL PARAMETERS (van Genuchten-Mualem)
# ============================================================================
soil:
  theta_s: 0.46    # Saturated water content [-]
  theta_r: 0.05    # Residual water content [-]
  alpha: 4.0       # van Genuchten α [1/m]
  n: 1.45          # van Genuchten n (>1) [-]
  Ks: 2.0e-5       # Saturated conductivity [m/s]
  l: 0.50          # Mualem pore-connectivity [-]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
training:
  n_epochs: 5000
  learning_rate: 1.0e-4
  device: 'auto'      # 'auto', 'cuda', or 'cpu'
  seed: 42

# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================
network:
  h_net:              # Pressure head network
    hidden_dim: 64
    num_layers: 4
  zb_net:             # Water table network
    hidden_dim: 32
    num_layers: 3

# ============================================================================
# SAMPLING CONFIGURATION
# ============================================================================
sampling:
  batch_size: 500            # Points sampled per iteration
  batch_size: 500
  boundary_ratio: 0.7        # Fraction near boundaries
  boundary_ratio: 0.7
  temperature: 0.3

boundary_sampling:
  batch_size_bc: 450
  interp_ratio: 0.30
  neighbor_ratio: 0.00
  baseline_ratio: 0.70
  neighbor_expansion: 0
  gradient_threshold: 0.7
  power: 2.0

# ============================================================================
# OPTIMIZATION
# ============================================================================
optimization:
  weight_update_freq: 10000000000
  weight_lr: 0.1
  use_initial_scales: true
  use_amp: false
  use_multi_gpu: true
  grad_accumulation_steps: 1

# ============================================================================
# CHECKPOINTING
# ============================================================================
checkpointing:
  dir: 'checkpoints_train'
  freq: 10000
  keep_last_n: 3

# ============================================================================
# PHYSICS PARAMETERS
# ============================================================================
physics:
  Sy: 0.3           # Specific yield [-]
  zr: 0.5           # Root zone depth [m]
  L: 4.0            # Characteristic length [m]
  S_max: 1.0e-7     # Maximum sink term [1/s]
  zb_initial: 6.1   # Initial water table depth [m]
  ic_type: 'obs'    # 'obs', 'linear', or 'hydrostatic'

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
output:
  dir: 'outputs'
```

---

## Section-by-Section Guide

### 1. **Data Configuration**

```yaml
data:
  path: 'data/my_site_data.xlsx'
  start_date: '2020-06-01'
  end_date: '2020-06-30'
  interpolate: true
  max_gap_hours: 6
```

**Fields:**
- `path`: Path to Excel file (relative to project root)
- `start_date`: Start date (YYYY-MM-DD format)
- `end_date`: End date (YYYY-MM-DD format)
- `interpolate`: Fill small gaps in data? (true/false)
- `max_gap_hours`: Maximum gap to interpolate (hours)

**Tips:**
- Use relative paths: `'data/my_file.xlsx'`
- Date format must be `'YYYY-MM-DD'` (with quotes!)
- Set `interpolate: false` if you want raw data only

---

### 2. **Column Mapping** ⭐ MOST IMPORTANT!

This tells the code which columns in your Excel file contain what data.

#### **Case 1: Simple Single-Level Headers**

Your Excel looks like:
```
| Date       | SWC_2cm | SWC_15cm | SWC_30cm | WTD   |
|------------|---------|----------|----------|-------|
| 2020-06-01 | 0.245   | 0.312    | 0.289    | -0.25 |
```

YAML config:
```yaml
column_mapping:
  datetime: 'Date'
  multi_level_header: false
  unit_conversion: 1.0

  depths:
    2cm: 'SWC_2cm'
    15cm: 'SWC_15cm'
    30cm: 'SWC_30cm'

  wtd: 'WTD'
  wtd_sign: -1.0
```

#### **Case 2: Multi-Level Headers (Calhoun Style)**

Your Excel looks like:
```
| Unnamed: 0     | CR1000_2589 | CR1000_2589 | CR1000_2588 |
| Date Time      | 2cm theta   | 15cm theta  | 30cm_theta  |
|----------------|-------------|-------------|-------------|
| 2017-05-01     | 0.245       | 0.312       | 0.289       |
```

YAML config:
```yaml
column_mapping:
  datetime: ['Unnamed: 0_level_0', 'Date Time']  # Use list for multi-level
  multi_level_header: true
  unit_conversion: 1.0

  depths:
    2cm: ['CR1000_2589', '2cm theta']   # Use list for multi-level
    15cm: ['CR1000_2589', '15cm theta']
    30cm: ['CR1000_2588', '30cm_theta']
```

#### **Case 3: Percentage Units**

Your data is in percentage (0-100) instead of fraction (0-1):
```
| Date       | SWC_2cm |
|------------|---------|
| 2020-06-01 | 24.5    | ← Percentage
```

YAML config:
```yaml
column_mapping:
  unit_conversion: 0.01  # Convert percentage → fraction
  depths:
    2cm: 'SWC_2cm'
```

#### **Depth Naming Rules:**

**✅ DO:**
```yaml
depths:
  2cm: 'sensor_A'    # Shallowest FIRST (becomes surface BC)
  15cm: 'sensor_B'
  30cm: 'sensor_C'
  60cm: 'sensor_D'   # Deepest LAST
```

**❌ DON'T:**
```yaml
depths:
  60cm: 'sensor_D'   # Deep first - WRONG ORDER!
  30cm: 'sensor_C'
  15cm: 'sensor_B'
  2cm: 'sensor_A'
```

**Why order matters:** The first depth is used as the surface boundary condition!

---

### 3. **Soil Parameters**

```yaml
soil:
  theta_s: 0.46    # Saturated water content [-]
  theta_r: 0.05    # Residual water content [-]
  alpha: 4.0       # van Genuchten α [1/m]
  n: 1.45          # van Genuchten n (>1) [-]
  Ks: 2.0e-5       # Saturated conductivity [m/s]
  l: 0.50          # Mualem pore-connectivity [-]
```

**How to get these values:**
1. **From literature** for your soil type (sand, loam, clay)
2. **From lab measurements** (soil cores)
3. **From calibration** (inverse modeling)
4. **From databases** (USDA SSURGO, ROSETTA)

**Typical ranges:**

| Parameter | Sand | Loam | Clay |
|-----------|------|------|------|
| θs | 0.35-0.43 | 0.43-0.51 | 0.38-0.50 |
| θr | 0.02-0.05 | 0.04-0.06 | 0.05-0.10 |
| α [1/m] | 3-15 | 1-3 | 0.5-2 |
| n | 1.5-3.0 | 1.3-2.0 | 1.1-1.5 |
| Ks [m/s] | 1e-4 - 1e-5 | 1e-5 - 1e-6 | 1e-6 - 1e-8 |
| l | 0.5 | 0.5 | 0.5 |

---

### 4. **Training Configuration**

```yaml
training:
  n_epochs: 5000         # Number of training iterations
  learning_rate: 1.0e-4  # Step size for gradient descent
  device: 'auto'         # 'auto', 'cuda', 'cpu'
  seed: 42               # Random seed for reproducibility
```

**Guidelines:**
- **n_epochs**:
  - Quick test: 1000
  - Normal: 5000-10000
  - High quality: 20000-50000

- **learning_rate**:
  - Too high (>1e-3): Training unstable
  - Good (1e-4): Standard
  - Too low (<1e-5): Training very slow

- **device**:
  - `'auto'`: Use GPU if available, else CPU
  - `'cuda'`: Force GPU (fails if no GPU)
  - `'cpu'`: Force CPU (slower)

---

### 5. **Network Architecture**

```yaml
network:
  h_net:              # Pressure head network
    hidden_dim: 64    # Neurons per layer
    num_layers: 4     # Number of hidden layers

  zb_net:             # Water table network
    hidden_dim: 32
    num_layers: 3
```

**Guidelines:**
- **Bigger network** (more layers/neurons):
  - ✅ Better accuracy
  - ❌ Slower training
  - ❌ Risk of overfitting

- **Smaller network**:
  - ✅ Faster training
  - ❌ May underfit complex dynamics

**Recommended starting points:**
- Simple case: `h_net: {hidden_dim: 32, num_layers: 3}`
- Standard: `h_net: {hidden_dim: 64, num_layers: 4}` ✅
- Complex: `h_net: {hidden_dim: 128, num_layers: 5}`

---

### 6. **Physics Parameters**

```yaml
physics:
  Sy: 0.3           # Specific yield [-]
  zr: 0.5           # Root zone depth [m]
  L: 4.0            # Characteristic length [m]
  S_max: 1.0e-7     # Maximum sink term [1/s]
  zb_initial: 6.1   # Initial water table depth [m]
  ic_type: 'obs'    # 'obs', 'linear', 'hydrostatic'
```

**How to set:**

- **Sy** (Specific yield):
  - Sandy: 0.20-0.35
  - Loamy: 0.10-0.25
  - Clayey: 0.01-0.10
  - Start with 0.3 if unsure

- **zr** (Root zone depth):
  - Shallow roots (grass): 0.2-0.5 m
  - Medium (crops): 0.5-1.0 m
  - Deep (trees): 1.0-3.0 m

- **L** (Characteristic length):
  - Depth to water table or domain depth
  - Typical: 2-10 m

- **S_max** (Maximum sink):
  - Vegetation uptake rate
  - Low (sparse): 1e-8 1/s
  - Medium: 1e-7 1/s ✅
  - High (dense forest): 1e-6 1/s

- **zb_initial** (Initial water table):
  - Measured depth [m]
  - Or estimate from data

- **ic_type** (Initial condition):
  - `'obs'`: Use measured profile (best if available) ✅
  - `'linear'`: Linear interpolation surface → water table
  - `'hydrostatic'`: h(z) = -zb - z

---

## Complete Examples

### **Example 1: Simple Site (Single Header, Percentage Units)**

```yaml
# Simple agricultural site with 4 sensors
data:
  path: 'data/farm_site_2021.xlsx'
  start_date: '2021-05-01'
  end_date: '2021-05-31'
  interpolate: true
  max_gap_hours: 6

  column_mapping:
    datetime: 'timestamp'
    multi_level_header: false
    unit_conversion: 0.01  # Data in percentage

    depths:
      5cm: 'soil_moisture_A'
      15cm: 'soil_moisture_B'
      30cm: 'soil_moisture_C'
      60cm: 'soil_moisture_D'

soil:
  theta_s: 0.41
  theta_r: 0.05
  alpha: 2.5
  n: 1.5
  Ks: 5.0e-6
  l: 0.5

training:
  n_epochs: 5000
  learning_rate: 1.0e-4
  device: 'auto'
  seed: 42

network:
  h_net: {hidden_dim: 64, num_layers: 4}
  zb_net: {hidden_dim: 32, num_layers: 3}

sampling:
  batch_size: 500            # Points sampled per iteration
  batch_size: 500
  boundary_ratio: 0.7        # Fraction near boundaries
  boundary_ratio: 0.7
  temperature: 0.3

boundary_sampling:
  batch_size_bc: 450
  interp_ratio: 0.30
  neighbor_ratio: 0.00
  baseline_ratio: 0.70
  neighbor_expansion: 0
  gradient_threshold: 0.7
  power: 2.0

optimization:
  weight_update_freq: 10000000000
  weight_lr: 0.1
  use_initial_scales: true
  use_amp: false
  use_multi_gpu: true
  grad_accumulation_steps: 1

checkpointing:
  dir: 'checkpoints_farm'
  freq: 1000
  keep_last_n: 3

physics:
  Sy: 0.25
  zr: 0.8
  L: 3.0
  S_max: 1.0e-7
  zb_initial: 4.5
  ic_type: 'obs'

output:
  dir: 'outputs_farm'
```

### **Example 2: Research Site with WTD**

```yaml
# Research site with water table measurements
data:
  path: 'data/research_site.xlsx'
  start_date: '2022-07-01'
  end_date: '2022-07-31'
  interpolate: true
  max_gap_hours: 6

  column_mapping:
    datetime: 'Date_Time'
    multi_level_header: false
    unit_conversion: 1.0  # Already in m³/m³

    depths:
      2cm: 'TDR_shallow'
      15cm: 'TDR_mid1'
      30cm: 'TDR_mid2'
      60cm: 'TDR_deep'

    # Include water table depth
    wtd: 'WaterTableDepth'
    wtd_sign: -1.0  # Convert negative to positive

soil:
  theta_s: 0.45
  theta_r: 0.08
  alpha: 3.0
  n: 1.6
  Ks: 8.0e-6
  l: 0.5

training:
  n_epochs: 10000
  learning_rate: 1.0e-4
  device: 'cuda'
  seed: 123

network:
  h_net: {hidden_dim: 128, num_layers: 5}  # Larger network
  zb_net: {hidden_dim: 64, num_layers: 4}

sampling:
  batch_size: 500            # Points sampled per iteration
  batch_size: 800
  boundary_ratio: 0.7        # Fraction near boundaries
  boundary_ratio: 0.8
  temperature: 0.2

boundary_sampling:
  batch_size_bc: 600
  interp_ratio: 0.40
  neighbor_ratio: 0.05
  baseline_ratio: 0.55
  neighbor_expansion: 1
  gradient_threshold: 0.8
  power: 2.5

optimization:
  weight_update_freq: 10000000000
  weight_lr: 0.1
  use_initial_scales: true
  use_amp: true  # Use mixed precision
  use_multi_gpu: true
  grad_accumulation_steps: 1

checkpointing:
  dir: 'checkpoints_research'
  freq: 2000
  keep_last_n: 5

physics:
  Sy: 0.30
  zr: 0.5
  L: 5.0
  S_max: 5.0e-8
  zb_initial: 8.2
  ic_type: 'obs'

output:
  dir: 'outputs_research'
```

---

## Common Patterns

### **Pattern 1: Quick Test Run**

Fast training for debugging:

```yaml
training:
  n_epochs: 1000       # Fewer epochs
  learning_rate: 5.0e-4  # Higher LR

network:
  h_net: {hidden_dim: 32, num_layers: 3}  # Smaller network
  zb_net: {hidden_dim: 16, num_layers: 2}

sampling:
  batch_size: 500            # Points sampled per iteration
  batch_size: 300

checkpointing:
  freq: 500            # Save often
  keep_last_n: 2
```

### **Pattern 2: High Quality Run**

Best accuracy (slower):

```yaml
training:
  n_epochs: 50000      # Many epochs
  learning_rate: 1.0e-4  # Standard LR

network:
  h_net: {hidden_dim: 128, num_layers: 6}  # Large network
  zb_net: {hidden_dim: 64, num_layers: 4}

sampling:
  batch_size: 500            # Points sampled per iteration
  batch_size: 1000
  boundary_ratio: 0.7        # Fraction near boundaries

optimization:
  use_amp: true        # Mixed precision for speed
```

### **Pattern 3: Fine-Tuning Config**

For transfer learning:

```yaml
training:
  n_epochs: 5000       # Fewer than base training
  learning_rate: 5.0e-5  # LOWER than base training

optimization:
  weight_update_freq: 10000000000  # Fixed weights
```

---

## Troubleshooting

### **Problem: FileNotFoundError**

**Error:**
```
FileNotFoundError: data/my_file.xlsx
```

**Solution:**
Check path is relative to project root:
```yaml
# ✅ Correct
data:
  path: 'data/my_file.xlsx'

# ❌ Wrong
data:
  path: '/absolute/path/data/my_file.xlsx'
```

---

### **Problem: KeyError - Column not found**

**Error:**
```
KeyError: 'SWC_2cm'
```

**Solution:**
Check column names match Excel exactly:
1. Open Excel file
2. Look at column headers
3. Copy exact names (case-sensitive!)

For multi-level headers:
```yaml
# ✅ Correct
depths:
  2cm: ['Logger_ID', 'Sensor_Name']

# ❌ Wrong (typo)
depths:
  2cm: ['Logger_ID', 'Sensor_name']  # lowercase 'n'
```

---

### **Problem: Training is very slow**

**Solutions:**

1. **Use GPU:**
```yaml
training:
  device: 'cuda'  # Instead of 'cpu'
```

2. **Reduce cache/batch size:**
```yaml
sampling:
  batch_size: 500            # Points sampled per iteration
  batch_size: 300    # Instead of 1000
```

3. **Use mixed precision:**
```yaml
optimization:
  use_amp: true
```

---

### **Problem: Loss is NaN or explodes**

**Solutions:**

1. **Lower learning rate:**
```yaml
training:
  learning_rate: 1.0e-5  # Instead of 1.0e-3
```

2. **Check soil parameters:**
```yaml
soil:
  n: 1.45  # Must be > 1
  Ks: 2.0e-5  # Must be positive
```

3. **Check data quality:**
- Are there NaN values?
- Are units correct (fraction vs percentage)?

---

## Quick Reference

### **YAML Syntax Rules**

```yaml
# Comments start with #

# Key-value pairs
key: value

# Numbers
integer: 42
float: 3.14
scientific: 1.0e-5

# Strings (quotes optional but recommended)
string1: 'hello'
string2: "world"
path: 'data/file.xlsx'

# Booleans
flag1: true
flag2: false

# Lists (two ways)
list1: [1, 2, 3]
list2:
  - item1
  - item2
  - item3

# Dictionaries (nested)
parent:
  child1: value1
  child2: value2
  nested:
    key: value

# Multi-level headers use lists
datetime: ['Level0', 'Level1']
```

### **Units Reference**

| Parameter | Unit | Example |
|-----------|------|---------|
| θs, θr | [-] | 0.46 |
| α | [1/m] | 4.0 |
| n | [-] | 1.45 |
| Ks | [m/s] | 2.0e-5 |
| Sy | [-] | 0.3 |
| zr, L, zb | [m] | 0.5, 4.0, 6.1 |
| S_max | [1/s] | 1.0e-7 |
| Time | [s] or dates | '2020-06-01' |

---

## Summary Checklist

Before running your config:

- [ ] **Data path** is correct
- [ ] **Dates** are in 'YYYY-MM-DD' format with quotes
- [ ] **Column names** match Excel exactly (case-sensitive)
- [ ] **multi_level_header** matches your data structure
- [ ] **unit_conversion** is correct (1.0 or 0.01)
- [ ] **Depths ordered** shallowest to deepest
- [ ] **Soil parameters** are reasonable for your soil type
- [ ] **n_epochs** is appropriate (1000 test, 5000+ production)
- [ ] **device** is set ('auto' is safe default)
- [ ] **All sections present** (data, soil, training, network, etc.)

---

**Last Updated**: 2025-01-28
**Version**: 1.0.0
