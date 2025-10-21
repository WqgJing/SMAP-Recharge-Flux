#!/usr/bin/env python3
"""
Forward PINN Training Script - Soil Moisture BC (HPC Version)
Converted from forward_pinn.ipynb

This script trains a Physics-Informed Neural Network (PINN) for Richards equation
using soil moisture boundary conditions from Calhoun Experimental Forest data.

Usage:
    python train_forward_pinn.py

Or with SLURM:
    sbatch run_training.slurm

GPU-OPTIMIZED: 2-3x faster training with automatic GPU utilization
"""

import sys
import os
import argparse
import time

# Add project root to path (3 levels up: script → untitled folder → hpc → project_root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# ============================================================================
# Imports
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch.nn.functional as F
from torch.optim import Adam

# Import PINN models
from src.pinn_models import PressureHeadNet, WaterTableNet, RichardsPINN
# Import normalization helper
from src.normalization_helper import NormalizationHelper
# Import training utilities
from src.training_utils import (
    SamplingHelpers,
    compute_losses,
    apply_weights_and_compute_gradients,
    apply_weights_fixed_mode,
    WeightManager,
    CachePoolManager,
    compute_grad_norm,
    compute_total_grad_norm,
)
# Import training logger
from src.training_logger import TrainingLogger
# Import training function
from src.train_loop import train_pinn_pool_batch_autoweight
# Import gradient-based sampling
from src.gradient_based_sampling import (
    gradient_based_sampling,
    visualize_gradient_sampling,
    compute_gradient_weights
)
# Import data loader
from src.data_loader import load_calhoun_soil_moisture
# Import visualization
from src.visualization import plot_comprehensive_results, plot_training_losses

print("=" * 70)
print("FORWARD PINN TRAINING - SOIL MOISTURE BC")
print("GPU-Optimized Version (2-3x faster)")
print("=" * 70)

# ============================================================================
# Configuration
# ============================================================================

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Forward PINN with Soil Moisture BC')
parser.add_argument('--n_epochs', type=int, default=500000, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--cache_size', type=int, default=80000, help='Cache pool size')
parser.add_argument('--batch_size', type=int, default=4000, help='Batch size')
parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, or cpu')
parser.add_argument('--checkpoint_freq', type=int, default=5000, help='Checkpoint frequency')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_train', help='Checkpoint directory')
parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for plots')
parser.add_argument('--data_path', type=str, default='data/2017_HalfHourly_UTC_ForestSite1.xlsx',
                    help='Path to soil moisture data')
parser.add_argument('--start_date', type=str, default='2017-05-19', help='Start date (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, default='2017-05-26', help='End date (YYYY-MM-DD)')
parser.add_argument('--use_amp', action='store_true', help='Enable mixed precision training (30-50% faster)')
parser.add_argument('--seed', type=int, default=44, help='Random seed')
parser.add_argument("run_name", nargs="?", default="default_run", help="Optional run name")

args = parser.parse_args()

# Set device
if args.device == 'auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

print("✓ Modules loaded")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
print(f"Output directory: {args.output_dir}")

# ============================================================================
# LOAD SOIL MOISTURE DATA FROM CALHOUN EXPERIMENTAL FOREST
# ============================================================================

print("\n" + "=" * 70)
print("LOADING SOIL MOISTURE DATA")
print("=" * 70)

# Get absolute path to data file (use project_root already defined at top)
data_path = os.path.join(project_root, args.data_path)

print(f"Data file: {data_path}")
print(f"Date range: {args.start_date} to {args.end_date}")

# Load soil moisture data
soil_data = load_calhoun_soil_moisture(
    filepath=data_path,
    interpolate=True,
    max_gap_hours=6,
    start_date=args.start_date,
    end_date=args.end_date,
    verbose=True
)

# Extract cleaned data
datetime_col = soil_data['datetime']
times_seconds = soil_data['times_seconds']
theta_2cm = soil_data['theta_2cm']
theta_15cm = soil_data['theta_15cm']
theta_30cm = soil_data['theta_30cm']
theta_40cm = soil_data['theta_40cm']
theta_60cm = soil_data['theta_60cm']
theta_80cm = soil_data['theta_80cm']
depths_names = soil_data['depths_names']

print(f"\nData Summary:")
print(f"  Total time points: {len(times_seconds)}")
print(f"  Duration: {times_seconds[-1]/86400:.2f} days")
print(f"  Surface moisture (2cm) range: {np.nanmin(theta_2cm):.4f} - {np.nanmax(theta_2cm):.4f} m³/m³")
print(f"  Data completeness (2cm): {soil_data['interpolation_stats']['2cm']['completeness']:.1f}%")

# ============================================================================
# PREPARE MOISTURE BC DATA
# ============================================================================

print("\n" + "=" * 70)
print("PREPARING MOISTURE BOUNDARY CONDITION")
print("=" * 70)

# Remove NaN values from surface moisture data
valid_mask = ~pd.isna(theta_2cm)
theta0_times = times_seconds[valid_mask]
theta0_values = theta_2cm[valid_mask]

# Convert to numpy arrays
theta0_times = np.array(theta0_times, dtype=np.float64)
theta0_values = np.array(theta0_values, dtype=np.float64)

theta0_data = (theta0_times, theta0_values)

print(f"Moisture BC Data:")
print(f"  Surface BC: θ(z=0, t) = θ₀(t)")
print(f"  Data points: {len(theta0_data[0])}")
print(f"  Duration: {theta0_data[0][-1]/86400:.1f} days")
print(f"  Moisture range: {min(theta0_data[1]):.4f} - {max(theta0_data[1]):.4f} m³/m³")

# ============================================================================
# SOIL PARAMETERS
# ============================================================================

soil_params = {
    'theta_s': 0.40,   # saturated water content [-]
    'theta_r': 0.01,   # residual water content [-]
    'alpha': 3.4,      # 1/m (van Genuchten α)
    'n': 1.7,          # van Genuchten n (>1)
    'Ks': 1.0e-5,      # m/s (saturated conductivity)
    'l': 0.50,         # Mualem pore-connectivity parameter
}

print(f"\nSoil Parameters:")
for key, val in soil_params.items():
    print(f"  {key}: {val}")

# ============================================================================
# GRADIENT-BASED SAMPLING ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("GRADIENT-BASED SAMPLING ANALYSIS")
print("=" * 70)

# Convert to tensors
theta0_times_t = torch.tensor(theta0_times, dtype=torch.float32).to(device)
theta0_values_t = torch.tensor(theta0_values, dtype=torch.float32).to(device)

# Compute gradient weights
gradient_weights, gradients = compute_gradient_weights(theta0_times_t, theta0_values_t, device)

# ✅ GPU-OPTIMIZED: Batch gradient statistics computation before sync
grad_abs = gradients.abs()
grad_max = grad_abs.max()
grad_mean = grad_abs.mean()
grad_min = grad_abs.min()

print(f"\nGradient Statistics:")
print(f"  Max |dθ/dt|:  {grad_max.item():.3e} m³/m³/s")
print(f"  Mean |dθ/dt|: {grad_mean.item():.3e} m³/m³/s")
print(f"  Min |dθ/dt|:  {grad_min.item():.3e} m³/m³/s")

# Sampling parameters (matching notebook)
batch_size_bc = 250
interp_ratio = 0.30
neighbor_ratio = 0.00
baseline_ratio = 0.70
neighbor_expansion = 0
power = 2.0
gradient_threshold = 0.7

# Generate test samples
t_bc_gradient, sample_info = gradient_based_sampling(
    theta0_times_t,
    theta0_values_t,
    batch_size_bc,
    device=device,
    use_interpolation=True,
    interp_ratio=interp_ratio,
    neighbor_ratio=neighbor_ratio,
    baseline_ratio=baseline_ratio,
    neighbor_expansion=neighbor_expansion,
    gradient_threshold=gradient_threshold,
    power=power
)

print(f"\nGenerated {len(t_bc_gradient)} gradient-weighted samples")
print(f"  Interpolated:  {sample_info['n_interp_samples']} ({sample_info['actual_ratios']['interp']*100:.1f}%)")
print(f"  Neighbors:     {sample_info['n_neighbor_samples']} ({sample_info['actual_ratios']['neighbor']*100:.1f}%)")
print(f"  Baseline:      {sample_info['n_baseline_samples']} ({sample_info['actual_ratios']['baseline']*100:.1f}%)")

# ============================================================================
# PREPARE INITIAL CONDITION FROM MEASURED THETA PROFILE
# ============================================================================

print("\n" + "=" * 70)
print("CREATING INITIAL CONDITION FROM MEASURED THETA PROFILE")
print("=" * 70)

ic_depths = []
ic_theta = []

depth_mapping = {
    '2cm': (0.02, theta_2cm),
    '15cm': (0.15, theta_15cm),
    '30cm': (0.30, theta_30cm),
    '40cm': (0.40, theta_40cm),
    '60cm': (0.60, theta_60cm),
    '80cm': (0.80, theta_80cm),
}

for depth_name, (depth_m, theta_series) in depth_mapping.items():
    # Get first valid (non-NaN) value
    valid_mask = ~pd.isna(theta_series)
    if valid_mask.any():
        if isinstance(theta_series, pd.Series):
            theta_ic = theta_series[valid_mask].iloc[0]
        else:
            theta_ic = theta_series[valid_mask][0]
        ic_depths.append(depth_m)
        ic_theta.append(theta_ic)
        print(f"  {depth_name:6s}: θ = {theta_ic:.4f} m³/m³")

ic_profile = {
    'depths': ic_depths,  # meters below surface
    'theta': ic_theta     # volumetric water content
}

print(f"\nIC Profile has {len(ic_depths)} measurement points")

# IC type: 'obs', 'linear', or 'hydrostatic'
ic_type = 'hydrostatic'
print(f"Using IC type: {ic_type}")

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

# Network configurations
h_net_config = {'hidden_dim': 64, 'num_layers': 4}
zb_net_config = {'hidden_dim': 32, 'num_layers': 3}

print(f"Configuration:")
print(f"  Epochs: {args.n_epochs}")
print(f"  Learning rate: {args.learning_rate}")
print(f"  Cache size: {args.cache_size}")
print(f"  Batch size: {args.batch_size}")
print(f"  Resample freq: 100")
print(f"  Boundary ratio: 0.7")
print(f"  High residual ratio: 0.6")
print(f"  Temperature: 0.3")
print(f"  Mixed precision (AMP): {args.use_amp}")
print(f"  Checkpoint frequency: {args.checkpoint_freq}")
print(f"  Checkpoint directory: {args.checkpoint_dir}")

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "=" * 70)
print("STARTING PINN TRAINING")
print("=" * 70)

start_time = time.time()

model_pool, losses_pool, comps_pool, sample_losses, sample_comps, sample_epochs = train_pinn_pool_batch_autoweight(
    soil_params=soil_params,
    theta0_data=theta0_data,  # Moisture BC data
    Sy=0.3,
    zr=0.5,
    h_net_config=h_net_config,
    zb_net_config=zb_net_config,
    L=4.0,
    S_max=1e-7,
    n_epochs=args.n_epochs,
    learning_rate=args.learning_rate,
    zb_initial=6.1,
    weight_update_freq=1e10,  # Fixed weights
    weight_lr=0.1,
    use_initial_scales=True,

    # Pool + batch parameters
    cache_size=args.cache_size,
    batch_size=args.batch_size,
    resample_freq=100,
    boundary_ratio=0.7,
    high_residual_ratio=0.6,
    temperature=0.3,

    # Gradient-based boundary sampling parameters
    batch_size_bc=batch_size_bc,
    interp_ratio=interp_ratio,
    neighbor_ratio=neighbor_ratio,
    baseline_ratio=baseline_ratio,
    gradient_neighbor_expansion=neighbor_expansion,
    gradient_threshold=gradient_threshold,
    gradient_power=power,

    # Measured initial condition profile
    ic_profile=ic_profile,
    ic_type=ic_type,

    # GPU optimizations
    device=device,
    use_amp=args.use_amp,
    use_multi_gpu=True,

    # Checkpointing
    checkpoint_dir=args.checkpoint_dir,
    checkpoint_freq=args.checkpoint_freq,
)

elapsed_time = time.time() - start_time

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Total training time: {elapsed_time/3600:.2f} hours ({elapsed_time:.1f} seconds)")
print(f"Time per epoch: {elapsed_time/args.n_epochs*1000:.1f} ms")
print(f"Final loss: {losses_pool[-1]:.3e}")

# ============================================================================
# VISUALIZATION AND RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Comprehensive results plot
print("Creating comprehensive results plot...")
plot_comprehensive_results(model_pool, theta0_data, soil_params, device=device)
output_path = os.path.join(args.output_dir, 'results_comprehensive.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# Training loss plot
print("Creating training loss plot...")
plot_training_losses(losses_pool, comps_pool, sample_losses, sample_comps, sample_epochs)
output_path = os.path.join(args.output_dir, 'training_losses.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# Save training summary
summary_path = os.path.join(args.output_dir, 'training_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("FORWARD PINN TRAINING SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("Configuration:\n")
    f.write(f"  Data: {args.data_path}\n")
    f.write(f"  Date range: {args.start_date} to {args.end_date}\n")
    f.write(f"  Epochs: {args.n_epochs}\n")
    f.write(f"  Learning rate: {args.learning_rate}\n")
    f.write(f"  Cache size: {args.cache_size}\n")
    f.write(f"  Batch size: {args.batch_size}\n")
    f.write(f"  Device: {device}\n")
    f.write(f"  Mixed precision: {args.use_amp}\n\n")

    f.write("Results:\n")
    f.write(f"  Total training time: {elapsed_time/3600:.2f} hours\n")
    f.write(f"  Time per epoch: {elapsed_time/args.n_epochs*1000:.1f} ms\n")
    f.write(f"  Final total loss: {losses_pool[-1]:.3e}\n")
    f.write(f"  Final PDE loss: {comps_pool['pde'][-1]:.3e}\n")
    f.write(f"  Final surface BC loss: {comps_pool['surf'][-1]:.3e}\n")
    f.write(f"  Final WT head loss: {comps_pool['wt_head'][-1]:.3e}\n")
    f.write(f"  Final WT kin loss: {comps_pool['wt_kin'][-1]:.3e}\n")
    f.write(f"  Final IC h loss: {comps_pool['ic_h'][-1]:.3e}\n")
    f.write(f"  Final IC zb loss: {comps_pool['ic_zb'][-1]:.3e}\n\n")

    f.write("Soil Parameters:\n")
    for key, val in soil_params.items():
        f.write(f"  {key}: {val}\n")
    f.write("\n")

    f.write("Outputs:\n")
    f.write(f"  Checkpoints: {args.checkpoint_dir}/\n")
    f.write(f"  Plots: {args.output_dir}/\n")
    f.write("=" * 70 + "\n")

print(f"✓ Saved: {summary_path}")

print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)
print(f"\nOutputs:")
print(f"  Checkpoints: {args.checkpoint_dir}/")
print(f"  Plots: {args.output_dir}/")
print(f"  Summary: {summary_path}")
print("=" * 70)
