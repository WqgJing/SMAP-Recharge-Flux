#!/usr/bin/env python3
"""
Forward PINN Training Script - Config-based Version

Uses YAML configuration files for all hyperparameters.
Loads preprocessed data from explore_data.ipynb.

Usage:
    # 1. First, run exploration notebook to preprocess data:
    jupyter notebook notebooks/explore_data.ipynb

    # 2. Then train:
    python hpc/train_with_config.py --config configs/baseline.yaml
    python hpc/train_with_config.py --config configs/fast_test.yaml
    python hpc/train_with_config.py --config configs/aggressive_bc_focus.yaml

GPU-OPTIMIZED: 2-3x faster training with automatic GPU utilization
"""

import sys
import os
import argparse
import time
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# Imports
# ============================================================================

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt

# Import PINN components
from src.train_loop import train_pinn_pool_batch_autoweight
from src.visualization import plot_comprehensive_results, plot_training_losses

# Import config loader
from src.config_loader import load_config

print("=" * 70)
print("FORWARD PINN TRAINING - CONFIG-BASED VERSION")
print("GPU-Optimized | Loads Preprocessed Data")
print("=" * 70)

# ============================================================================
# Parse Arguments
# ============================================================================

parser = argparse.ArgumentParser(description='Train Forward PINN with config file')
parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                    help='Path to YAML config file (default: configs/baseline.yaml)')
parser.add_argument('--run_name', type=str, default=None,
                    help='Optional run name for outputs')
args = parser.parse_args()

# ============================================================================
# Load Configuration
# ============================================================================

print("\n" + "=" * 70)
print("LOADING CONFIGURATION")
print("=" * 70)

# Warn if using default config
if args.config == 'configs/baseline.yaml' and '--config' not in sys.argv:
    print("⚠️  Using default config: configs/baseline.yaml")
    print("   (Specify --config to use a different one)")
    print()

config = load_config(args.config)
config.print_summary()

# Set random seed
torch.manual_seed(config.seed)
np.random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

# Set device
device = torch.device(config.device)
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Create output directories
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)

print(f"\n✓ Configuration loaded from: {args.config}")

# ============================================================================
# Load Preprocessed Data
# ============================================================================

print("\n" + "=" * 70)
print("LOADING PREPROCESSED SOIL MOISTURE DATA")
print("=" * 70)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preprocessed_path = os.path.join(project_root, 'data', 'preprocessed_soil_data.pkl')

if not os.path.exists(preprocessed_path):
    print(f"❌ Preprocessed data not found: {preprocessed_path}")
    print(f"\n⚠️  Please run the exploration notebook first:")
    print(f"   jupyter notebook notebooks/explore_data.ipynb")
    print(f"   (Run all cells to generate preprocessed_soil_data.pkl)")
    sys.exit(1)

with open(preprocessed_path, 'rb') as f:
    preprocessed_data = pickle.load(f)

# Extract boundary condition data (surface moisture at 2cm)
theta0_times = preprocessed_data['theta0_times']
theta0_values = preprocessed_data['theta0_values']
theta0_data = (theta0_times, theta0_values)

# Extract time array for all depths (for validation plot)
times_seconds = preprocessed_data['times_seconds']

# Extract IC profile (prepared in exploration notebook)
ic_profile = preprocessed_data['ic_profile']

print(f"✓ Loaded preprocessed data from: {preprocessed_path}")
print(f"  Config used: {preprocessed_data['config_file']}")
print(f"  Date range: {preprocessed_data['date_range']}")
print(f"  Surface BC points: {len(theta0_times)}")
print(f"  Duration: {theta0_times[-1]/86400:.1f} days")
print(f"  Moisture range: {min(theta0_values):.4f} - {max(theta0_values):.4f} m³/m³")
print(f"  IC profile: {len(ic_profile['depths'])} measurement points")
print(f"  IC type: {config.ic_type}")

# ============================================================================
# Training
# ============================================================================

print("\n" + "=" * 70)
print("STARTING PINN TRAINING")
print("=" * 70)

start_time = time.time()

# ✅ ALL PARAMETERS FROM CONFIG - NO HARDCODING!
model, losses, comps, sample_losses, sample_comps, sample_epochs = train_pinn_pool_batch_autoweight(
    # Soil and physics
    soil_params=config.soil_params,
    theta0_data=theta0_data,
    Sy=config.Sy,
    zr=config.zr,
    L=config.L,
    S_max=config.S_max,
    zb_initial=config.zb_initial,
    ic_profile=ic_profile,
    ic_type=config.ic_type,

    # Network architecture
    h_net_config=config.h_net_config,
    zb_net_config=config.zb_net_config,

    # Training
    n_epochs=config.n_epochs,
    learning_rate=config.learning_rate,

    # Cache pool & batch sampling
    cache_size=config.cache_size,
    batch_size=config.batch_size,
    resample_freq=config.resample_freq,
    boundary_ratio=config.boundary_ratio,
    high_residual_ratio=config.high_residual_ratio,
    temperature=config.temperature,

    # Boundary condition sampling
    batch_size_bc=config.batch_size_bc,
    interp_ratio=config.interp_ratio,
    neighbor_ratio=config.neighbor_ratio,
    baseline_ratio=config.baseline_ratio,
    gradient_neighbor_expansion=config.gradient_neighbor_expansion,
    gradient_threshold=config.gradient_threshold,
    gradient_power=config.gradient_power,

    # Optimization
    weight_update_freq=config.weight_update_freq,
    weight_lr=config.weight_lr,
    use_initial_scales=config.use_initial_scales,
    use_amp=config.use_amp,
    use_multi_gpu=config.use_multi_gpu,
    grad_accumulation_steps=config.grad_accumulation_steps,

    # Checkpointing
    checkpoint_dir=config.checkpoint_dir,
    checkpoint_freq=config.checkpoint_freq,
    keep_last_n_checkpoints=config.keep_last_n_checkpoints,

    # Device
    device=device,
)

elapsed_time = time.time() - start_time

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Total training time: {elapsed_time/3600:.2f} hours ({elapsed_time:.1f} seconds)")
print(f"Time per epoch: {elapsed_time/config.n_epochs*1000:.1f} ms")
print(f"Final loss: {losses[-1]:.3e}")

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Prepare observation data for validation subplot
obs_data = {
    'times': times_seconds,  # Already loaded from preprocessed data
    'depths': [-0.02, -0.15, -0.30, -0.40, -0.60, -0.80],  # Negative depths in meters
    'theta': [
        preprocessed_data['theta_2cm'],
        preprocessed_data['theta_15cm'],
        preprocessed_data['theta_30cm'],
        preprocessed_data['theta_40cm'],
        preprocessed_data['theta_60cm'],
        preprocessed_data['theta_80cm'],
    ]
}

# Comprehensive results
print("Creating comprehensive results plot...")
plot_comprehensive_results(model, theta0_data, config.soil_params, device=device, obs_data=obs_data)
output_path = os.path.join(config.output_dir, 'results_comprehensive.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# Training losses
print("Creating training loss plot...")
plot_training_losses(losses, comps, sample_losses, sample_comps, sample_epochs)
output_path = os.path.join(config.output_dir, 'training_losses.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_path}")

# ============================================================================
# Save Summary
# ============================================================================

summary_path = os.path.join(config.output_dir, 'training_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("FORWARD PINN TRAINING SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("Configuration:\n")
    f.write(f"  Config file: {args.config}\n")
    f.write(f"  Data: {config.data_path}\n")
    f.write(f"  Date range: {config.start_date} to {config.end_date}\n")
    f.write(f"  Epochs: {config.n_epochs}\n")
    f.write(f"  Learning rate: {config.learning_rate}\n")
    f.write(f"  Cache size: {config.cache_size}\n")
    f.write(f"  Batch size: {config.batch_size}\n")
    f.write(f"  Device: {device}\n")
    f.write(f"  Mixed precision: {config.use_amp}\n\n")

    f.write("Results:\n")
    f.write(f"  Total training time: {elapsed_time/3600:.2f} hours\n")
    f.write(f"  Time per epoch: {elapsed_time/config.n_epochs*1000:.1f} ms\n")
    f.write(f"  Final total loss: {losses[-1]:.3e}\n")
    f.write(f"  Final PDE loss: {comps['pde'][-1]:.3e}\n")
    f.write(f"  Final surface BC loss: {comps['surf'][-1]:.3e}\n")
    f.write(f"  Final WT head loss: {comps['wt_head'][-1]:.3e}\n")
    f.write(f"  Final WT kin loss: {comps['wt_kin'][-1]:.3e}\n")
    f.write(f"  Final IC h loss: {comps['ic_h'][-1]:.3e}\n")
    f.write(f"  Final IC zb loss: {comps['ic_zb'][-1]:.3e}\n\n")

    f.write("Outputs:\n")
    f.write(f"  Checkpoints: {config.checkpoint_dir}/\n")
    f.write(f"  Plots: {config.output_dir}/\n")
    f.write("=" * 70 + "\n")

print(f"✓ Saved: {summary_path}")

print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)
print(f"\nOutputs:")
print(f"  Config used: {args.config}")
print(f"  Checkpoints: {config.checkpoint_dir}/")
print(f"  Plots: {config.output_dir}/")
print(f"  Summary: {summary_path}")
print("=" * 70)
