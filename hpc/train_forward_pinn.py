#!/usr/bin/env python3
"""
Forward PINN Training Script (HPC Version)
Converted from forward_pinn.ipynb

Usage:
    python train_forward_pinn.py

Or with SLURM:
    sbatch run_training.slurm
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# Imports
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam

# Import PINN models from external module
from src.pinn_models import PressureHeadNet, WaterTableNet, RichardsPINN
# Import normalization helper from external module
from src.normalization_helper import NormalizationHelper
# Import sampling helpers and loss computation functions from external module
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
# Import TrainingLogger from external module
from src.training_logger import TrainingLogger
# Import training function
from src.train_loop import train_pinn_pool_batch_autoweight
# Import gradient-based sampling
from src.gradient_based_sampling import (
    gradient_based_sampling,
    compute_gradient_weights
)
# Import visualization
from src.visualization import plot_comprehensive_results, plot_training_losses

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed
torch.manual_seed(44)
np.random.seed(44)

print("✓ Modules loaded")

# ============================================================================
# Load Data
# ============================================================================

print("\nLoading Calhoun precipitation data...")
data = np.genfromtxt('../data/CalhounPrecipWY2015.csv', delimiter=',', skip_header=1, dtype=str)

# Extract second column (precipitation in mm at 5-min resolution)
precip_mm_5min = np.array([float(row[1]) for row in data])

print(f"Original data: {len(precip_mm_5min)} points at 5-min resolution")

# Group every 12 points (60 minutes)
n_groups = len(precip_mm_5min) // 12
precip_mm_30min = []

for i in range(n_groups):
    # Sum precipitation over 12 points (60 minutes)
    total_mm_30min = precip_mm_5min[i*12:(i+1)*12].sum()
    precip_mm_30min.append(total_mm_30min)

precip_mm_30min = np.array(precip_mm_30min)

# Convert to m/s
q = precip_mm_30min / 1000.0 / 3600.0  # m/s
q = -q[24*150 : 24 * 165]
t = np.arange(len(q)) * 3600

print(f"Resampled data: {len(t)} points at 30-min resolution")
print(f"Time span: {t[-1]/86400:.2f} days")
print(f"Precipitation rate range: {q.min():.2e} to {q.max():.2e} m/s")

# Create q_actual_flux tuple
q_cczo_flux = (t.tolist(), q.tolist())

print("✓ Data loaded successfully")

# ============================================================================
# Gradient-Based Sampling Test
# ============================================================================

print("\n" + "="*70)
print("GRADIENT-BASED SAMPLING ANALYSIS")
print("="*70)

# Prepare data
q0_times_t = torch.tensor(q_cczo_flux[0], dtype=torch.float32).to(device)
q0_values = torch.tensor(q_cczo_flux[1], dtype=torch.float32).to(device)

# Compute gradient weights
gradient_weights, gradients = compute_gradient_weights(q0_times_t, q0_values, device)

print(f"\nGradient Statistics:")
print(f"  Max |dq/dt|:  {gradients.abs().max().item():.3e} m/s²")
print(f"  Mean |dq/dt|: {gradients.abs().mean().item():.3e} m/s²")
print(f"  Min |dq/dt|:  {gradients.abs().min().item():.3e} m/s²")

# Set sampling parameters
batch_size_bc = 250
interp_ratio = 0.80
neighbor_ratio = 0.05
baseline_ratio = 0.15
neighbor_expansion = 2
power = 2.0
gradient_threshold = 0.7

# Generate samples
t_bc_gradient, sample_info = gradient_based_sampling(
    q0_times_t,
    q0_values,
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
print("="*70)

# ============================================================================
# Training Configuration
# ============================================================================

soil_params = {
    'theta_s': 0.4,
    'theta_r': 0.1,
    'alpha': 0.005,
    'n': 1.3,
    'Ks': 1e-5,
    'l': 0.5
}

# Network configurations
h_net_config = {'hidden_dim': 64, 'num_layers': 4}
zb_net_config = {'hidden_dim': 32, 'num_layers': 3}

print("\n" + "="*70)
print("STARTING PINN TRAINING")
print("="*70)
print("Configuration:")
print(f"  Epochs: 40000")
print(f"  Cache size: 10000")
print(f"  Batch size: 500")
print(f"  Resample freq: 100 (optimized)")
print(f"  High residual ratio: 0.6 (optimized)")
print(f"  Temperature: 0.3 (optimized)")
print("="*70 + "\n")

# ============================================================================
# Training
# ============================================================================

model_pool, losses_pool, comps_pool, sample_losses, sample_comps, sample_epochs = train_pinn_pool_batch_autoweight(
    soil_params=soil_params,
    q0_data=q_cczo_flux,
    Sy=0.3,
    zr=0.5,
    h_net_config=h_net_config,
    zb_net_config=zb_net_config,
    L=4.0,
    S_max=1e-7,
    n_epochs=40000,
    learning_rate=1e-3,
    zb_initial=1,
    weight_update_freq=1e10,
    weight_lr=0.1,
    use_initial_scales=True,

    # Optimized pool + batch parameters
    cache_size=10000,
    batch_size=500,
    resample_freq=100,              # Optimized: more responsive updates
    boundary_ratio=0.7,
    high_residual_ratio=0.6,        # Optimized: stronger focus on hard regions
    temperature=0.3,                # Optimized: sharper probability concentration

    # Gradient-based boundary sampling parameters
    batch_size_bc=batch_size_bc,
    interp_ratio=interp_ratio,
    neighbor_ratio=neighbor_ratio,
    baseline_ratio=baseline_ratio,
    gradient_neighbor_expansion=neighbor_expansion,
    gradient_threshold=gradient_threshold,
    gradient_power=power,

    # Device and checkpointing
    device=device,
    checkpoint_dir='checkpoints_train',
    checkpoint_freq=5000,
)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

# ============================================================================
# Visualization and Results
# ============================================================================

print("\nGenerating visualizations...")

# Comprehensive results plot
plot_comprehensive_results(model_pool, q_cczo_flux, soil_params, device=device)
plt.savefig('results_comprehensive.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results_comprehensive.png")

# Training loss plot
plot_training_losses(losses_pool, comps_pool, sample_losses, sample_comps, sample_epochs)
plt.savefig('training_losses.png', dpi=150, bbox_inches='tight')
print("✓ Saved: training_losses.png")

print("\n" + "="*70)
print("ALL DONE!")
print("="*70)
print("\nOutputs:")
print("  - Checkpoints: checkpoints_train/")
print("  - Plots: results_comprehensive.png, training_losses.png")
print("="*70)
