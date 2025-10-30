#!/usr/bin/env python3
"""
Simple PINN Training Script - NEW UNIFIED API

This is the new simplified training script using the PINNDataset class.
Everything comes from YAML - just specify the config file!

Usage:
    # Train from scratch
    python hpc/train_simple.py --config configs/baseline.yaml
    python hpc/train_simple.py --config configs/us_uaf_2019.yaml

    # Resume from checkpoint
    python hpc/train_simple.py --config configs/baseline.yaml --load-checkpoint checkpoints_train/checkpoint_epoch_5000.pt

Key Benefits:
- 🚀 Ultra-simple: 3 lines to train!
- 📁 YAML-driven: All parameters in config
- 🔄 Works with ANY dataset format
- ✅ Automatic depth handling
- 💾 Checkpoint loading/resuming
- 🎯 Clean, maintainable code
"""

import sys
import os
import argparse
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt

# Import NEW unified API
from src.dataset import PINNDataset
from src.train_loop import train_pinn
from src.visualization import plot_results, plot_losses

print("=" * 70)
print("SIMPLE PINN TRAINING - NEW UNIFIED API")
print("=" * 70)

# ============================================================================
# Parse Arguments
# ============================================================================

parser = argparse.ArgumentParser(description='Train PINN with unified dataset API')
parser.add_argument('--config', type=str, required=True,
                    help='Path to YAML config file')
parser.add_argument('--device', type=str, default='auto',
                    help='Device: auto, cuda, or cpu (default: auto)')
parser.add_argument('--load-checkpoint', type=str, default=None,
                    help='Path to checkpoint file to resume training from')
args = parser.parse_args()

print(f"\nConfig: {args.config}")
print(f"Device: {args.device}")
if args.load_checkpoint:
    print(f"Resume from: {args.load_checkpoint}")

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Create checkpoint subdirectory inside output directory
checkpoint_dir = os.path.join(output_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# ============================================================================
# Initialize Dataset (Everything in one object!)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1: INITIALIZE DATASET")
print("=" * 70)

dataset = PINNDataset(args.config, verbose=True)

# ============================================================================
# Train Model (One function call!)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: TRAIN PINN MODEL")
print("=" * 70)

start_time = time.time()

# 🚀 THIS IS IT! One line to train!
model, losses, comps, sample_losses, sample_comps, sample_epochs = train_pinn(
    dataset,
    device=args.device,
    checkpoint_dir=checkpoint_dir,  # Use output-specific checkpoint directory
    checkpoint_path=args.load_checkpoint  # Pass checkpoint path if provided
)

elapsed_time = time.time() - start_time

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Total time: {elapsed_time/3600:.2f} hours ({elapsed_time:.1f} seconds)")
print(f"Time per epoch: {elapsed_time/dataset.n_epochs*1000:.1f} ms")
print(f"Final loss: {losses[-1]:.3e}")

# ============================================================================
# Visualization (One function call!)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: GENERATE VISUALIZATIONS")
print("=" * 70)



# 🎨 Plot results - automatically handles all depths, WTD, etc!
print("Creating comprehensive results plot...")
# Get actual device from model (args.device might be 'auto')
actual_device = next(model.parameters()).device
plot_results(model, dataset, device=actual_device)
plt.savefig(f'{output_dir}/results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/results.png")

# Plot training losses
print("Creating training loss plot...")
plot_losses(losses, comps, sample_losses, sample_comps, sample_epochs)
plt.savefig(f'{output_dir}/losses.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/losses.png")

# ============================================================================
# Save Summary
# ============================================================================

summary_path = f'{output_dir}/summary.txt'
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("SIMPLE PINN TRAINING SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("Configuration:\n")
    f.write(f"  Config file: {args.config}\n")
    f.write(f"  Data: {dataset.config.data_path}\n")
    f.write(f"  Date range: {dataset.config.start_date} to {dataset.config.end_date}\n")
    f.write(f"  Observation depths: {len(dataset.obs_depths)} → {dataset.obs_depths}\n")
    f.write(f"  WTD available: {dataset.has_wtd()}\n\n")

    f.write("Soil Parameters (van Genuchten-Mualem):\n")
    for key, val in dataset.soil_params.items():
        f.write(f"  {key}: {val}\n")
    f.write("\n")

    f.write("Physics Parameters:\n")
    f.write(f"  Sy (specific yield): {dataset.Sy}\n")
    f.write(f"  zr (root zone depth): {dataset.zr} m\n")
    f.write(f"  L (characteristic length): {dataset.L} m\n")
    f.write(f"  S_max (max sink): {dataset.S_max} 1/s\n")
    f.write(f"  zb_initial (initial water table): {dataset.zb_initial} m\n")
    f.write(f"  ic_type: {dataset.ic_type}\n\n")

    f.write("Network Architecture:\n")
    f.write(f"  h_net (pressure head):\n")
    f.write(f"    hidden_dim: {dataset.h_net_config['hidden_dim']}\n")
    f.write(f"    num_layers: {dataset.h_net_config['num_layers']}\n")
    f.write(f"  zb_net (water table):\n")
    f.write(f"    hidden_dim: {dataset.zb_net_config['hidden_dim']}\n")
    f.write(f"    num_layers: {dataset.zb_net_config['num_layers']}\n\n")

    f.write("Sampling Configuration:\n")
    f.write(f"  PDE collocation:\n")
    f.write(f"    cache_size: {dataset.cache_size}\n")
    f.write(f"    batch_size: {dataset.batch_size}\n")
    f.write(f"    resample_freq: {dataset.resample_freq}\n")
    f.write(f"    boundary_ratio: {dataset.boundary_ratio}\n")
    f.write(f"    high_residual_ratio: {dataset.high_residual_ratio}\n")
    f.write(f"    temperature: {dataset.temperature}\n")
    f.write(f"  Boundary condition:\n")
    f.write(f"    batch_size_bc: {dataset.config.batch_size_bc}\n")
    f.write(f"    interp_ratio: {dataset.config.interp_ratio}\n")
    f.write(f"    neighbor_ratio: {dataset.config.neighbor_ratio}\n")
    f.write(f"    baseline_ratio: {dataset.config.baseline_ratio}\n")
    f.write(f"    gradient_neighbor_expansion: {dataset.config.gradient_neighbor_expansion}\n")
    f.write(f"    gradient_threshold: {dataset.config.gradient_threshold}\n")
    f.write(f"    gradient_power: {dataset.config.gradient_power}\n\n")

    f.write("Training:\n")
    f.write(f"  Epochs: {dataset.n_epochs}\n")
    f.write(f"  Learning rate: {dataset.learning_rate}\n")
    f.write(f"  Device: {args.device}\n")
    if args.load_checkpoint:
        f.write(f"  Resumed from: {args.load_checkpoint}\n")
    f.write(f"  Weight update freq: {dataset.config.weight_update_freq}\n")
    f.write(f"  Weight lr: {dataset.config.weight_lr}\n")
    f.write(f"  Use initial scales: {dataset.config.use_initial_scales}\n")
    f.write(f"  Total time: {elapsed_time/3600:.2f} hours\n")
    f.write(f"  Time per epoch: {elapsed_time/dataset.n_epochs*1000:.1f} ms\n\n")

    f.write("Results:\n")
    f.write(f"  Final total loss: {losses[-1]:.3e}\n")
    f.write(f"  Final PDE loss: {comps['pde'][-1]:.3e}\n")
    f.write(f"  Final surface BC loss: {comps['surf'][-1]:.3e}\n")
    f.write(f"  Final WT head loss: {comps['wt_head'][-1]:.3e}\n")
    f.write(f"  Final IC loss: {comps['ic_h'][-1]:.3e}\n\n")

    f.write("Outputs:\n")
    f.write(f"  Directory: {output_dir}/\n")
    f.write(f"  Checkpoints: {checkpoint_dir}/\n")
    f.write(f"  Final checkpoint: {checkpoint_dir}/checkpoint_final.pt\n")
    f.write("=" * 70 + "\n")

print(f"✓ Saved: {summary_path}")

print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)
print(f"\nOutputs saved to: {output_dir}/")
print(f"  - results.png")
print(f"  - losses.png")
print(f"  - summary.txt")
print(f"\nCheckpoints saved to: {checkpoint_dir}/")
print(f"  - checkpoint_final.pt (use this for inference/fine-tuning)")
print("=" * 70)
