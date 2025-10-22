#!/usr/bin/env python3
"""
Standalone script to load checkpoint and visualize with 6 subplots
Run: python visualize_checkpoint.py
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pinn_models import RichardsPINN
from src.normalization_helper import NormalizationHelper
from src.visualization import plot_comprehensive_results
from src.data_loader import load_calhoun_soil_moisture

print("="*70)
print("LOADING CHECKPOINT AND VISUALIZING WITH 6 SUBPLOTS")
print("="*70)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
print("\nLoading soil moisture data...")
soil_data = load_calhoun_soil_moisture(
    filepath='data/2017_HalfHourly_UTC_ForestSite1.xlsx',
    interpolate=True,
    max_gap_hours=6,
    start_date='2017-05-19',
    end_date='2017-05-26',
    verbose=False
)

datetime_col = soil_data['datetime']
times_seconds = soil_data['times_seconds']
theta_2cm = soil_data['theta_2cm']
theta_15cm = soil_data['theta_15cm']
theta_30cm = soil_data['theta_30cm']
theta_40cm = soil_data['theta_40cm']
theta_60cm = soil_data['theta_60cm']
theta_80cm = soil_data['theta_80cm']

# Prepare BC data
valid_mask = ~pd.isna(theta_2cm)
theta0_times = np.array(times_seconds[valid_mask], dtype=np.float64)
theta0_values = np.array(theta_2cm[valid_mask], dtype=np.float64)
theta0_data = (theta0_times, theta0_values)

print(f"✓ Data loaded: {len(theta0_times)} BC points")

# Load checkpoint
checkpoint_path = 'notebooks/checkpoints_train/checkpoint_final.pt'
print(f"\nLoading checkpoint: {checkpoint_path}")

if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)

# Extract configuration from checkpoint structure
norm_params = checkpoint['normalization_params']
soil_params = norm_params['soil_params']
L = norm_params['L']
S_max = norm_params['S_max']
Sy = norm_params['Sy']
zr = norm_params['zr']

# Infer network architecture from state dict
# h_net: hidden_dim=64, num_layers=4 (based on checkpoint analysis)
# zb_net: hidden_dim=32, num_layers=3 (typical configuration)
h_net_config = {'hidden_dim': 64, 'num_layers': 4}
zb_net_config = {'hidden_dim': 32, 'num_layers': 3}

ic_type = 'hydrostatic'  # Default from training
ic_profile = None  # Was not used in this checkpoint

print(f"✓ Checkpoint loaded:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  IC type: {ic_type}")

# Create model
print("\nRecreating model...")
normalizer = NormalizationHelper(soil_params, L=L, S_max=S_max)

model = RichardsPINN(
    soil_params=soil_params,
    theta0_data=theta0_data,  # BC data
    h_net_config=h_net_config,
    zb_net_config=zb_net_config,
    normalizer=normalizer,
    Sy=Sy,
    zr=zr,
    ic_profile=ic_profile,
    ic_type=ic_type,
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded successfully")

# Prepare observation data for subplot 6
obs_data = {
    'times': times_seconds,
    'depths': [-0.02, -0.15, -0.30, -0.40, -0.60, -0.80],
    'theta': [theta_2cm, theta_15cm, theta_30cm, theta_40cm, theta_60cm, theta_80cm]
}

print("\n" + "="*70)
print("GENERATING VISUALIZATION WITH 6 SUBPLOTS")
print("="*70)

# Visualize
plot_comprehensive_results(model, theta0_data, soil_params, device=device, obs_data=obs_data)

print("\n✓ Visualization complete!")
print("  You should see 6 subplots including theta validation (bottom-right)")
print("="*70)
