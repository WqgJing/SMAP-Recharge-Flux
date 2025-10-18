"""
Example: Comparing Surface Flux BC vs Surface Moisture BC

This script demonstrates how to switch between two types of surface boundary conditions:
1. Flux BC (Neumann): Prescribes surface water flux q(z=0, t) = q₀(t)
2. Moisture BC (Dirichlet): Prescribes surface soil moisture θ(z=0, t) = θ₀(t)

The moisture BC is particularly useful for SMAP soil moisture observations.
"""

import sys
sys.path.append('..')

import numpy as np
import torch
from src.train_loop import train_pinn_pool_batch_autoweight

# ==================== SOIL PARAMETERS ====================
# CCZO Sandy Loam (van Genuchten parameters)
soil_params = {
    'theta_s': 0.43,      # Saturated water content [-]
    'theta_r': 0.057,     # Residual water content [-]
    'alpha': 0.75,        # van Genuchten α [1/m]
    'n': 1.89,            # van Genuchten n [-]
    'Ks': 1.23e-5,        # Saturated hydraulic conductivity [m/s]
    'l': 0.5              # Pore connectivity parameter [-]
}

# ==================== SYNTHETIC DATA GENERATION ====================

def generate_synthetic_flux_data(t_days=30, dt_hours=1):
    """
    Generate synthetic surface flux data (ET + rainfall)

    Returns:
        times [s], fluxes [m/s]
    """
    dt = dt_hours * 3600  # Convert to seconds
    t_max = t_days * 86400
    times = np.arange(0, t_max, dt)

    # Diurnal ET cycle (positive = upward flux)
    ET_amplitude = 5e-8  # [m/s] ~ 4.3 mm/day peak
    ET_mean = 2.5e-8     # [m/s] ~ 2.2 mm/day average
    fluxes = ET_mean + ET_amplitude * np.sin(2 * np.pi * times / 86400)

    # Add stochastic rainfall events (negative = downward flux)
    np.random.seed(42)
    n_rain_events = 5
    for _ in range(n_rain_events):
        rain_start = np.random.uniform(0, t_max)
        rain_duration = np.random.uniform(3600, 7200)  # 1-2 hours
        rain_intensity = -np.random.uniform(1e-6, 3e-6)  # [m/s] ~ 3-10 mm/hr

        rain_mask = (times >= rain_start) & (times < rain_start + rain_duration)
        fluxes[rain_mask] = rain_intensity

    return times, fluxes


def generate_synthetic_moisture_data(t_days=30, dt_hours=1, soil_params=None):
    """
    Generate synthetic surface soil moisture data

    Returns:
        times [s], moisture [m³/m³]
    """
    dt = dt_hours * 3600
    t_max = t_days * 86400
    times = np.arange(0, t_max, dt)

    theta_s = soil_params['theta_s']
    theta_r = soil_params['theta_r']

    # Surface moisture varies between field capacity and wilting point
    # Field capacity ~ 70% saturation, wilting point ~ 20% saturation
    theta_fc = theta_r + 0.7 * (theta_s - theta_r)  # ~0.32 for this soil
    theta_wp = theta_r + 0.2 * (theta_s - theta_r)  # ~0.13 for this soil

    # Oscillating pattern with dry-down trend
    theta_mean = (theta_fc + theta_wp) / 2
    theta_amplitude = (theta_fc - theta_wp) / 3

    # Diurnal cycle + slow dry-down
    moisture = theta_mean + theta_amplitude * np.sin(2 * np.pi * times / 86400)
    moisture -= 0.05 * (times / t_max)  # Gradual dry-down

    # Add rainfall rewetting events
    np.random.seed(42)
    n_rain_events = 5
    for i in range(n_rain_events):
        rain_time = np.random.uniform(0, t_max)
        rain_idx = np.argmin(np.abs(times - rain_time))

        # Rapid increase during rain
        moisture[rain_idx:rain_idx+3] = theta_fc
        # Gradual decay after rain
        decay_length = 24  # hours
        decay_indices = np.arange(rain_idx+3, min(rain_idx+3+decay_length, len(times)))
        decay_factor = np.exp(-np.arange(len(decay_indices)) / (decay_length / 3))
        moisture[decay_indices] = (theta_fc - moisture[rain_idx+3-1]) * decay_factor + moisture[rain_idx+3-1]

    # Clamp to valid range
    moisture = np.clip(moisture, theta_r + 0.01, theta_s - 0.01)

    return times, moisture


# ==================== NETWORK CONFIGURATION ====================

h_net_config = {
    'hidden_dim': 64,
    'num_layers': 6
}

zb_net_config = {
    'hidden_dim': 32,
    'num_layers': 4
}

# ==================== TRAINING HYPERPARAMETERS ====================

training_config = {
    'L': 5.0,                    # Domain depth [m]
    'S_max': 1e-7,               # Root uptake [1/s]
    'Sy': 0.3,                   # Specific yield [-]
    'zr': 0.5,                   # Root zone depth [m]
    'zb_initial': 1.5,           # Initial water table depth [m]
    'n_epochs': 5000,            # Training epochs
    'learning_rate': 1e-3,
    'batch_size': 500,
    'batch_size_bc': 250,
    'cache_size': 5000,
    'resample_freq': 100,
    'weight_update_freq': 100,
    'checkpoint_freq': 1000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==================== EXAMPLE 1: FLUX BC (ORIGINAL) ====================

def example_flux_bc():
    """Train PINN with surface flux boundary condition"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Surface Flux BC (Neumann)")
    print("="*70)

    # Generate synthetic flux data
    times, fluxes = generate_synthetic_flux_data(t_days=10, dt_hours=1)
    q0_data = (times, fluxes)

    print(f"\nFlux data:")
    print(f"  Time range: {times[0]/86400:.1f} - {times[-1]/86400:.1f} days")
    print(f"  Flux range: {fluxes.min()*86400*1000:.2f} - {fluxes.max()*86400*1000:.2f} mm/day")
    print(f"  Number of points: {len(times)}")

    # Train with flux BC
    model, losses, _, _, _, _ = train_pinn_pool_batch_autoweight(
        soil_params=soil_params,
        q0_data=q0_data,
        bc_type='flux',  # ← FLUX BC
        h_net_config=h_net_config,
        zb_net_config=zb_net_config,
        checkpoint_dir='checkpoints_flux_bc',
        **training_config
    )

    print(f"\nFinal losses:")
    for key, val in losses.items():
        if len(val) > 0:
            print(f"  {key}: {val[-1]:.6f}")

    return model, losses


# ==================== EXAMPLE 2: MOISTURE BC (NEW) ====================

def example_moisture_bc():
    """Train PINN with surface moisture boundary condition"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Surface Moisture BC (Dirichlet)")
    print("="*70)

    # Generate synthetic moisture data
    times, moisture = generate_synthetic_moisture_data(t_days=10, dt_hours=1, soil_params=soil_params)
    theta0_data = (times, moisture)

    print(f"\nMoisture data:")
    print(f"  Time range: {times[0]/86400:.1f} - {times[-1]/86400:.1f} days")
    print(f"  Moisture range: {moisture.min():.3f} - {moisture.max():.3f} m³/m³")
    print(f"  Number of points: {len(times)}")
    print(f"  Soil parameters: θr={soil_params['theta_r']:.3f}, θs={soil_params['theta_s']:.3f}")

    # Train with moisture BC
    model, losses, _, _, _, _ = train_pinn_pool_batch_autoweight(
        soil_params=soil_params,
        theta0_data=theta0_data,  # ← Pass moisture data instead of flux
        bc_type='moisture',       # ← MOISTURE BC
        h_net_config=h_net_config,
        zb_net_config=zb_net_config,
        checkpoint_dir='checkpoints_moisture_bc',
        **training_config
    )

    print(f"\nFinal losses:")
    for key, val in losses.items():
        if len(val) > 0:
            print(f"  {key}: {val[-1]:.6f}")

    print(f"\nNote: Surface BC weight for moisture BC is automatically set to 20")
    print(f"      (lower than flux BC's 50) because Dirichlet BCs are more stable")

    return model, losses


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Surface BC Comparison: Flux BC vs Moisture BC")
    print("="*70)
    print("\nThis example demonstrates two types of surface boundary conditions:")
    print("  1. Flux BC (Neumann):    q(z=0, t) = q₀(t)  [prescribed flux]")
    print("  2. Moisture BC (Dirichlet): θ(z=0, t) = θ₀(t)  [prescribed moisture]")
    print("\nThe moisture BC is ideal for SMAP satellite observations!")

    # Run both examples
    print("\n" + "-"*70)
    print("Running FLUX BC example...")
    print("-"*70)
    model_flux, losses_flux = example_flux_bc()

    print("\n" + "-"*70)
    print("Running MOISTURE BC example...")
    print("-"*70)
    model_moisture, losses_moisture = example_moisture_bc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nKey Differences:")
    print("  1. Data: flux uses q₀(t) [m/s], moisture uses θ₀(t) [m³/m³]")
    print("  2. BC type: flux='flux', moisture='moisture'")
    print("  3. Loss weight: flux=50 (higher), moisture=20 (lower)")
    print("  4. Stability: moisture BC typically more stable for PINNs")
    print("\nTo use SMAP data:")
    print("  1. Load SMAP surface soil moisture time series")
    print("  2. Create theta0_data = (times, smap_moisture)")
    print("  3. Set bc_type='moisture'")
    print("  4. Train!")
    print("="*70 + "\n")
