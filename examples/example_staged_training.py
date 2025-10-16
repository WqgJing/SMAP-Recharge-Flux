"""
Example: Staged Training for PINN to Resolve BC-PDE Gradient Conflicts

This script demonstrates how to use the 3-phase staged training approach
to successfully train a PINN when boundary conditions and PDE loss compete.

Based on the findings in CONVERSATION_SUMMARY.md:
- Phase 1 (0-20%): Train BCs only with minimal PDE weight
- Phase 2 (20-50%): Gradually increase PDE weight while monitoring BCs
- Phase 3 (50-100%): Balanced training with optional adaptive weights

Success rate: ~90% (vs ~30% for simultaneous training)
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.train_loop import train_pinn_pool_batch_autoweight

# ============================================================================
# SETUP: Soil parameters and boundary conditions
# ============================================================================

# Soil parameters (Van Genuchten)
soil_params = {
    'theta_s': 0.43,      # Saturated water content
    'theta_r': 0.078,     # Residual water content
    'alpha': 3.6,         # Van Genuchten α parameter (1/m)
    'n': 1.56,            # Van Genuchten n parameter
    'Ks': 1.04e-5,        # Saturated hydraulic conductivity (m/s)
}

# Specific yield and root zone depth
Sy = 0.2
zr = 0.3  # meters

# Boundary condition data (surface flux q0)
# Example: 15-day period with varying surface flux
t_data = np.linspace(0, 15*86400, 100)  # 15 days in seconds
q0_data_values = 5e-8 * (1 + 0.5 * np.sin(2 * np.pi * t_data / (5*86400)))  # m/s
q0_data = (t_data, q0_data_values)

# Network architecture
h_net_config = {
    'hidden_dims': [64, 64, 64],
    'activation': 'tanh',
}

zb_net_config = {
    'hidden_dims': [32, 32],
    'activation': 'tanh',
}

# ============================================================================
# STAGED TRAINING CONFIGURATION
# ============================================================================

# Training parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epochs = 100000  # For larger alpha, use 150k-200k
learning_rate = 5e-4  # Lower LR for stiff problems
batch_size = 300
batch_size_bc = 200  # Increased BC sampling

# Normalization parameters
L = 5.0  # Characteristic length (m)
S_max = 1e-7  # Maximum sink term (1/s)

# Initial water table depth
zb_initial = 1.5  # meters

print(f"\n{'='*70}")
print("PINN Training with Staged Training (3-Phase Approach)")
print(f"{'='*70}")
print(f"Device: {device}")
print(f"Total epochs: {n_epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Batch sizes: interior={batch_size}, BC={batch_size_bc}")
print(f"\nProblem configuration:")
print(f"  Soil: θs={soil_params['theta_s']}, α={soil_params['alpha']}, n={soil_params['n']}")
print(f"  Domain: L={L} m, zb_initial={zb_initial} m")
print(f"  Time: {t_data[-1]/86400:.1f} days")
print(f"{'='*70}\n")

# ============================================================================
# EXAMPLE 1: Default Staged Training (Recommended)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Default Staged Training")
print("="*70)

model, losses, comps, sample_losses, sample_comps, sample_epochs = \
    train_pinn_pool_batch_autoweight(
        soil_params=soil_params,
        q0_data=q0_data,
        Sy=Sy,
        zr=zr,
        h_net_config=h_net_config,
        zb_net_config=zb_net_config,
        L=L,
        S_max=S_max,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        zb_initial=zb_initial,
        batch_size=batch_size,
        batch_size_bc=batch_size_bc,
        device=device,
        # Enable staged training with default settings
        use_staged_training=True,
        # Phase boundaries (default: 20% and 50%)
        staged_phase1_end=0.20,
        staged_phase2_end=0.50,
        # Enable adaptive weights in Phase 3
        staged_enable_adaptive_phase3=True,
        # Enable BC monitoring to pause PDE increase if BCs degrade
        staged_bc_monitoring=True,
        # Adaptive weight settings (for Phase 3)
        weight_update_freq=500,
        weight_lr=0.1,
        # Checkpointing
        checkpoint_freq=10000,
        checkpoint_dir='checkpoints_staged',
    )

print("\n" + "="*70)
print("Training completed successfully!")
print(f"Final losses:")
print(f"  PDE: {sample_losses[-1][sample_epochs[-1]]['pde']:.3e}")
print(f"  Surface BC: {sample_losses[-1][sample_epochs[-1]]['surf']:.3e}")
print(f"  WT head BC: {sample_losses[-1][sample_epochs[-1]]['wt_head']:.3e}")
print(f"  WT kinematic BC: {sample_losses[-1][sample_epochs[-1]]['wt_kin']:.3e}")
print("="*70 + "\n")

# ============================================================================
# EXAMPLE 2: Custom Staged Training for Larger Alpha
# ============================================================================

# For larger alpha (α > 5.0), use more aggressive BC weights and longer Phase 2

print("\n" + "="*70)
print("EXAMPLE 2: Custom Staged Training for Larger Alpha")
print("="*70)
print("For stiff problems (large α), use:")
print("  - Stronger BC weights (surf=200+, wt=100+)")
print("  - Longer Phase 1 and 2 (30% and 60%)")
print("  - More epochs (150k-200k)")
print("  - Smaller learning rate (5e-4)")
print("="*70 + "\n")

# This example shows the parameters but doesn't run (to save time)
# Uncomment to run for larger alpha problems:

"""
from src.training_utils import StagedTrainingScheduler

# Create custom scheduler for stiff problems
custom_scheduler = StagedTrainingScheduler(
    n_epochs=150000,
    # Longer phases for stiff problems
    phase1_end=0.30,  # 30% for BC convergence
    phase2_end=0.60,  # 30% for gradual PDE introduction
    # Phase 1: Very strong BC emphasis
    phase1_pde=0.00001,
    phase1_surf=200.0,
    phase1_wt_head=100.0,
    phase1_wt_kin=100.0,
    phase1_ic_h=100.0,
    phase1_ic_zb=100.0,
    # Phase 2: Slower PDE ramp-up
    phase2_pde_start=0.00001,
    phase2_pde_end=0.005,  # Lower than default
    phase2_surf=200.0,
    phase2_wt_head=100.0,
    phase2_wt_kin=100.0,
    phase2_ic_h=50.0,
    phase2_ic_zb=50.0,
    # Phase 3: Moderate balance
    phase3_pde=0.05,  # Lower than default
    phase3_surf=200.0,
    phase3_wt_head=100.0,
    phase3_wt_kin=100.0,
    phase3_ic_h=20.0,
    phase3_ic_zb=20.0,
    # Stricter BC monitoring
    bc_loss_threshold_multiplier=3.0,  # More sensitive
    enable_bc_monitoring=True,
    enable_adaptive_phase3=True,
)

# Pass custom scheduler via WeightManager
from src.training_utils import WeightManager

weight_manager = WeightManager(
    use_initial_scales=True,
    weight_lr=0.05,  # Lower LR for adaptive weights
    staged_scheduler=custom_scheduler,
)

# Then pass weight_manager to training function
# (requires modification to accept pre-created weight_manager)
"""

# ============================================================================
# EXAMPLE 3: Disable BC Monitoring (if BCs are very stable)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Staged Training Without BC Monitoring")
print("="*70)
print("If your BCs converge very stably, you can disable monitoring:")
print("="*70 + "\n")

# Uncomment to run:
"""
model3, losses3, comps3, sl3, sc3, se3 = train_pinn_pool_batch_autoweight(
    soil_params=soil_params,
    q0_data=q0_data,
    Sy=Sy,
    zr=zr,
    h_net_config=h_net_config,
    zb_net_config=zb_net_config,
    L=L,
    S_max=S_max,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    zb_initial=zb_initial,
    batch_size=batch_size,
    batch_size_bc=batch_size_bc,
    device=device,
    use_staged_training=True,
    staged_bc_monitoring=False,  # Disable monitoring
    checkpoint_freq=10000,
)
"""

# ============================================================================
# VISUALIZE RESULTS
# ============================================================================

print("\n" + "="*70)
print("Visualization")
print("="*70)

# Plot training history
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss components over time
    ax = axes[0, 0]
    epochs = list(range(len(losses)))
    ax.semilogy(epochs, [comps[e]['pde'] for e in epochs], label='PDE', alpha=0.7)
    ax.semilogy(epochs, [comps[e]['surf'] for e in epochs], label='Surface BC', alpha=0.7)
    ax.semilogy(epochs, [comps[e]['wt_head'] for e in epochs], label='WT Head BC', alpha=0.7)
    ax.semilogy(epochs, [comps[e]['wt_kin'] for e in epochs], label='WT Kinematic BC', alpha=0.7)
    ax.axvline(n_epochs * 0.20, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
    ax.axvline(n_epochs * 0.50, color='green', linestyle='--', alpha=0.5, label='Phase 2→3')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Components (Staged Training)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sample losses (full dataset)
    ax = axes[0, 1]
    if len(sample_epochs) > 0:
        se_list = list(sample_epochs)
        ax.semilogy(se_list, [sample_comps[e]['pde'] for e in se_list], 'o-', label='PDE', alpha=0.7)
        ax.semilogy(se_list, [sample_comps[e]['surf'] for e in se_list], 'o-', label='Surface BC', alpha=0.7)
        ax.axvline(n_epochs * 0.20, color='red', linestyle='--', alpha=0.5)
        ax.axvline(n_epochs * 0.50, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Sample Loss')
        ax.set_title('Full Dataset Loss (Every 500 epochs)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Weight evolution
    ax = axes[1, 0]
    from src.training_utils import WeightManager
    # Note: In actual implementation, weight history should be accessible from weight_manager
    # This is a placeholder
    ax.text(0.5, 0.5, 'Weight evolution\n(access weight_manager.weight_history)',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Weight Evolution (PDE, BCs, ICs)')

    # Phase diagram
    ax = axes[1, 1]
    phase_text = """
Staged Training Phases:

Phase 1 (0-20%): BC-Focused
  • PDE weight: 0.0001 (minimal)
  • BC weights: 50-100 (strong)
  • Goal: BCs converge to ~1e-4

Phase 2 (20-50%): Gradual PDE
  • PDE: 0.0001 → 0.01 (ramp up)
  • BC monitoring: ACTIVE
  • Auto-pause if BCs degrade

Phase 3 (50-100%): Balanced
  • PDE weight: 0.1
  • Adaptive weights: ENABLED
  • Final refinement
    """
    ax.text(0.1, 0.9, phase_text, ha='left', va='top',
            transform=ax.transAxes, fontsize=10, family='monospace')
    ax.axis('off')
    ax.set_title('Training Strategy')

    plt.tight_layout()
    plt.savefig('staged_training_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: staged_training_results.png")
    plt.show()

except ImportError:
    print("Matplotlib not available. Skipping visualization.")
    print("Install with: pip install matplotlib")

print("\n" + "="*70)
print("Example completed!")
print("="*70)
