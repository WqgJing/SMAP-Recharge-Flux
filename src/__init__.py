"""
SMAP Recharge Flux PINN - Core modules
"""

from .pinn_models import PressureHeadNet, WaterTableNet, RichardsPINN
from .train_loop import train_pinn_pool_batch_autoweight, finetune_pinn, load_pretrained_model
from .normalization_helper import NormalizationHelper
from .surf_flux import synth_surface_flux
from .spike_detection import detect_spike_events
from .boundary_sampling import sample_boundary_points
from .adaptive_boundary_sampling import (
    adaptive_boundary_sampling,
    sample_boundary_points_with_interpolation,
    create_interpolated_spike_samples,
    visualize_sampling_distribution
)
from .visualization import plot_comprehensive_results, plot_training_losses

__all__ = [
    'PressureHeadNet',
    'WaterTableNet',
    'RichardsPINN',
    'train_pinn_pool_batch_autoweight',
    'finetune_pinn',
    'load_pretrained_model',
    'NormalizationHelper',
    'synth_surface_flux',
    'detect_spike_events',
    'sample_boundary_points',
    'adaptive_boundary_sampling',
    'sample_boundary_points_with_interpolation',
    'create_interpolated_spike_samples',
    'visualize_sampling_distribution',
    'plot_comprehensive_results',
    'plot_training_losses',
]
