"""
SMAP Recharge Flux PINN - Core modules
"""

from .pinn_models import PressureHeadNet, WaterTableNet, RichardsPINN
from .train_loop import train_pinn_pool_batch_autoweight, finetune_pinn, load_pretrained_model
from .normalization_helper import NormalizationHelper
from .surf_flux import synth_surface_flux
from .gradient_based_sampling import gradient_based_sampling
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
    'gradient_based_sampling',
    'plot_comprehensive_results',
    'plot_training_losses',
]
