"""
Configuration loader for PINN training.

Loads hyperparameters from YAML files to centralize configuration management.
"""

import yaml
import os
from typing import Dict, Any, Optional
import torch


class PINNConfig:
    """Configuration class for PINN training."""

    def __init__(self, config_path: str):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self.config_path = config_path
        self._validate_config()

    def _validate_config(self):
        """Validate that all required sections are present."""
        required_sections = [
            'data', 'soil', 'training', 'network',
            'sampling', 'boundary_sampling', 'optimization'
        ]
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required section '{section}' in config")

    # ========== Data Configuration ==========
    @property
    def data_path(self) -> str:
        return self._config['data']['path']

    @property
    def start_date(self) -> str:
        return self._config['data']['start_date']

    @property
    def end_date(self) -> str:
        return self._config['data']['end_date']

    @property
    def interpolate(self) -> bool:
        return self._config['data'].get('interpolate', True)

    @property
    def max_gap_hours(self) -> int:
        return self._config['data'].get('max_gap_hours', 6)

    @property
    def column_mapping(self) -> Dict[str, Any]:
        """
        Get column mapping configuration for data loading.
        Converts YAML lists to tuples where needed for pandas column indexing.
        """
        mapping = self._config['data'].get('column_mapping', {})

        # Convert list to tuple for datetime column (if it's a list)
        if 'datetime' in mapping and isinstance(mapping['datetime'], list):
            mapping = mapping.copy()  # Don't modify original
            mapping['datetime'] = tuple(mapping['datetime'])

        # Convert lists to tuples for depth columns
        if 'depths' in mapping:
            depths = {}
            for depth_name, col_name in mapping['depths'].items():
                if isinstance(col_name, list):
                    depths[depth_name] = tuple(col_name)
                else:
                    depths[depth_name] = col_name
            mapping['depths'] = depths

        return mapping

    # ========== Soil Parameters ==========
    @property
    def soil_params(self) -> Dict[str, float]:
        return self._config['soil']

    # ========== Training Configuration ==========
    @property
    def n_epochs(self) -> int:
        return int(self._config['training']['n_epochs'])

    @property
    def learning_rate(self) -> float:
        return float(self._config['training']['learning_rate'])

    @property
    def device(self) -> str:
        device_str = self._config['training'].get('device', 'auto')
        if device_str == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_str

    @property
    def seed(self) -> int:
        return self._config['training'].get('seed', 42)

    # ========== Network Configuration ==========
    @property
    def h_net_config(self) -> Dict[str, int]:
        return self._config['network']['h_net']

    @property
    def zb_net_config(self) -> Dict[str, int]:
        return self._config['network']['zb_net']

    # ========== Sampling Configuration ==========
    @property
    def cache_size(self) -> int:
        return int(self._config['sampling']['cache_size'])

    @property
    def batch_size(self) -> int:
        return int(self._config['sampling']['batch_size'])

    @property
    def resample_freq(self) -> int:
        return int(self._config['sampling']['resample_freq'])

    @property
    def boundary_ratio(self) -> float:
        return float(self._config['sampling']['boundary_ratio'])

    @property
    def high_residual_ratio(self) -> float:
        return float(self._config['sampling']['high_residual_ratio'])

    @property
    def temperature(self) -> float:
        return float(self._config['sampling']['temperature'])

    # ========== Boundary Sampling Configuration ==========
    @property
    def batch_size_bc(self) -> int:
        return int(self._config['boundary_sampling']['batch_size_bc'])

    @property
    def interp_ratio(self) -> float:
        return float(self._config['boundary_sampling']['interp_ratio'])

    @property
    def neighbor_ratio(self) -> float:
        return float(self._config['boundary_sampling']['neighbor_ratio'])

    @property
    def baseline_ratio(self) -> float:
        return float(self._config['boundary_sampling']['baseline_ratio'])

    @property
    def gradient_neighbor_expansion(self) -> int:
        return int(self._config['boundary_sampling']['neighbor_expansion'])

    @property
    def gradient_threshold(self) -> float:
        return float(self._config['boundary_sampling']['gradient_threshold'])

    @property
    def gradient_power(self) -> float:
        return float(self._config['boundary_sampling']['power'])

    # ========== Optimization Configuration ==========
    @property
    def weight_update_freq(self) -> float:
        return float(self._config['optimization']['weight_update_freq'])

    @property
    def weight_lr(self) -> float:
        return float(self._config['optimization']['weight_lr'])

    @property
    def use_initial_scales(self) -> bool:
        return self._config['optimization'].get('use_initial_scales', True)

    @property
    def use_amp(self) -> bool:
        return self._config['optimization'].get('use_amp', False)

    @property
    def use_multi_gpu(self) -> bool:
        return self._config['optimization'].get('use_multi_gpu', True)

    @property
    def grad_accumulation_steps(self) -> int:
        return int(self._config['optimization'].get('grad_accumulation_steps', 1))

    # ========== Checkpointing Configuration ==========
    @property
    def checkpoint_dir(self) -> str:
        return self._config.get('checkpointing', {}).get('dir', 'checkpoints')

    @property
    def checkpoint_freq(self) -> Optional[int]:
        freq = self._config.get('checkpointing', {}).get('freq', None)
        return int(freq) if freq is not None else None

    @property
    def keep_last_n_checkpoints(self) -> Optional[int]:
        keep_n = self._config.get('checkpointing', {}).get('keep_last_n', 3)
        return int(keep_n) if keep_n is not None else None

    # ========== Physics Configuration ==========
    @property
    def Sy(self) -> float:
        return float(self._config.get('physics', {}).get('Sy', 0.3))

    @property
    def zr(self) -> float:
        return float(self._config.get('physics', {}).get('zr', 0.5))

    @property
    def L(self) -> float:
        return float(self._config.get('physics', {}).get('L', 4.0))

    @property
    def S_max(self) -> float:
        return float(self._config.get('physics', {}).get('S_max', 1e-7))

    @property
    def zb_initial(self) -> float:
        return float(self._config.get('physics', {}).get('zb_initial', 6.1))

    @property
    def ic_type(self) -> str:
        return self._config.get('physics', {}).get('ic_type', 'hydrostatic')

    # ========== Output Configuration ==========
    @property
    def output_dir(self) -> str:
        return self._config.get('output', {}).get('dir', 'outputs')

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"Config file: {self.config_path}")
        print(f"\nData:")
        print(f"  Path: {self.data_path}")
        print(f"  Date range: {self.start_date} to {self.end_date}")
        print(f"\nTraining:")
        print(f"  Epochs: {self.n_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Device: {self.device}")
        print(f"  Seed: {self.seed}")
        print(f"\nNetwork:")
        print(f"  h_net: {self.h_net_config}")
        print(f"  zb_net: {self.zb_net_config}")
        print(f"\nCache Pool & Batch Sampling:")
        print(f"  Cache size: {self.cache_size}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Resample freq: {self.resample_freq}")
        print(f"  Boundary ratio: {self.boundary_ratio}")
        print(f"  High residual ratio: {self.high_residual_ratio}")
        print(f"  Temperature: {self.temperature}")
        print(f"\nBoundary Sampling (Gradient-based):")
        print(f"  Batch size BC: {self.batch_size_bc}")
        print(f"  Interp ratio: {self.interp_ratio}")
        print(f"  Neighbor ratio: {self.neighbor_ratio}")
        print(f"  Baseline ratio: {self.baseline_ratio}")
        print(f"  Gradient threshold: {self.gradient_threshold}")
        print(f"  Gradient power: {self.gradient_power}")
        print(f"\nOptimization:")
        print(f"  Weight update freq: {self.weight_update_freq}")
        print(f"  Use AMP: {self.use_amp}")
        print(f"  Use multi-GPU: {self.use_multi_gpu}")
        print(f"\nPhysics:")
        print(f"  Sy: {self.Sy}, zr: {self.zr}")
        print(f"  L: {self.L}, S_max: {self.S_max}")
        print(f"  zb_initial: {self.zb_initial}")
        print(f"  IC type: {self.ic_type}")
        print(f"\nCheckpointing:")
        print(f"  Directory: {self.checkpoint_dir}")
        print(f"  Frequency: {self.checkpoint_freq}")
        print(f"\nOutput:")
        print(f"  Directory: {self.output_dir}")
        print("=" * 70)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (useful for logging)."""
        return self._config.copy()


def load_config(config_path: str) -> PINNConfig:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        PINNConfig object

    Example:
        >>> config = load_config('configs/baseline.yaml')
        >>> print(config.cache_size)
        80000
    """
    return PINNConfig(config_path)
