"""
Unified Dataset Container for PINN Training and Visualization

This module provides a unified interface for loading, storing, and accessing
all data required for PINN training and visualization. The PINNDataset class
reads from YAML configuration and handles all data loading automatically.

Key Features:
- Single source of truth for all data
- YAML-driven: specify what data you have, get what you need
- Flexible: works with any number of observation depths
- Optional data: WTD, flux, multiple depths all handled gracefully
- Clean API: check availability with .has_wtd(), .has_flux(), etc.

Usage:
    dataset = PINNDataset('configs/us_uaf_2019.yaml')
    model = train_pinn(dataset, device='cuda')
    plot_comprehensive_results(model, dataset)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from src.config_loader import load_config
from src.data_loader import load_soil_moisture


class PINNDataset:
    """
    Unified data container for PINN training and visualization.

    Loads all data from YAML config and provides clean access methods.
    Handles variable number of observation depths, optional WTD, etc.

    Attributes:
        config: Loaded configuration object

        # Boundary Condition (always required)
        bc_times: np.ndarray - Time points for BC [seconds]
        bc_values: np.ndarray - BC values (theta or flux)
        bc_type: str - 'dirichlet' (moisture) or 'neumann' (flux)

        # Initial Condition (computed automatically from first observation)
        # Uses unified parabolic profile: h(z) = a*z² + b*z + c
        # Satisfies: h(0)=h_surface, h(-zb)=0, dh/dz|_{z=-zb}=-1

        # Observations (optional - for validation)
        obs_times: np.ndarray - Time array for all observations
        obs_datetime: pd.DatetimeIndex - Datetime array
        obs_depths: List[str] - Available depth names (e.g., ['2cm', '15cm', '30cm'])
        obs_theta: Dict[str, np.ndarray] - {depth_name: theta_array}

        # Water Table Depth (optional - for validation)
        wtd_times: Optional[np.ndarray] - Time points for WTD
        wtd_values: Optional[np.ndarray] - WTD depths [m, positive downward]

        # Soil & Physics Parameters (always required)
        soil_params: dict - van Genuchten-Mualem parameters
        Sy, zr, L, S_max, zb_initial: float - Physics parameters

        # Training Configuration
        h_net_config, zb_net_config: dict - Network architectures
        n_epochs, learning_rate: Training hyperparameters
        ... (all other training config)
    """

    def __init__(self, config_path: str, verbose: bool = True):
        """
        Initialize dataset from YAML configuration.

        Args:
            config_path: Path to YAML config file
            verbose: Print loading information
        """
        self.config_path = config_path
        self.verbose = verbose

        if self.verbose:
            print("=" * 70)
            print("INITIALIZING PINN DATASET")
            print("=" * 70)
            print(f"Config: {config_path}")

        # Load configuration
        self.config = load_config(config_path)

        # Fix data path if running from notebook or different directory
        self._fix_data_path()

        # Load all data from config
        self._load_all_data()

        # Extract training/physics parameters from config
        self._extract_parameters()

        if self.verbose:
            print("\n" + "=" * 70)
            print("DATASET INITIALIZATION COMPLETE")
            print("=" * 70)
            self.print_summary()

    def _fix_data_path(self):
        """
        Fix data path if running from a different directory (e.g., notebooks/).

        If data path is relative and doesn't exist, tries prepending '../' for notebook context.
        """
        data_path = self.config.data_path

        # If path is already absolute or exists, no fix needed
        if os.path.isabs(data_path) or os.path.exists(data_path):
            return

        # Try prepending ../ for notebook context
        parent_path = os.path.join('..', data_path)
        if os.path.exists(parent_path):
            self.config._config['data']['path'] = parent_path
            if self.verbose:
                print(f"Note: Adjusted data path for notebook context: {parent_path}\n")
        # If that doesn't work, leave as-is and let it fail with clear error

    def _load_all_data(self):
        """Load all data specified in YAML configuration."""
        config = self.config

        if self.verbose:
            print("\n" + "=" * 70)
            print("LOADING DATA FROM YAML SPECIFICATION")
            print("=" * 70)

        # Get column mapping (if present)
        column_mapping = config.column_mapping

        if not column_mapping:
            raise ValueError(
                "No column_mapping found in config! "
                "Please specify data columns in YAML under data.column_mapping"
            )

        # Load soil moisture data using column mapping
        soil_data = load_soil_moisture(
            filepath=config.data_path,
            column_mapping=column_mapping,
            interpolate=config.interpolate,
            max_gap_hours=config.max_gap_hours,
            start_date=config.start_date,
            end_date=config.end_date,
            verbose=self.verbose
        )

        # Extract common time array
        self.obs_times = soil_data['times_seconds']
        self.obs_datetime = soil_data['datetime']

        # Extract available depth names
        self.obs_depths = soil_data['depths_names']

        # Extract theta observations for each depth
        self.obs_theta = {}
        for depth_name in self.obs_depths:
            key = f'theta_{depth_name}'
            if key in soil_data:
                self.obs_theta[depth_name] = soil_data[key]

        if self.verbose:
            print(f"\n✓ Loaded observations at {len(self.obs_depths)} depths: {self.obs_depths}")

        # ====================================================================
        # BOUNDARY CONDITION - Surface moisture (first depth = surface)
        # ====================================================================
        surface_depth = self.obs_depths[0]
        theta_surface = self.obs_theta[surface_depth]

        # Remove NaN values from surface BC
        valid_mask = ~pd.isna(theta_surface)
        self.bc_times = np.array(self.obs_times[valid_mask], dtype=np.float64)
        self.bc_values = np.array(theta_surface[valid_mask], dtype=np.float64)
        self.bc_type = 'dirichlet'  # Surface moisture is Dirichlet BC

        if self.verbose:
            print(f"\n✓ Boundary Condition (Dirichlet):")
            print(f"  Surface depth: {surface_depth}")
            print(f"  BC points: {len(self.bc_times)}")
            print(f"  Value range: {self.bc_values.min():.4f} - {self.bc_values.max():.4f} m³/m³")

        # ====================================================================
        # INITIAL CONDITION - Depth profile at t=0
        # ====================================================================
        ic_depths = []
        ic_theta = []

        for depth_name in self.obs_depths:
            theta_series = self.obs_theta[depth_name]
            depth_m = self._depth_name_to_meters(depth_name)

            # Get first valid theta value as IC
            valid_mask = ~pd.isna(theta_series)
            if valid_mask.any():
                if isinstance(theta_series, pd.Series):
                    theta_ic = theta_series[valid_mask].iloc[0]
                else:
                    theta_ic = theta_series[valid_mask][0]

                ic_depths.append(depth_m)
                ic_theta.append(theta_ic)

        # Store initial condition metadata for external access (e.g., notebooks)
        self.ic_type = 'parabolic'
        self.ic_profile = {
            'depths': np.array(ic_depths, dtype=np.float64),
            'theta': np.array(ic_theta, dtype=np.float64),
        }

        if self.verbose:
            print(f"\n✓ Initial Condition: Using parabolic profile")
            print(f"  Surface moisture from first observation: θ₀ = {ic_theta[0]:.4f} m³/m³")
            print(f"  Measurement points: {len(ic_depths)}")
            for depth_m, theta_val in zip(ic_depths, ic_theta):
                depth_cm = depth_m * 100
                print(f"    {depth_cm:6.1f}cm: θ = {theta_val:.4f} m³/m³")

        # ====================================================================
        # WATER TABLE DEPTH (optional)
        # ====================================================================
        wtd_col = column_mapping.get('wtd', None)

        if wtd_col is not None:
            # WTD data exists in config - load it separately
            if self.verbose:
                print(f"\n" + "=" * 70)
                print("LOADING WATER TABLE DEPTH (WTD) DATA")
                print("=" * 70)

            try:
                # Read file (CSV or Excel) with date filtering
                file_ext = config.data_path.lower().split('.')[-1]

                if file_ext == 'csv':
                    # CSV file (e.g., AmeriFlux data)
                    df = pd.read_csv(config.data_path, comment='#')
                else:
                    # Excel file
                    df = pd.read_excel(config.data_path)

                # Get datetime column
                datetime_col_name = column_mapping.get('datetime')
                if column_mapping.get('multi_level_header', False):
                    if isinstance(datetime_col_name, list):
                        # Multi-level header - reload with proper header
                        if file_ext == 'csv':
                            df = pd.read_csv(config.data_path, comment='#', header=[0, 1])
                        else:
                            df = pd.read_excel(config.data_path, header=[0, 1])
                        dt_col = df[tuple(datetime_col_name)]
                    else:
                        dt_col = df[datetime_col_name]
                else:
                    dt_col = df[datetime_col_name]

                # Convert to datetime
                # Handle AmeriFlux timestamp format for CSV files
                if file_ext == 'csv':
                    try:
                        dt_col = pd.to_datetime(dt_col, format='%Y%m%d%H%M')
                    except (ValueError, TypeError):
                        dt_col = pd.to_datetime(dt_col)
                else:
                    dt_col = pd.to_datetime(dt_col)

                # Apply date range filter (same as main data)
                if config.start_date is not None:
                    mask = dt_col >= config.start_date
                    df = df[mask]
                    dt_col = dt_col[mask]
                if config.end_date is not None:
                    mask = dt_col <= config.end_date
                    df = df[mask]
                    dt_col = dt_col[mask]

                # Now extract WTD from filtered dataframe
                if column_mapping.get('multi_level_header', False):
                    if isinstance(wtd_col, tuple) or isinstance(wtd_col, list):
                        wtd_raw = df[tuple(wtd_col)].values
                    else:
                        wtd_raw = df[wtd_col].values
                else:
                    if isinstance(wtd_col, tuple) or isinstance(wtd_col, list):
                        wtd_raw = df[wtd_col[0]].values
                    else:
                        wtd_raw = df[wtd_col].values

                # Convert to numeric
                wtd_raw = pd.to_numeric(wtd_raw, errors='coerce')

                # Replace missing value codes with NaN
                from src.data_loader import replace_missing_codes
                wtd_raw = replace_missing_codes(wtd_raw)

                # Apply sign conversion if specified
                wtd_sign = column_mapping.get('wtd_sign', 1.0)
                wtd_clean = wtd_raw * wtd_sign

                # Remove NaN values and align with obs_times
                # Note: WTD and obs data should have same length after date filtering
                valid_mask = ~pd.isna(wtd_clean)
                if valid_mask.any() and len(wtd_clean) == len(self.obs_times):
                    self.wtd_times = self.obs_times[valid_mask]
                    self.wtd_values = wtd_clean[valid_mask]

                    if self.verbose:
                        print(f"✓ WTD data loaded successfully")
                        print(f"  Column: {wtd_col}")
                        print(f"  Sign conversion: {wtd_sign}")
                        print(f"  Valid points: {len(self.wtd_values)}")
                        print(f"  Range: {self.wtd_values.min():.3f} - {self.wtd_values.max():.3f} m (positive downward)")
                else:
                    self.wtd_times = None
                    self.wtd_values = None
                    if self.verbose:
                        print(f"⚠ WTD column found but no valid data")

            except Exception as e:
                self.wtd_times = None
                self.wtd_values = None
                if self.verbose:
                    print(f"⚠ Failed to load WTD: {e}")
        else:
            self.wtd_times = None
            self.wtd_values = None

            if self.verbose:
                print(f"\n✗ No WTD data specified in config")

        # ==================== Load ET Data (Optional) ====================
        et_col = column_mapping.get('et', None)
        if et_col is not None:
            try:
                if self.verbose:
                    print(f"\n--- Loading ET Data ---")

                # Get ET column from dataframe
                if column_mapping.get('multi_level_header', False):
                    if isinstance(et_col, list):
                        et_raw = df[tuple(et_col)]
                    else:
                        et_raw = df[et_col]
                else:
                    et_raw = df[et_col]

                # Apply unit conversion if specified
                et_conversion = column_mapping.get('et_conversion', 1.0)
                et_clean = et_raw * et_conversion

                # Remove NaN values and align with obs_times
                valid_mask = ~pd.isna(et_clean)
                if valid_mask.any() and len(et_clean) == len(self.obs_times):
                    self.et_times = self.obs_times[valid_mask]
                    self.et_values = et_clean[valid_mask].values

                    if self.verbose:
                        print(f"✓ ET data loaded successfully")
                        print(f"  Column: {et_col}")
                        print(f"  Unit conversion: {et_conversion}")
                        print(f"  Valid points: {len(self.et_values)}")
                        print(f"  Range: {self.et_values.min():.2e} - {self.et_values.max():.2e} [1/s]")
                else:
                    self.et_times = None
                    self.et_values = None
                    if self.verbose:
                        print(f"⚠ ET column found but no valid data")

            except Exception as e:
                self.et_times = None
                self.et_values = None
                if self.verbose:
                    print(f"⚠ Failed to load ET: {e}")
        else:
            self.et_times = None
            self.et_values = None

            if self.verbose:
                print(f"\n✗ No ET data specified in config (will use constant S_max)")

    def _extract_parameters(self):
        """Extract all training and physics parameters from config."""
        config = self.config

        # Soil parameters
        self.soil_params = config.soil_params

        # Physics parameters
        self.Sy = config.Sy
        self.zr = config.zr
        self.L = config.L
        self.S_max = config.S_max
        self.zb_initial = config.zb_initial

        # Network architecture
        self.h_net_config = config.h_net_config
        self.zb_net_config = config.zb_net_config

        # Training hyperparameters
        self.n_epochs = config.n_epochs
        self.learning_rate = config.learning_rate
        self.seed = config.seed
        self.device = config.device

        # Sampling parameters
        self.batch_size = config.batch_size
        self.boundary_ratio = config.boundary_ratio

        # Boundary sampling parameters
        self.batch_size_bc = config.batch_size_bc
        self.interp_ratio = config.interp_ratio
        self.neighbor_ratio = config.neighbor_ratio
        self.baseline_ratio = config.baseline_ratio
        self.gradient_neighbor_expansion = config.gradient_neighbor_expansion
        self.gradient_threshold = config.gradient_threshold
        self.gradient_power = config.gradient_power

        # Optimization parameters
        self.weight_update_freq = config.weight_update_freq
        self.weight_lr = config.weight_lr
        self.use_initial_scales = config.use_initial_scales
        self.use_amp = config.use_amp
        self.use_multi_gpu = config.use_multi_gpu
        self.grad_accumulation_steps = config.grad_accumulation_steps

        # L-BFGS optimizer switching
        self.switch_to_lbfgs_epoch = config.switch_to_lbfgs_epoch
        self.lbfgs_lr = config.lbfgs_lr
        self.lbfgs_max_iter = config.lbfgs_max_iter
        self.lbfgs_history_size = config.lbfgs_history_size
        self.lbfgs_tolerance_grad = config.lbfgs_tolerance_grad
        self.lbfgs_tolerance_change = config.lbfgs_tolerance_change

        # Checkpointing
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_freq = config.checkpoint_freq
        self.keep_last_n_checkpoints = config.keep_last_n_checkpoints

    def _depth_name_to_meters(self, depth_name: str) -> float:
        """Convert depth name like '2cm' or '15cm' to meters."""
        if depth_name.endswith('cm'):
            return float(depth_name[:-2]) / 100.0
        elif depth_name.endswith('m'):
            return float(depth_name[:-1])
        else:
            raise ValueError(f"Unknown depth format: {depth_name}")

    def get_bc_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get boundary condition data as tuple for training.

        Returns:
            (times, values): BC time points and values
        """
        return (self.bc_times, self.bc_values)

    def has_wtd(self) -> bool:
        """Check if water table depth observations are available."""
        return self.wtd_values is not None and len(self.wtd_values) > 0

    def get_et_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get ET data as tuple for training.

        Returns:
            (times, values): ET time points and values [s, 1/s], or None if not available
        """
        if self.et_values is not None and len(self.et_values) > 0:
            return (self.et_times, self.et_values)
        return None

    def has_et(self) -> bool:
        """Check if ET observations are available."""
        return self.et_values is not None and len(self.et_values) > 0

    def has_obs_at_depth(self, depth_name: str) -> bool:
        """Check if observations are available at specific depth."""
        return depth_name in self.obs_theta

    def get_obs_data_for_viz(self) -> Dict[str, Any]:
        """
        Get observation data formatted for visualization functions.

        Returns:
            dict: {'times': array, 'depths': [z1, z2, ...], 'theta': [theta1, theta2, ...]}
        """
        obs_depths_m = []
        obs_theta_list = []

        for depth_name in self.obs_depths:
            depth_m = -self._depth_name_to_meters(depth_name)  # Negative for plotting
            obs_depths_m.append(depth_m)
            obs_theta_list.append(self.obs_theta[depth_name])

        return {
            'times': self.obs_times,
            'depths': obs_depths_m,
            'theta': obs_theta_list
        }

    def get_wtd_data_for_viz(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get WTD data formatted for visualization functions.

        Returns:
            Dictionary {'times': array, 'wtd': array} if WTD available, else None
        """
        if self.has_wtd():
            return {
                'times': self.wtd_times,
                'wtd': self.wtd_values
            }
        else:
            return None

    def print_summary(self):
        """Print summary of loaded dataset."""
        print("\nDataset Summary:")
        print(f"  Config: {self.config_path}")
        print(f"  Data source: {self.config.data_path}")
        print(f"  Date range: {self.config.start_date} to {self.config.end_date}")
        print(f"  Duration: {self.obs_times[-1]/86400:.1f} days")
        print(f"\n  Boundary Condition:")
        print(f"    Type: {self.bc_type}")
        print(f"    Points: {len(self.bc_times)}")
        print(f"    Range: {self.bc_values.min():.4f} - {self.bc_values.max():.4f} m³/m³")
        print(f"\n  Observations:")
        print(f"    Depths: {len(self.obs_depths)} → {self.obs_depths}")
        print(f"    Time points: {len(self.obs_times)}")
        print(f"\n  Initial Condition:")
        print(f"    Type: Parabolic profile (unified)")
        print(f"    Computed from surface observation at t=0")
        print(f"\n  Water Table Depth:")
        if self.has_wtd():
            print(f"    Available: Yes ({len(self.wtd_values)} points)")
        else:
            print(f"    Available: No")
        print(f"\n  Training:")
        print(f"    Epochs: {self.n_epochs}")
        print(f"    Learning rate: {self.learning_rate}")
        print(f"    Device: {self.device}")
