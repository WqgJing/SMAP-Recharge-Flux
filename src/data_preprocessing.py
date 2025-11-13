"""
Data preprocessing for PINN training.

Loads raw soil moisture data, cleans/interpolates, and saves preprocessed data.
"""

import os
import pickle
import numpy as np
import pandas as pd

from src.data_loader import load_calhoun_soil_moisture, load_soil_moisture


def preprocess_soil_data(config, output_path='data/preprocessed_soil_data.pkl'):
    """
    Preprocess soil moisture data for PINN training.

    Args:
        config: Config object from config_loader
        output_path: Path to save preprocessed data pickle file

    Saves:
        preprocessed_soil_data.pkl containing:
        - theta0_times: Surface BC time points
        - theta0_values: Surface BC moisture values
        - theta_2cm, theta_15cm, ..., theta_80cm: Full depth data
        - times_seconds: Time array
        - datetime_col: Datetime array
        - ic_profile: (Legacy - for reference only, not used by model)
        - config_file: Config path used
        - date_range: Date range string
    """

    # Load soil moisture data using generic loader with column mapping
    column_mapping = config.column_mapping

    if not column_mapping:
        # Fallback to old Calhoun-specific loader if no column mapping provided
        soil_data = load_calhoun_soil_moisture(
            filepath=config.data_path,
            interpolate=config.interpolate,
            max_gap_hours=config.max_gap_hours,
            start_date=config.start_date,
            end_date=config.end_date,
            verbose=False  # Silent operation
        )
    else:
        # Use new generic loader with column mapping
        soil_data = load_soil_moisture(
            filepath=config.data_path,
            column_mapping=column_mapping,
            interpolate=config.interpolate,
            max_gap_hours=config.max_gap_hours,
            start_date=config.start_date,
            end_date=config.end_date,
            verbose=False  # Silent operation
        )

    # Extract data
    datetime_col = soil_data['datetime']
    times_seconds = soil_data['times_seconds']

    # Get depth names from the data (supports variable number of depths)
    depths_names = soil_data.get('depths_names', ['2cm', '15cm', '30cm', '40cm', '60cm', '80cm'])

    # Extract theta values for each depth
    theta_dict = {}
    for depth_name in depths_names:
        key = f'theta_{depth_name}'
        if key in soil_data:
            theta_dict[depth_name] = soil_data[key]
        else:
            raise KeyError(f"Missing theta data for depth: {depth_name}")

    # For backward compatibility, also create individual variables
    theta_2cm = theta_dict.get('2cm')
    theta_15cm = theta_dict.get('15cm')
    theta_30cm = theta_dict.get('30cm')
    theta_40cm = theta_dict.get('40cm')
    theta_60cm = theta_dict.get('60cm')
    theta_80cm = theta_dict.get('80cm')

    # Prepare surface BC data (remove NaN values)
    valid_mask = ~pd.isna(theta_2cm)
    theta0_times = times_seconds[valid_mask]
    theta0_values = theta_2cm[valid_mask]

    theta0_times_np = np.array(theta0_times, dtype=np.float64)
    theta0_values_np = np.array(theta0_values, dtype=np.float64)

    # Prepare initial condition profile (extract initial theta at t=0 for all depths)
    ic_depths = []
    ic_theta = []

    # Convert depth names to meters (flexible mapping)
    def depth_name_to_meters(depth_name):
        """Convert depth name like '2cm' or '15cm' to meters."""
        if depth_name.endswith('cm'):
            return float(depth_name[:-2]) / 100.0
        elif depth_name.endswith('m'):
            return float(depth_name[:-1])
        else:
            raise ValueError(f"Unknown depth format: {depth_name}")

    # Build depth mapping dynamically
    depth_mapping = {}
    for depth_name in depths_names:
        if depth_name in theta_dict:
            depth_m = depth_name_to_meters(depth_name)
            depth_mapping[depth_name] = (depth_m, theta_dict[depth_name])

    for depth_name, (depth_m, theta_series) in depth_mapping.items():
        if theta_series is not None:
            valid_mask = ~pd.isna(theta_series)
            if valid_mask.any():
                if isinstance(theta_series, pd.Series):
                    theta_ic = theta_series[valid_mask].iloc[0]
                else:
                    theta_ic = theta_series[valid_mask][0]
                ic_depths.append(depth_m)
                ic_theta.append(theta_ic)

    # Note: IC profile is for reference only
    # The model now uses automatic parabolic IC from surface obs at t=0
    ic_profile = {
        'depths': ic_depths,
        'theta': ic_theta
    }

    # Package all data
    preprocessed_data = {
        'theta0_times': theta0_times_np,
        'theta0_values': theta0_values_np,
        'times_seconds': times_seconds,
        'datetime_col': datetime_col,
        'ic_profile': ic_profile,
        'config_file': config.config_path,
        'date_range': f"{config.start_date} to {config.end_date}",
    }

    # Add theta data for each depth dynamically
    for depth_name in depths_names:
        preprocessed_data[f'theta_{depth_name}'] = theta_dict[depth_name]

    # For backward compatibility with code that expects specific depth names
    if theta_2cm is not None:
        preprocessed_data['theta_2cm'] = theta_2cm
    if theta_15cm is not None:
        preprocessed_data['theta_15cm'] = theta_15cm
    if theta_30cm is not None:
        preprocessed_data['theta_30cm'] = theta_30cm
    if theta_40cm is not None:
        preprocessed_data['theta_40cm'] = theta_40cm
    if theta_60cm is not None:
        preprocessed_data['theta_60cm'] = theta_60cm
    if theta_80cm is not None:
        preprocessed_data['theta_80cm'] = theta_80cm

    # Save to pickle file (overwrite if exists)
    with open(output_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)

    return preprocessed_data
