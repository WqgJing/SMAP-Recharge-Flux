"""
Data preprocessing for PINN training.

Loads raw soil moisture data, cleans/interpolates, and saves preprocessed data.
"""

import os
import pickle
import numpy as np
import pandas as pd

from src.data_loader import load_calhoun_soil_moisture


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
        - ic_profile: Initial condition profile
        - config_file: Config path used
        - date_range: Date range string
    """

    # Load soil moisture data
    soil_data = load_calhoun_soil_moisture(
        filepath=config.data_path,
        interpolate=config.interpolate,
        max_gap_hours=config.max_gap_hours,
        start_date=config.start_date,
        end_date=config.end_date,
        verbose=False  # Silent operation
    )

    # Extract data
    datetime_col = soil_data['datetime']
    times_seconds = soil_data['times_seconds']
    theta_2cm = soil_data['theta_2cm']
    theta_15cm = soil_data['theta_15cm']
    theta_30cm = soil_data['theta_30cm']
    theta_40cm = soil_data['theta_40cm']
    theta_60cm = soil_data['theta_60cm']
    theta_80cm = soil_data['theta_80cm']

    # Prepare surface BC data (remove NaN values)
    valid_mask = ~pd.isna(theta_2cm)
    theta0_times = times_seconds[valid_mask]
    theta0_values = theta_2cm[valid_mask]

    theta0_times_np = np.array(theta0_times, dtype=np.float64)
    theta0_values_np = np.array(theta0_values, dtype=np.float64)

    # Prepare initial condition profile (extract initial theta at t=0 for all depths)
    ic_depths = []
    ic_theta = []

    depth_mapping = {
        '2cm': (0.02, theta_2cm),
        '15cm': (0.15, theta_15cm),
        '30cm': (0.30, theta_30cm),
        '40cm': (0.40, theta_40cm),
        '60cm': (0.60, theta_60cm),
        '80cm': (0.80, theta_80cm),
    }

    for depth_name, (depth_m, theta_series) in depth_mapping.items():
        valid_mask = ~pd.isna(theta_series)
        if valid_mask.any():
            if isinstance(theta_series, pd.Series):
                theta_ic = theta_series[valid_mask].iloc[0]
            else:
                theta_ic = theta_series[valid_mask][0]
            ic_depths.append(depth_m)
            ic_theta.append(theta_ic)

    ic_profile = {
        'depths': ic_depths,
        'theta': ic_theta
    }

    # Package all data
    preprocessed_data = {
        'theta0_times': theta0_times_np,
        'theta0_values': theta0_values_np,
        'theta_2cm': theta_2cm,
        'theta_15cm': theta_15cm,
        'theta_30cm': theta_30cm,
        'theta_40cm': theta_40cm,
        'theta_60cm': theta_60cm,
        'theta_80cm': theta_80cm,
        'times_seconds': times_seconds,
        'datetime_col': datetime_col,
        'ic_profile': ic_profile,
        'config_file': config.config_path,
        'date_range': f"{config.start_date} to {config.end_date}",
    }

    # Save to pickle file (overwrite if exists)
    with open(output_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)

    return preprocessed_data
