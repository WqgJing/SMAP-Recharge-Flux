"""
Data loading and preprocessing utilities for soil moisture data

This module provides functions to load, clean, and interpolate soil moisture
data from various sources, particularly for use as boundary conditions in PINNs.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from datetime import datetime


def replace_missing_codes(data, missing_codes=None):
    """
    Replace common missing value codes with NaN

    Args:
        data: numpy array with potential missing value codes
        missing_codes: list of missing value codes (default: [-9999, -9999.0, -6999, -6999.0])

    Returns:
        numpy array with missing codes replaced by NaN
    """
    if missing_codes is None:
        missing_codes = [-9999, -9999.0, -6999, -6999.0]

    data_copy = data.copy()
    for code in missing_codes:
        data_copy[data_copy == code] = np.nan
    return data_copy


def interpolate_missing_data(times, theta_raw, method='linear', max_gap_hours=6):
    """
    Interpolate missing (NaN) values in soil moisture data

    Args:
        times: time array in seconds
        theta_raw: soil moisture array with potential NaNs
        method: 'linear' or 'cubic' interpolation method
        max_gap_hours: maximum gap to interpolate (hours), longer gaps left as NaN

    Returns:
        tuple: (theta_filled, n_filled)
            - theta_filled: interpolated data
            - n_filled: number of values filled
    """
    theta_filled = theta_raw.copy()

    # Find valid (non-NaN) indices
    valid_mask = ~pd.isna(theta_raw)

    if valid_mask.sum() < 2:
        print(f"    WARNING: Too few valid points ({valid_mask.sum()}), skipping interpolation")
        return theta_filled, 0

    # Get valid times and values
    times_valid = times[valid_mask]
    theta_valid = theta_raw[valid_mask]

    # Create interpolation function
    if method == 'cubic' and len(times_valid) >= 4:
        interp_func = interpolate.interp1d(times_valid, theta_valid, kind='cubic',
                                           bounds_error=False, fill_value=np.nan)
    else:
        interp_func = interpolate.interp1d(times_valid, theta_valid, kind='linear',
                                           bounds_error=False, fill_value=np.nan)

    # Interpolate all missing points
    theta_interp = interp_func(times)

    # Only fill gaps smaller than max_gap_hours
    max_gap_seconds = max_gap_hours * 3600
    nan_indices = np.where(pd.isna(theta_raw))[0]
    n_filled = 0

    for idx in nan_indices:
        # Find nearest valid points before and after
        valid_before = valid_mask[:idx]
        valid_after = valid_mask[idx+1:]

        if valid_before.any() and valid_after.any():
            idx_before = np.where(valid_before)[0][-1]
            idx_after = idx + 1 + np.where(valid_after)[0][0]

            gap_time = times[idx_after] - times[idx_before]

            # Only fill if gap is small enough
            if gap_time <= max_gap_seconds:
                theta_filled[idx] = theta_interp[idx]
                n_filled += 1

    return theta_filled, n_filled


def load_calhoun_soil_moisture(
    filepath,
    interpolate=True,
    max_gap_hours=6,
    start_date=None,
    end_date=None,
    start_day=None,
    duration_days=None,
    verbose=True
):
    """
    Load and clean soil moisture data from Calhoun Experimental Forest Excel file

    Args:
        filepath: path to Excel file
        interpolate: whether to interpolate missing values (default: True)
        max_gap_hours: maximum gap to interpolate in hours (default: 6)
        start_date: starting date for subset (datetime or str 'YYYY-MM-DD', default: None = use all)
        end_date: ending date for subset (datetime or str 'YYYY-MM-DD', default: None = use all)
        start_day: starting day offset from first data point (int, default: None)
        duration_days: duration in days from start_day (int, default: None)
        verbose: print progress and statistics (default: True)

    Time Period Selection:
        Option 1 - Use start_date and end_date:
            load_calhoun_soil_moisture(..., start_date='2017-03-01', end_date='2017-05-31')

        Option 2 - Use start_day and duration_days:
            load_calhoun_soil_moisture(..., start_day=0, duration_days=30)  # First 30 days
            load_calhoun_soil_moisture(..., start_day=60, duration_days=90) # Days 60-150

        Option 3 - Use all data (default):
            load_calhoun_soil_moisture(...)

    Returns:
        dict with keys:
            - 'datetime': pandas datetime index
            - 'times_seconds': time in seconds since start
            - 'theta_2cm': soil moisture at 2cm depth
            - 'theta_15cm': soil moisture at 15cm depth
            - 'theta_30cm': soil moisture at 30cm depth
            - 'theta_40cm': soil moisture at 40cm depth
            - 'theta_60cm': soil moisture at 60cm depth
            - 'theta_80cm': soil moisture at 80cm depth
            - 'depths_names': list of depth names
            - 'interpolation_stats': dict of interpolation statistics
    """

    if verbose:
        print("="*70)
        print("LOADING CALHOUN SOIL MOISTURE DATA")
        print("="*70)

    # Load Excel file with multi-level headers
    df = pd.read_excel(filepath, header=[0,1])

    # Extract datetime column
    datetime_col = df[('Unnamed: 0_level_0', 'Date Time')]

    # Extract soil moisture columns at different depths
    theta_2cm_raw = df[('CR1000_2589', '2cm theta')].values
    theta_15cm_raw = df[('CR1000_2589', '15cm theta')].values
    theta_40cm_raw = df[('CR1000_2589', '40cm theta')].values
    theta_30cm_raw = df[('CR1000_2588', '30cm_theta')].values
    theta_60cm_raw = df[('CR1000_2588', '60cm_theta')].values
    theta_80cm_raw = df[('CR1000_2588', '80cm_theta')].values

    if verbose:
        print(f"\nData loaded:")
        print(f"  Total records: {len(df)}")
        print(f"  Time range: {datetime_col.iloc[2]} to {datetime_col.iloc[-1]}")
        print(f"  Duration: ~{(datetime_col.iloc[-1] - datetime_col.iloc[2]).days} days")

        # Calculate temporal resolution
        datetime_temp = pd.to_datetime(datetime_col.iloc[2:])
        if len(datetime_temp) > 1:
            time_diff = (datetime_temp.iloc[1] - datetime_temp.iloc[0]).total_seconds()
            if time_diff < 60:
                print(f"  Resolution: {time_diff:.0f} seconds")
            elif time_diff < 3600:
                print(f"  Resolution: {time_diff/60:.0f} minutes")
            else:
                print(f"  Resolution: {time_diff/3600:.1f} hours")

    # Remove header rows
    datetime_col = datetime_col.iloc[2:].reset_index(drop=True)
    theta_2cm_raw = theta_2cm_raw[2:]
    theta_15cm_raw = theta_15cm_raw[2:]
    theta_40cm_raw = theta_40cm_raw[2:]
    theta_30cm_raw = theta_30cm_raw[2:]
    theta_60cm_raw = theta_60cm_raw[2:]
    theta_80cm_raw = theta_80cm_raw[2:]

    # Replace missing value codes with NaN
    if verbose:
        print("\n" + "="*70)
        print("REPLACING MISSING VALUE CODES (-9999) WITH NaN")
        print("="*70)

    theta_2cm_raw = replace_missing_codes(theta_2cm_raw)
    theta_15cm_raw = replace_missing_codes(theta_15cm_raw)
    theta_40cm_raw = replace_missing_codes(theta_40cm_raw)
    theta_30cm_raw = replace_missing_codes(theta_30cm_raw)
    theta_60cm_raw = replace_missing_codes(theta_60cm_raw)
    theta_80cm_raw = replace_missing_codes(theta_80cm_raw)

    if verbose:
        print("Missing value codes replaced with NaN")

    # Convert datetime to pandas datetime and seconds since start
    datetime_col = pd.to_datetime(datetime_col)
    t_start = datetime_col.iloc[0]
    times_seconds = (datetime_col - t_start).dt.total_seconds().values

    # ============================================================================
    # TIME PERIOD SELECTION
    # ============================================================================

    # Validate that only one selection method is used
    if (start_date is not None or end_date is not None) and (start_day is not None or duration_days is not None):
        raise ValueError("Cannot use both date-based (start_date/end_date) and day-based (start_day/duration_days) selection. Choose one method.")

    # Apply time period selection
    time_mask = np.ones(len(datetime_col), dtype=bool)  # Default: select all

    if start_date is not None or end_date is not None:
        # Date-based selection
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            time_mask &= (datetime_col >= start_date)

        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            time_mask &= (datetime_col <= end_date)

        if verbose:
            print(f"\nTime period selection (date-based):")
            print(f"  Start date: {start_date if start_date else 'first available'}")
            print(f"  End date:   {end_date if end_date else 'last available'}")

    elif start_day is not None or duration_days is not None:
        # Day-based selection (relative to first data point)
        if start_day is None:
            start_day = 0

        start_seconds = start_day * 86400

        if duration_days is not None:
            end_seconds = start_seconds + duration_days * 86400
            time_mask = (times_seconds >= start_seconds) & (times_seconds <= end_seconds)

            if verbose:
                print(f"\nTime period selection (day-based):")
                print(f"  Start day:  {start_day} (offset from first data point)")
                print(f"  Duration:   {duration_days} days")
                print(f"  End day:    {start_day + duration_days}")
        else:
            time_mask = times_seconds >= start_seconds

            if verbose:
                print(f"\nTime period selection (day-based):")
                print(f"  Start day:  {start_day} (offset from first data point)")
                print(f"  Duration:   all remaining data")

    # Apply time mask to all data
    if not time_mask.all():
        datetime_col = datetime_col[time_mask].reset_index(drop=True)
        times_seconds = times_seconds[time_mask]
        theta_2cm_raw = theta_2cm_raw[time_mask]
        theta_15cm_raw = theta_15cm_raw[time_mask]
        theta_30cm_raw = theta_30cm_raw[time_mask]
        theta_40cm_raw = theta_40cm_raw[time_mask]
        theta_60cm_raw = theta_60cm_raw[time_mask]
        theta_80cm_raw = theta_80cm_raw[time_mask]

        # Recalculate times_seconds from new start
        t_start = datetime_col.iloc[0]
        times_seconds = (datetime_col - datetime_col.iloc[0]).dt.total_seconds().values

        if verbose:
            print(f"  Selected:   {len(datetime_col)} points")
            print(f"  Date range: {datetime_col.iloc[0]} to {datetime_col.iloc[-1]}")
            print(f"  Duration:   {times_seconds[-1]/86400:.1f} days")

    # Print raw data quality
    depths_names = ['2cm', '15cm', '30cm', '40cm', '60cm', '80cm']
    raw_data = [theta_2cm_raw, theta_15cm_raw, theta_30cm_raw, theta_40cm_raw, theta_60cm_raw, theta_80cm_raw]

    if verbose:
        print(f"\nRaw data quality (after replacing -9999 with NaN):")
        print("-"*70)
        for depth, theta_raw in zip(depths_names, raw_data):
            n_total = len(theta_raw)
            n_nan = pd.isna(theta_raw).sum()
            n_valid = n_total - n_nan
            pct_nan = (n_nan / n_total) * 100
            if n_valid > 0:
                print(f"{depth:6s}: {n_valid:5d} valid, {n_nan:5d} NaN ({pct_nan:5.2f}%), "
                      f"range=[{np.nanmin(theta_raw):.4f}, {np.nanmax(theta_raw):.4f}]")
            else:
                print(f"{depth:6s}: {n_valid:5d} valid, {n_nan:5d} NaN ({pct_nan:5.2f}%) - ALL MISSING!")

    # Interpolate missing values if requested
    interpolation_stats = {}

    if interpolate:
        if verbose:
            print("\n" + "="*70)
            print("DATA CLEANING: INTERPOLATING MISSING VALUES")
            print("="*70)
            print(f"\nInterpolating with max gap = {max_gap_hours} hours:")
            print("-"*70)

        theta_2cm, n_filled_2cm = interpolate_missing_data(times_seconds, theta_2cm_raw, max_gap_hours=max_gap_hours)
        theta_15cm, n_filled_15cm = interpolate_missing_data(times_seconds, theta_15cm_raw, max_gap_hours=max_gap_hours)
        theta_30cm, n_filled_30cm = interpolate_missing_data(times_seconds, theta_30cm_raw, max_gap_hours=max_gap_hours)
        theta_40cm, n_filled_40cm = interpolate_missing_data(times_seconds, theta_40cm_raw, max_gap_hours=max_gap_hours)
        theta_60cm, n_filled_60cm = interpolate_missing_data(times_seconds, theta_60cm_raw, max_gap_hours=max_gap_hours)
        theta_80cm, n_filled_80cm = interpolate_missing_data(times_seconds, theta_80cm_raw, max_gap_hours=max_gap_hours)

        filled_counts = [n_filled_2cm, n_filled_15cm, n_filled_30cm, n_filled_40cm, n_filled_60cm, n_filled_80cm]
        clean_data = [theta_2cm, theta_15cm, theta_30cm, theta_40cm, theta_60cm, theta_80cm]

        if verbose:
            for depth, n_filled in zip(depths_names, filled_counts):
                print(f"{depth:6s}: {n_filled:5d} values interpolated")

        # Store interpolation statistics
        for depth, n_filled, theta_clean, theta_raw in zip(depths_names, filled_counts, clean_data, raw_data):
            n_total = len(theta_clean)
            n_nan_before = pd.isna(theta_raw).sum()
            n_nan_after = pd.isna(theta_clean).sum()

            interpolation_stats[depth] = {
                'n_filled': n_filled,
                'n_nan_before': n_nan_before,
                'n_nan_after': n_nan_after,
                'completeness': (n_total - n_nan_after) / n_total * 100
            }

        if verbose:
            print("\n" + "="*70)
            print("DATA QUALITY AFTER CLEANING")
            print("="*70)
            for depth, stats in interpolation_stats.items():
                n_total = len(theta_2cm)  # All same length
                n_valid = n_total - stats['n_nan_after']
                print(f"{depth:6s}: {n_valid:5d} valid ({stats['completeness']:5.1f}%), "
                      f"{stats['n_nan_after']:5d} NaN remaining, "
                      f"{stats['n_filled']:5d} filled")
    else:
        # No interpolation
        clean_data = raw_data
        theta_2cm, theta_15cm, theta_30cm, theta_40cm, theta_60cm, theta_80cm = clean_data

    if verbose:
        print("\n" + "="*70)
        print("DATA LOADING COMPLETE")
        print("="*70)

    # Return data dictionary
    return {
        'datetime': datetime_col,
        'times_seconds': times_seconds,
        'theta_2cm': theta_2cm,
        'theta_15cm': theta_15cm,
        'theta_30cm': theta_30cm,
        'theta_40cm': theta_40cm,
        'theta_60cm': theta_60cm,
        'theta_80cm': theta_80cm,
        'depths_names': depths_names,
        'raw_data': raw_data,
        'interpolation_stats': interpolation_stats if interpolate else None
    }


def load_soil_moisture(
    filepath,
    column_mapping,
    interpolate=True,
    max_gap_hours=6,
    start_date=None,
    end_date=None,
    start_day=None,
    duration_days=None,
    verbose=True
):
    """
    Generic soil moisture data loader with configurable column mappings

    Args:
        filepath: path to Excel file
        column_mapping: dict specifying column names for each depth and datetime
            Example:
            {
                'datetime': 'TIMESTAMP',  # or ('Unnamed: 0_level_0', 'Date Time') for multi-level
                'depths': {
                    '2cm': 'VWC01_Avg',  # or ('CR1000_2589', '2cm theta')
                    '15cm': 'VWC02_Avg',
                    '30cm': 'VWC03_Avg',
                    '40cm': 'VWC04_Avg',
                    '60cm': 'VWC05_Avg',
                    '80cm': 'VWC06_Avg',
                },
                'unit_conversion': 1.0,  # Multiply values by this (e.g., 0.01 for percentage to fraction)
                'multi_level_header': False,  # True if Excel has multi-level headers
            }
        interpolate: whether to interpolate missing values (default: True)
        max_gap_hours: maximum gap to interpolate in hours (default: 6)
        start_date: starting date for subset (datetime or str 'YYYY-MM-DD', default: None = use all)
        end_date: ending date for subset (datetime or str 'YYYY-MM-DD', default: None = use all)
        start_day: starting day offset from first data point (int, default: None)
        duration_days: duration in days from start_day (int, default: None)
        verbose: print progress and statistics (default: True)

    Returns:
        dict with keys:
            - 'datetime': pandas datetime index
            - 'times_seconds': time in seconds since start
            - 'theta_2cm', 'theta_15cm', etc.: soil moisture at each depth
            - 'depths_names': list of depth names
            - 'interpolation_stats': dict of interpolation statistics
    """

    if verbose:
        print("="*70)
        print("LOADING SOIL MOISTURE DATA (GENERIC LOADER)")
        print("="*70)

    # Load Excel file
    if column_mapping.get('multi_level_header', False):
        df = pd.read_excel(filepath, header=[0,1])
    else:
        df = pd.read_excel(filepath)

    # Extract datetime column
    datetime_col_name = column_mapping['datetime']
    if isinstance(datetime_col_name, tuple):
        datetime_col = df[datetime_col_name]
    else:
        datetime_col = df[datetime_col_name]

    # Extract soil moisture columns at different depths
    depths_mapping = column_mapping['depths']
    unit_conversion = column_mapping.get('unit_conversion', 1.0)

    # Handle potential header rows for multi-level headers FIRST
    if column_mapping.get('multi_level_header', False):
        datetime_col = datetime_col.iloc[2:]

    # Dictionary to store raw theta values
    theta_raw_dict = {}
    depths_names = list(depths_mapping.keys())

    for depth_name, col_name in depths_mapping.items():
        if isinstance(col_name, tuple):
            theta_values = df[col_name].values
        else:
            theta_values = df[col_name].values

        # Remove header rows for multi-level headers
        if column_mapping.get('multi_level_header', False):
            theta_values = theta_values[2:]

        # Ensure it's a numpy array before unit conversion
        theta_values = np.array(theta_values, dtype=float)

        # Apply unit conversion
        theta_values = theta_values * unit_conversion
        theta_raw_dict[depth_name] = theta_values

    # Convert to datetime
    datetime_col = pd.to_datetime(datetime_col).reset_index(drop=True)

    if verbose:
        print(f"\nData loaded:")
        print(f"  Total records: {len(datetime_col)}")

        print(f"  Time range: {datetime_col.iloc[0]} to {datetime_col.iloc[-1]}")
        print(f"  Duration: ~{(datetime_col.iloc[-1] - datetime_col.iloc[0]).days} days")

        # Calculate temporal resolution
        if len(datetime_col) > 1:
            time_diff = (datetime_col.iloc[1] - datetime_col.iloc[0]).total_seconds()
            if time_diff < 60:
                print(f"  Resolution: {time_diff:.0f} seconds")
            elif time_diff < 3600:
                print(f"  Resolution: {time_diff/60:.0f} minutes")
            else:
                print(f"  Resolution: {time_diff/3600:.1f} hours")

    # Replace missing value codes with NaN
    if verbose:
        print("\n" + "="*70)
        print("REPLACING MISSING VALUE CODES (-9999, -6999) WITH NaN")
        print("="*70)

    for depth_name in depths_names:
        theta_raw_dict[depth_name] = replace_missing_codes(theta_raw_dict[depth_name])

    if verbose:
        print("Missing value codes replaced with NaN")

    # Convert datetime to seconds since start
    t_start = datetime_col.iloc[0]
    times_seconds = (datetime_col - t_start).dt.total_seconds().values

    # ============================================================================
    # TIME PERIOD SELECTION
    # ============================================================================

    # Validate that only one selection method is used
    if (start_date is not None or end_date is not None) and (start_day is not None or duration_days is not None):
        raise ValueError("Cannot use both date-based (start_date/end_date) and day-based (start_day/duration_days) selection. Choose one method.")

    # Apply time period selection
    time_mask = np.ones(len(datetime_col), dtype=bool)  # Default: select all

    if start_date is not None or end_date is not None:
        # Date-based selection
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            time_mask &= (datetime_col >= start_date)

        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            time_mask &= (datetime_col <= end_date)

        if verbose:
            print(f"\nTime period selection (date-based):")
            print(f"  Start date: {start_date if start_date else 'first available'}")
            print(f"  End date:   {end_date if end_date else 'last available'}")

    elif start_day is not None or duration_days is not None:
        # Day-based selection (relative to first data point)
        if start_day is None:
            start_day = 0

        start_seconds = start_day * 86400

        if duration_days is not None:
            end_seconds = start_seconds + duration_days * 86400
            time_mask = (times_seconds >= start_seconds) & (times_seconds <= end_seconds)

            if verbose:
                print(f"\nTime period selection (day-based):")
                print(f"  Start day:  {start_day} (offset from first data point)")
                print(f"  Duration:   {duration_days} days")
                print(f"  End day:    {start_day + duration_days}")
        else:
            time_mask = times_seconds >= start_seconds

            if verbose:
                print(f"\nTime period selection (day-based):")
                print(f"  Start day:  {start_day} (offset from first data point)")
                print(f"  Duration:   all remaining data")

    # Apply time mask to all data
    if not time_mask.all():
        datetime_col = datetime_col[time_mask].reset_index(drop=True)
        times_seconds = times_seconds[time_mask]
        for depth_name in depths_names:
            theta_raw_dict[depth_name] = theta_raw_dict[depth_name][time_mask]

        # Recalculate times_seconds from new start
        t_start = datetime_col.iloc[0]
        times_seconds = (datetime_col - datetime_col.iloc[0]).dt.total_seconds().values

        if verbose:
            print(f"  Selected:   {len(datetime_col)} points")
            print(f"  Date range: {datetime_col.iloc[0]} to {datetime_col.iloc[-1]}")
            print(f"  Duration:   {times_seconds[-1]/86400:.1f} days")

    # Print raw data quality
    if verbose:
        print(f"\nRaw data quality (after replacing missing codes with NaN):")
        print("-"*70)
        for depth_name in depths_names:
            theta_raw = theta_raw_dict[depth_name]
            n_total = len(theta_raw)
            n_nan = pd.isna(theta_raw).sum()
            n_valid = n_total - n_nan
            pct_nan = (n_nan / n_total) * 100
            if n_valid > 0:
                print(f"{depth_name:6s}: {n_valid:5d} valid, {n_nan:5d} NaN ({pct_nan:5.2f}%), "
                      f"range=[{np.nanmin(theta_raw):.4f}, {np.nanmax(theta_raw):.4f}]")
            else:
                print(f"{depth_name:6s}: {n_valid:5d} valid, {n_nan:5d} NaN ({pct_nan:5.2f}%) - ALL MISSING!")

    # Interpolate missing values if requested
    interpolation_stats = {}
    theta_clean_dict = {}

    if interpolate:
        if verbose:
            print("\n" + "="*70)
            print("DATA CLEANING: INTERPOLATING MISSING VALUES")
            print("="*70)
            print(f"\nInterpolating with max gap = {max_gap_hours} hours:")
            print("-"*70)

        for depth_name in depths_names:
            theta_raw = theta_raw_dict[depth_name]
            theta_clean, n_filled = interpolate_missing_data(times_seconds, theta_raw, max_gap_hours=max_gap_hours)
            theta_clean_dict[depth_name] = theta_clean

            if verbose:
                print(f"{depth_name:6s}: {n_filled:5d} values interpolated")

            # Store interpolation statistics
            n_total = len(theta_clean)
            n_nan_before = pd.isna(theta_raw).sum()
            n_nan_after = pd.isna(theta_clean).sum()

            interpolation_stats[depth_name] = {
                'n_filled': n_filled,
                'n_nan_before': n_nan_before,
                'n_nan_after': n_nan_after,
                'completeness': (n_total - n_nan_after) / n_total * 100
            }

        if verbose:
            print("\n" + "="*70)
            print("DATA QUALITY AFTER CLEANING")
            print("="*70)
            for depth_name, stats in interpolation_stats.items():
                n_total = len(theta_clean_dict[depth_name])
                n_valid = n_total - stats['n_nan_after']
                print(f"{depth_name:6s}: {n_valid:5d} valid ({stats['completeness']:5.1f}%), "
                      f"{stats['n_nan_after']:5d} NaN remaining, "
                      f"{stats['n_filled']:5d} filled")
    else:
        # No interpolation
        theta_clean_dict = theta_raw_dict.copy()

    if verbose:
        print("\n" + "="*70)
        print("DATA LOADING COMPLETE")
        print("="*70)

    # Build return dictionary with standardized depth names
    result = {
        'datetime': datetime_col,
        'times_seconds': times_seconds,
        'depths_names': depths_names,
        'raw_data': [theta_raw_dict[d] for d in depths_names],
        'interpolation_stats': interpolation_stats if interpolate else None
    }

    # Add theta values with standardized names
    for depth_name in depths_names:
        result[f'theta_{depth_name}'] = theta_clean_dict[depth_name]

    return result


def prepare_surface_bc_data(
    soil_data,
    depth='2cm',
    remove_outliers=False,
    outlier_sigma=4.0,
    verbose=True
):
    """
    Prepare surface soil moisture data for PINN boundary condition

    Args:
        soil_data: dictionary returned by load_calhoun_soil_moisture()
        depth: which depth to use as surface BC (default: '2cm')
        remove_outliers: whether to remove statistical outliers (default: False)
        outlier_sigma: number of standard deviations for outlier detection (default: 4.0)
        verbose: print statistics (default: True)

    Returns:
        tuple: (times, moisture)
            - times: list of times in seconds
            - moisture: list of moisture values in m³/m³
    """

    if verbose:
        print("="*70)
        print(f"PREPARING SURFACE MOISTURE DATA ({depth})")
        print("="*70)

    # Extract data
    theta_key = f'theta_{depth.replace("cm", "cm")}'
    theta_surface = soil_data[theta_key].copy()
    times_surface = soil_data['times_seconds'].copy()
    datetime_surface = soil_data['datetime'].copy()

    # Remove NaN values
    valid_mask = ~pd.isna(theta_surface)
    theta_clean = theta_surface[valid_mask]
    times_clean = times_surface[valid_mask]
    datetime_clean = datetime_surface[valid_mask]

    if verbose:
        print(f"\nInitial data:")
        print(f"  Total points: {len(theta_surface)}")
        print(f"  NaN points: {(~valid_mask).sum()}")
        print(f"  Valid points: {len(theta_clean)}")
        print(f"  Data completeness: {len(theta_clean)/len(theta_surface)*100:.1f}%")

    # Outlier detection
    if remove_outliers:
        theta_mean = theta_clean.mean()
        theta_std = theta_clean.std()
        theta_min_physical = 0.0
        theta_max_physical = 0.6

        outlier_mask = (theta_clean < theta_mean - outlier_sigma*theta_std) | \
                       (theta_clean > theta_mean + outlier_sigma*theta_std) | \
                       (theta_clean < theta_min_physical) | \
                       (theta_clean > theta_max_physical)

        n_outliers = outlier_mask.sum()

        if verbose and n_outliers > 0:
            print(f"\nOutlier detection ({outlier_sigma}σ):")
            print(f"  Outliers detected: {n_outliers}")
            print(f"  Outlier range: [{theta_clean[outlier_mask].min():.4f}, {theta_clean[outlier_mask].max():.4f}]")

        theta_clean = theta_clean[~outlier_mask]
        times_clean = times_clean[~outlier_mask]
        datetime_clean = datetime_clean[~outlier_mask]

        if verbose and n_outliers > 0:
            print(f"  Points after removal: {len(theta_clean)}")

    if verbose:
        print(f"\nFinal data statistics:")
        print(f"  Time range: {datetime_clean.iloc[0]} to {datetime_clean.iloc[-1]}")
        print(f"  Duration: {times_clean[-1]/86400:.1f} days")
        print(f"  Number of points: {len(times_clean)}")
        print(f"  Moisture range: {theta_clean.min():.4f} - {theta_clean.max():.4f} m³/m³")
        print(f"  Mean ± std: {theta_clean.mean():.4f} ± {theta_clean.std():.4f} m³/m³")
        print(f"  Temporal resolution: {(times_clean[1]-times_clean[0])/60:.0f} minutes")
        print("="*70)

    # Return as tuple for PINN
    return (times_clean.tolist(), theta_clean.tolist())
