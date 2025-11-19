import pandas as pd
import numpy as np
import glob
import os


def load_column_headers(headers_file_path):
    print(f"Loading column headers from {headers_file_path}...")
    try:
        with open(headers_file_path, 'r') as f:
            header_line = f.readline().strip()
            column_names = header_line.split(',')
        return column_names
    except FileNotFoundError:
        print(f"!! ERROR: Header file not found at: {headers_file_path}")
        return None


def create_gcm_yearly_stats(headers_file_path, gcm_folder_path, output_file_path):
    # --- Load Headers ---
    column_names = load_column_headers(headers_file_path)
    if column_names is None: return None

    # --- Load all 24 files ---
    csv_files = glob.glob(os.path.join(gcm_folder_path, "*.csv"))
    if not csv_files:
        print(f"!! ERROR: No CSV files found in folder: {gcm_folder_path}")
        return None

    print(f"Found {len(csv_files)} GCM files. Loading and concatenating...")
    all_dataframes = []
    for f in csv_files:
        df = pd.read_csv(f, sep=';', names=column_names, skiprows=[0])
        all_dataframes.append(df)

    full_data = pd.concat(all_dataframes)
    full_data = full_data.apply(pd.to_numeric, errors='coerce')

    # --- Define our key climate parameters ---
    # Added 'atm pressure' as requested
    key_params = [
        'extvar_29',  # Surface Pressure
        'extvar_44',  # Dust Opacity
        'extvar_28',  # Turbulence
        'atm pressure',  # Atmospheric Pressure
        'temperature',
        'zonal wind',
        'meridional wind'
    ]

    core_data = full_data[['latitude', 'longitude'] + key_params]

    # --- Aggregate ---
    print("Aggregating 24 files into one yearly min/max/mean map...")

    # Calculate total windspeed
    core_data['windspeed'] = np.sqrt(core_data['zonal wind'] ** 2 + core_data['meridional wind'] ** 2)

    # List of params to aggregate (replacing zonal/meridional with total windspeed)
    key_params_agg = [
        'extvar_29',
        'extvar_44',
        'extvar_28',
        'atm pressure',
        'temperature',
        'windspeed'
    ]

    agg_dict = {}
    for param in key_params_agg:
        agg_dict[param] = ['min', 'max', 'mean']

    yearly_stats_map = core_data.groupby(['latitude', 'longitude']).agg(agg_dict).reset_index()

    # Flatten column names
    new_cols = ['latitude', 'longitude']
    for param in key_params_agg:
        for stat in ['min', 'max', 'mean']:
            # This handles space in 'atm pressure' -> 'atm_pressure_min'
            clean_param = param.replace(' ', '_')
            new_cols.append(f"{clean_param}_{stat}")

    yearly_stats_map.columns = new_cols

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    yearly_stats_map.to_csv(output_file_path, index=False)
    print(f"Yearly stats map saved to: {output_file_path}")
    return yearly_stats_map


def prepare_single_gcm_file_for_prediction(headers_file_path, single_file_path, output_file_path):
    # --- Load Headers ---
    column_names = load_column_headers(headers_file_path)
    if column_names is None: return None

    # --- Load File ---
    print(f"\nLoading single GCM file from: {single_file_path}")
    try:
        df = pd.read_csv(single_file_path, sep=';', names=column_names, skiprows=[0])
        df = df.apply(pd.to_numeric, errors='coerce')
    except FileNotFoundError:
        print(f"!! ERROR: File not found at: {single_file_path}")
        return None

    print("Preparing file for prediction...")

    # Calculate windspeed
    df['windspeed'] = np.sqrt(df['zonal wind'] ** 2 + df['meridional wind'] ** 2)

    final_df = df[['latitude', 'longitude']].copy()

    # --- KEY FIX: Create min/max/mean for ALL relevant parameters ---
    # This ensures the prediction file matches the training file structure exactly.

    params_to_expand = [
        'extvar_29',
        'extvar_44',
        'extvar_28',
        'atm pressure',
        'temperature',
        'windspeed'
    ]

    for param in params_to_expand:
        clean_param = param.replace(' ', '_')
        # For a single file, min = max = mean = value
        final_df[f'{clean_param}_min'] = df[param]
        final_df[f'{clean_param}_max'] = df[param]
        final_df[f'{clean_param}_mean'] = df[param]

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    final_df.to_csv(output_file_path, index=False)
    print(f"Prepared prediction file saved to: {output_file_path}")
    return final_df


def main():
    # Adjust paths as needed
    headers_file = "data/gcm/gcm_headers.txt" 
    gcm_folder = "data/gcm/every_month"
    gcm_stats_output = "data/gcm/datasets/gcm_yearly_stats.csv"

    # 1. Create Training Data Source
    create_gcm_yearly_stats(headers_file, gcm_folder, gcm_stats_output)

    # 2. Create Prediction Data Source (using one example month)
    pred_input = "data/gcm/every_month/out_grid1x1deg_0h_0sollon.csv"
    pred_output = "data/gcm/datasets/gcm_prediction_input_prepared.csv"
    prepare_single_gcm_file_for_prediction(headers_file, pred_input, pred_output)


if __name__ == "__main__":
    main()