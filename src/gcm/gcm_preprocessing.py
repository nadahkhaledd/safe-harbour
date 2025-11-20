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
        'temperature',
        'zonal wind',
        'meridional wind',
        'extvar_40',
        'extvar_19',
        'extvar_28',
        'extvar_25',
        'atm pressure'
    ]

    core_data = full_data[['latitude', 'longitude'] + key_params]

    # --- Aggregate ---
    print("Aggregating 24 files into one yearly min/max/mean map...")

    # Calculate total windspeed
    core_data['windspeed'] = np.sqrt(core_data['zonal wind'] ** 2 + core_data['meridional wind'] ** 2)

    # List of params to aggregate (replacing zonal/meridional with total windspeed)
    key_params_agg = [
        'temperature',
        'extvar_40',
        'extvar_19',
        'extvar_28',
        'extvar_25',
        'atm pressure',
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
        'temperature',
        'extvar_40',
        'extvar_19',
        'extvar_28',
        'extvar_25',
        'atm pressure',
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


def load_clean_dataset(file_path):
    """Loads a simple, clean CSV file (like past_missions or mola_map)."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"!! ERROR: File not found at: {file_path}")
        return None

def load_mola(mola_file):
    """Loads MOLA dataset and standardizes column names."""
    df_mola = load_clean_dataset(mola_file)
    if df_mola is not None:
        if 'lat' in df_mola.columns and 'lon' in df_mola.columns:
            df_mola = df_mola.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    return df_mola

def load_all_datasets():
    """Loads all three core datasets for TRAINING."""
    missions_file = "../../data/past_missions.csv"
    gcm_file = "../../data/gcm/datasets/gcm_yearly_stats.csv"
    mola_file = "../../data/mola/mars_landing_sites_1deg.csv"

    print("\n--- Loading Datasets ---")
    df_missions = load_clean_dataset(missions_file)
    df_mola = load_mola(mola_file)
    df_gcm = load_clean_dataset(gcm_file)

    if df_missions is None or df_mola is None or df_gcm is None:
        return None, None, None

    print("All 3 training-related datasets loaded successfully.")
    return df_missions, df_mola, df_gcm



def create_training_data(df_missions, df_mola, df_gcm):
    print("\n--- Step 2: Creating Training Dataset ---")

    # Nearest Grid Cell Algorithm
    df_missions['grid_lat'] = np.round(df_missions['latitude'])
    df_missions['grid_lon'] = np.round(df_missions['longitude'])

    # Merge 1: Missions + GCM Data (Climate)
    print("Merging mission data with GCM climate data...")
    training_df = pd.merge(
        df_missions, df_gcm,
        left_on=['grid_lat', 'grid_lon'], right_on=['latitude', 'longitude'],
        how='left', suffixes=('_mission', '_gcm')
    )


    climate_cols = [c for c in df_gcm.columns if c not in ['latitude', 'longitude']]

    for index, row in training_df.iterrows():
        # Check if critical data is missing (using temperature as a proxy)
        if pd.isna(row['temperature_mean']) or row['temperature_mean'] == 0:
            print(f"Fixing missing climate data for mission: {row['mission']} at {row['grid_lat']}, {row['grid_lon']}")

            # Look for neighbors ( +/- 1 degree)
            lat, lon = row['grid_lat'], row['grid_lon']
            neighbors = df_gcm[
                (df_gcm['latitude'].between(lat - 2, lat + 2)) &
                (df_gcm['longitude'].between(lon - 2, lon + 2))
                ]

            if not neighbors.empty:
                # Take the mean of the neighbors
                avg_climate = neighbors[climate_cols].mean()
                # Fill in the missing values for this row
                training_df.loc[index, climate_cols] = avg_climate
                print(f"  -> Filled with average of {len(neighbors)} neighbors.")
            else:
                print("  -> No neighbors found! Data remains missing.")

    # Merge 2: Result + MOLA Data (Terrain Rank)
    print("Merging result with MOLA terrain data...")
    training_df = pd.merge(
        training_df, df_mola,
        left_on=['grid_lat', 'grid_lon'], right_on=['latitude', 'longitude'],
        how='left', suffixes=('_gcm', '_mola')
    )

    # Handle Missing Data (Fill rank with 3)
    training_df['rank'] = training_df['rank'].fillna(3)

    # Final cleanup: fill any remaining NaNs with 0 (only if smart fill failed)
    training_df = training_df.fillna(0)

    print("Training data merged successfully.")
    return training_df

def get_training_dataset():

    # 2. Load Data
    df_missions, df_mola, df_gcm_stats = load_all_datasets()
    if df_missions is None: return

    # 3. Create Training Set & Train Model
    training_df = create_training_data(df_missions, df_mola, df_gcm_stats)
    training_df.to_csv("../../output/final/training_dataset.csv", index=False)

    return training_df

def prepare_prediction_file(headers_file):
    pred_input = "../../data/gcm/every_month/out_grid1x1deg_0h_0sollon.csv"
    pred_output = "../../data/gcm/datasets/gcm_prediction_input_prepared.csv"
    prepare_single_gcm_file_for_prediction(headers_file, pred_input, pred_output)


def aggregate_gcm_data(headers_file):
    gcm_folder = "../../data/gcm/every_month"
    gcm_stats_output = "../../data/gcm/datasets/gcm_yearly_stats.csv"
    # 1. Create Training Data Source
    create_gcm_yearly_stats(headers_file, gcm_folder, gcm_stats_output)


def main():
    # Adjust paths as needed
    headers_file = "../../data/gcm/gcm_headers.txt"
    aggregate_gcm_data(headers_file)

    get_training_dataset()

    # Create Prediction Data Source (using one example month)
    prepare_prediction_file(headers_file)


if __name__ == "__main__":
    main()