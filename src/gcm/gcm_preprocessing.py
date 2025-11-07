import pandas as pd
import numpy as np
import glob
import os


def load_column_headers(headers_file_path):
    """
    Loads the column header list from your text file.
    """
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
    """
    Loads all 24 "monthly" CSVs, stacks them,
    and computes the single "Yearly Average Climate Map" (min/max/mean).
    This file is used to build the TRAINING dataset.
    """

    # --- Load Headers ---
    column_names = load_column_headers(headers_file_path)
    if column_names is None:
        return None

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
    # These are the key columns from your header file
    key_params = [
        'extvar_29',  # Surface Pressure
        'extvar_44',  # Dust Opacity
        'temperature',
        'zonal wind',
        'meridional wind'
    ]

    # Keep only the columns we need
    core_data = full_data[['latitude', 'longitude'] + key_params]

    # --- Aggregate to find min/max/mean for the year ---
    print("Aggregating 24 files into one yearly min/max/mean map...")

    # Calculate total windspeed *before* aggregating
    core_data['windspeed'] = np.sqrt(core_data['zonal wind'] ** 2 + core_data['meridional wind'] ** 2)

    # We no longer need the individual wind components
    key_params_agg = [
        'extvar_29',
        'extvar_44',
        'temperature',
        'windspeed'  # Aggregate on the calculated windspeed
    ]

    agg_dict = {}
    for param in key_params_agg:
        agg_dict[param] = ['min', 'max', 'mean']

    # Run the aggregation
    yearly_stats_map = core_data.groupby(['latitude', 'longitude']).agg(agg_dict).reset_index()

    # --- Clean up the column names ---
    new_cols = ['latitude', 'longitude']
    for param in key_params_agg:
        for stat in ['min', 'max', 'mean']:
            # e.g., 'extvar_29_min', 'windspeed_mean'
            new_cols.append(f"{param.replace(' ', '_')}_{stat}")

    yearly_stats_map.columns = new_cols

    # --- Save the final map ---
    yearly_stats_map.to_csv(output_file_path, index=False)
    print(f"\n--- GCM Training Data Prep Complete ---")
    print(f"Yearly stats map (min/max/avg) saved to: {output_file_path}")
    print("\nExample of GCM stats map:")
    print(yearly_stats_map.head())

    return yearly_stats_map


def prepare_single_gcm_file_for_prediction(headers_file_path, single_file_path, output_file_path):

    # --- Load Headers ---
    column_names = load_column_headers(headers_file_path)
    if column_names is None:
        return None

    # --- Load the single "dirty" GCM file ---
    print(f"\nLoading single GCM file from: {single_file_path}")
    try:
        df = pd.read_csv(single_file_path,
                         sep=';',
                         names=column_names,
                         skiprows=[0])
        df = df.apply(pd.to_numeric, errors='coerce')
    except FileNotFoundError:
        print(f"!! ERROR: File not found at: {single_file_path}")
        return None

    print("Preparing file for prediction...")

    # --- Calculate derived windspeed ---
    df['windspeed'] = np.sqrt(df['zonal wind'] ** 2 + df['meridional wind'] ** 2)

    # --- This is the key "imputation" step ---
    # The model expects 'temperature_mean', but this file only has 'temperature'.
    # We will "fake" the min, max, and mean columns by copying the single value.

    final_df = df[['latitude', 'longitude']].copy()

    # Engineer Temperature features
    final_df['temperature_min'] = df['temperature']
    final_df['temperature_max'] = df['temperature']
    final_df['temperature_mean'] = df['temperature']

    # Engineer Pressure features
    final_df['extvar_29_min'] = df['extvar_29']
    final_df['extvar_29_max'] = df['extvar_29']
    final_df['extvar_29_mean'] = df['extvar_29']

    # Engineer Dust features
    final_df['extvar_44_min'] = df['extvar_44']
    final_df['extvar_44_max'] = df['extvar_44']
    final_df['extvar_44_mean'] = df['extvar_44']

    # Engineer Windspeed features
    final_df['windspeed_min'] = df['windspeed']
    final_df['windspeed_max'] = df['windspeed']
    final_df['windspeed_mean'] = df['windspeed']

    # Save the prepared prediction file
    final_df.to_csv(output_file_path, index=False)
    print(f"\n--- GCM Prediction File Prep Complete ---")
    print(f"Prepared prediction file saved to: {output_file_path}")
    print("\nExample of prepared prediction file:")
    print(final_df.head())

    return final_df


def main():
    headers_file = "../../data/gcm/gcm_headers.txt"

    # --- Task 1: Create the TRAINING data file ---
    gcm_folder = "../../data/gcm/every_month"
    gcm_stats_output_file = "../../data/gcm/datasets/gcm_yearly_stats.csv"
    create_gcm_yearly_stats(headers_file, gcm_folder, gcm_stats_output_file)


    # prediction_input_file = "../../data/gcm/every_month/out_grid1x1deg_0h_0sollon.csv"
    # prediction_output_file = "../../data/gcm/datasets/gcm_prediction_input_prepared.csv"
    # prepare_single_gcm_file_for_prediction(headers_file, prediction_input_file, prediction_output_file)


if __name__ == "__main__":
    main()