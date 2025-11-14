import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

CONSTRAINTS = {
    # Using 'extvar_29' (Surface Pressure in Pascals)
    'MIN_PRESSURE_PA': 20.0,

    # This will use our calculated 'windspeed_ms' (Wind in m/s)
    'MAX_WINDSPEED_MS': 15.0,

    # Using 'extvar_46' (Total dust opacity)
    'MAX_DUST_OPACITY': 0.5,

    # Using 'temperature' (Temperature in Kelvin)
    'MIN_TEMP_K': 50.0
}


def load_column_headers(headers_file_path):
    print(f"Loading column headers from {headers_file_path}...")
    try:
        with open(headers_file_path, 'r') as f:
            header_line = f.readline().strip()
            column_names = header_line.split(',')

        if len(column_names) < 80:  # Check for a reasonable number of columns
            print(f"!! WARNING: Your header file has only {len(column_names)} columns!")

        return column_names
    except FileNotFoundError:
        print(f"!! ERROR: Header file not found at: {headers_file_path}")
        return None


def load_and_average_gcm_data(folder_path, column_names):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print(f"!! ERROR: No CSV files found in folder: {folder_path}")
        return None

    print(f"Found {len(csv_files)} GCM files. Loading and averaging...")

    all_dataframes = []
    for f in csv_files:
        df = pd.read_csv(f,
                         sep=';',
                         names=column_names,
                         skiprows=[0])
        all_dataframes.append(df)

    # Stack all DataFrames into one giant table
    full_data = pd.concat(all_dataframes)

    # Convert all data to numeric *after* concatenating
    print("Converting all data to numeric...")
    full_data = full_data.apply(pd.to_numeric, errors='coerce')

    # Group by location and calculate the mean for all climate variables.
    print("Averaging all 'monthly' maps into one 'yearly' map...")
    # This correctly uses 'latitude' and 'longitude' from your header file
    yearly_avg_map = full_data.groupby(['latitude', 'longitude']).mean().reset_index()

    print("Averaging complete.")
    return yearly_avg_map


def calculate_derived_parameters(df):
    print("Calculating derived parameters (e.g., windspeed)...")

    df['windspeed_ms'] = np.sqrt(df['zonal wind'] ** 2 + df['meridional wind'] ** 2)

    return df


def apply_suitability_constraints(df, constraints):
    print("Applying landing site constraints...")

    df['passes_pressure'] = df['extvar_29'] >= constraints['MIN_PRESSURE_PA']
    df['passes_wind'] = df['windspeed_ms'] <= constraints['MAX_WINDSPEED_MS']
    df['passes_dust'] = df['extvar_46'] <= constraints['MAX_DUST_OPACITY']
    df['passes_temp'] = df['temperature'] >= constraints['MIN_TEMP_K']

    # The final "is_suitable" column is True ONLY if all other checks are True
    df['is_suitable'] = (
            df['passes_pressure'] &
            df['passes_wind'] &
            df['passes_dust'] &
            df['passes_temp']
    )

    # Convert True/False to 1 (Suitable) / 0 (Unsuitable)
    df['is_suitable'] = df['is_suitable'].astype(int)

    return df


def get_results(df_map_final, output_file):
    final_columns = [
        'latitude',
        'longitude',
        'is_suitable',
        'passes_pressure',
        'passes_wind',
        'passes_dust',
        'passes_temp'
    ]

    existing_columns = [col for col in final_columns if col in df_map_final.columns]
    final_map_data = df_map_final[existing_columns]

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    final_map_data.to_csv(output_file, index=False)

    print("\n--- ANALYSIS COMPLETE ---")
    print(f"Final suitability map saved to: {output_file}")

    # Show how many sites passed
    suitable_sites = final_map_data['is_suitable'].sum()
    total_sites = len(final_map_data)
    print(f"Total suitable landing sites found: {suitable_sites} out of {total_sites}")

    print("\n--- Example of Suitable Sites ---")
    print(final_map_data[final_map_data['is_suitable'] == 1].head())


def main():

    # --- Define file paths ---
    headers_file = "../data/gcm/gcm_headers.txt"

    gcm_folder = "../data/gcm/every_month"

    output_file = "../output/gcm/gcm_suitability_map_binary.csv"

    # --- Run Analysis ---

    # 1. Load Headers
    column_names = load_column_headers(headers_file)
    if column_names is None:
        return  # Stop if headers didn't load

    # 2. Load and Average all GCM files
    df_map = load_and_average_gcm_data(gcm_folder, column_names)

    if df_map is None:
        return  # Stop if data loading failed

    # 3. Calculate Windspeed
    df_map = calculate_derived_parameters(df_map)

    # 4. Apply Binary Constraints
    df_map_final = apply_suitability_constraints(df_map, CONSTRAINTS)

    # --- 5. Get and Save Results ---
    get_results(df_map_final, output_file)


if __name__ == "__main__":
    main()