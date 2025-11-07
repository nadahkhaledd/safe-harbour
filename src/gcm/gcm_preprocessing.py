import pandas as pd
import glob
import os

def create_gcm_yearly_stats(headers_file, gcm_folder, output_file):
    # --- Load Headers ---
    print(f"Loading column headers from {headers_file}...")
    try:
        with open(headers_file, 'r') as f:
            header_line = f.readline().strip()
            column_names = header_line.split(',')
    except FileNotFoundError:
        print(f"!! ERROR: Header file not found at: {headers_file}")
        return

    # --- Load all 24 files ---
    csv_files = glob.glob(os.path.join(gcm_folder, "*.csv"))
    if not csv_files:
        print(f"!! ERROR: No CSV files found in folder: {gcm_folder}")
        return

    print(f"Found {len(csv_files)} GCM files. Loading and concatenating...")
    all_dataframes = []
    for f in csv_files:
        df = pd.read_csv(f, sep=';', names=column_names, skiprows=[0])
        all_dataframes.append(df)

    full_data = pd.concat(all_dataframes)
    full_data = full_data.apply(pd.to_numeric, errors='coerce')

    # --- Define our key climate parameters ---
    # These are the 6 we identified from your header file
    key_params = [
        'extvar_29',  # Surface Pressure
        'extvar_44',  # Dust Opacity
        'temperature',
        'zonal wind',
        'meridional wind'
    ]

    # Keep only the columns we need
    core_data = full_data[['latitude', 'longitude'] + key_params]

    # --- This is the "algorithm" you wanted ---
    # We group by location and apply multiple aggregations (min, max, mean)
    print("Aggregating 24 files into one yearly min/max/mean map...")

    # Create the aggregation dictionary
    # For each parameter, we want 'min', 'max', and 'mean'
    agg_dict = {}
    for param in key_params:
        agg_dict[param] = ['min', 'max', 'mean']

    # Run the aggregation
    yearly_stats_map = core_data.groupby(['latitude', 'longitude']).agg(agg_dict).reset_index()

    # --- Clean up the column names ---
    # The new columns will be messy (e.g., ('extvar_29', 'min'))
    # Let's flatten them to 'extvar_29_min', 'extvar_29_max', etc.
    new_cols = ['latitude', 'longitude']
    for param in key_params:
        for stat in ['min', 'max', 'mean']:
            new_cols.append(f"{param.replace(' ', '_')}_{stat}")  # e.g., 'zonal_wind_min'

    yearly_stats_map.columns = new_cols

    # --- Save the final map ---
    yearly_stats_map.to_csv(output_file, index=False)
    print(f"\n--- GCM Prep Complete ---")
    print(f"Yearly stats map (min/max/avg) saved to: {output_file}")
    print("\nExample of GCM stats map:")
    print(yearly_stats_map.head())

    return yearly_stats_map


def main():
    headers_file = "../../data/gcm/gcm_headers.txt"
    gcm_folder = "../../data/gcm/every_month"
    output_file = "../../data/gcm/datasets/gcm_yearly_stats.csv"
    create_gcm_yearly_stats(headers_file, gcm_folder, output_file)

if __name__ == "__main__":
    main()