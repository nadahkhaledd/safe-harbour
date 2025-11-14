import pandas as pd
import numpy as np


def create_mock_data():
    """
    Creates fake data for demonstration, since we don't have the real files.
    """
    print("Creating mock data for demonstration...")

    # 1. Mock Past Missions (Your "Ground Truth")
    # Using real data for Perseverance, Curiosity, and (unsuccessful) Beagle 2
    missions = {
        'mission_name': ['Perseverance', 'Curiosity', 'Beagle 2'],
        'latitude_exact': [18.4446, 4.5895, 11.5262],
        'longitude_exact': [77.4509, 137.4417, 90.4248],
        'success_landing': [1, 1, 0]  # 1=Success, 0=Failure
    }
    mock_missions_df = pd.DataFrame(missions)
    mock_missions_df.to_csv("past_missions.csv", index=False)
    print("Created 'past_missions.csv'")

    # 2. Mock MOLA Data (from your terrain team)
    # This must be a 1x1 degree grid, just like the GCM map
    lats = np.arange(-90, 91, 1)
    lons = np.arange(0, 360, 1)
    grid_lat, grid_lon = np.meshgrid(lats, lons)

    mola_data = {
        'latitude': grid_lat.flatten(),
        'longitude': grid_lon.flatten(),
        # Just random data for slope, elevation, and suitability
        'slope': np.random.uniform(0, 20, len(grid_lat.flatten())),
        'elevation_km': np.random.uniform(-4, 5, len(grid_lat.flatten())),
        'terrain_is_suitable': np.random.randint(0, 2, len(grid_lat.flatten()))
    }
    mock_mola_df = pd.DataFrame(mola_data)
    mock_mola_df.to_csv("mola_map.csv", index=False)
    print("Created 'mola_map.csv' (global 1x1 grid)")

    # 3. Mock GCM Yearly Stats (The file from Step 1)
    # We'll just copy the MOLA grid and add GCM stats
    mock_gcm_df = mock_mola_df[['latitude', 'longitude']].copy()
    mock_gcm_df['extvar_29_min'] = np.random.uniform(500, 700, len(mock_gcm_df))
    mock_gcm_df['extvar_29_max'] = mock_gcm_df['extvar_29_min'] + 100
    mock_gcm_df['extvar_29_mean'] = mock_gcm_df['extvar_29_min'] + 50
    # ... (add other min/max/mean for temp, wind, dust)

    mock_gcm_df.to_csv("gcm_yearly_stats.csv", index=False)
    print("Created 'gcm_yearly_stats.csv' (global 1x1 grid)")


def create_training_dataset(missions_file, mola_file, gcm_stats_file, output_file):
    """
    Merges the three data sources using the "nearest grid" (rounding)
    algorithm to create the final training dataset.
    """
    print("\n--- Creating Training Dataset ---")

    # 1. Load all 3 source files
    try:
        df_missions = pd.read_csv(missions_file)
        df_mola = pd.read_csv(mola_file)
        df_gcm = pd.read_csv(gcm_stats_file)
    except FileNotFoundError as e:
        print(f"!! ERROR: Missing data file. {e}")
        print("Please run `create_mock_data()` first if you are testing.")
        return

    # --- 2. The "Nearest Neighbor" Algorithm (Rounding) ---
    # We round the *exact* mission coordinates to the *nearest* 1-degree grid cell.
    print("Matching exact mission coordinates to GCM/MOLA grid...")
    df_missions['latitude'] = np.round(df_missions['latitude_exact'])
    df_missions['longitude'] = np.round(df_missions['longitude_exact'])

    # --- 3. Merge the datasets ---

    # Merge missions with MOLA data
    # This links the mission to its terrain data
    training_df = pd.merge(
        df_missions,
        df_mola,
        on=['latitude', 'longitude'],
        how='left'
    )

    # Merge the result with GCM data
    # This links the mission to its climate data
    training_df = pd.merge(
        training_df,
        df_gcm,
        on=['latitude', 'longitude'],
        how='left'
    )

    # --- 4. Clean and Save ---

    # Drop columns we don't need for training
    final_cols_to_keep = [
        'mission_name',
        'latitude_exact',
        'longitude_exact',
        'success_landing',  # This is our TARGET (y)
        'slope',  # This is a FEATURE (X)
        'elevation_km',  # This is a FEATURE (X)
        'terrain_is_suitable',  # This is a FEATURE (X)
        'extvar_29_min',  # This is a FEATURE (X)
        'extvar_29_max',  # ...
        'extvar_29_mean',
        # ... (all other GCM features)
    ]

    # Select only the columns that actually exist in our merged file
    final_cols = [col for col in final_cols_to_keep if col in training_df.columns]
    final_training_df = training_df[final_cols]

    final_training_df.to_csv(output_file, index=False)

    print(f"\n--- Training Data Ready ---")
    print(f"Final training dataset saved to: {output_file}")
    print(final_training_df)

    return final_training_df


# def main():
#     create_mock_data()
#     create_training_dataset(
#         "past_missions.csv",
#         "mola_map.csv",
#         "gcm_yearly_stats.csv",
#         "training_dataset.csv"
#     )
#
# if __name__ == "__main__":
#     main()