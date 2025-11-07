import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

import gcm.gcm_preprocessing as gcm


# --- 1. Load Data Functions ---

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


def load_clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"!! ERROR: File not found at: {file_path}")
        return None


def load_single_gcm_file(file_path, column_names):
    print(f"Loading single GCM file from {file_path}...")
    try:
        df = pd.read_csv(file_path,
                         sep=';',
                         names=column_names,
                         skiprows=[0])  # Skip the bad 7-column header

        # Convert all data to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    except FileNotFoundError:
        print(f"!! ERROR: File not found at: {file_path}")
        return None


def load_all_datasets(missions_file, mola_file, gcm_file):
    print("\n--- Step 1: Loading Datasets ---")
    df_missions = load_clean_dataset(missions_file)
    df_mola = load_clean_dataset(mola_file)
    df_gcm = load_clean_dataset(gcm_file)  # gcm_yearly_stats is clean

    if df_missions is None or df_mola is None or df_gcm is None:
        return None, None, None

    if 'lat' in df_mola.columns and 'lon' in df_mola.columns:
        df_mola = df_mola.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
        print("Standardized MOLA column names to 'latitude' and 'longitude'.")

    print("All 3 datasets loaded successfully.")
    return df_missions, df_mola, df_gcm


# --- 2. Merge Training Data ---
def create_training_data(df_missions, df_mola, df_gcm):
    print("\n--- Step 2: Creating Training Dataset ---")

    # --- The "Nearest Grid Cell" Algorithm ---
    print("Matching exact mission coordinates to GCM/MOLA 1x1 grid...")
    df_missions['grid_lat'] = np.round(df_missions['latitude'])
    df_missions['grid_lon'] = np.round(df_missions['longitude'])

    # --- Merge 1: Missions + GCM Data ---
    print("Merging mission data with GCM climate data...")
    training_df = pd.merge(
        df_missions,
        df_gcm,
        left_on=['grid_lat', 'grid_lon'],
        right_on=['latitude', 'longitude'],
        how='left',
        suffixes=('_mission', '_gcm')
    )

    # --- Merge 2: Result + MOLA Data ---
    print("Merging result with MOLA terrain data...")
    training_df = pd.merge(
        training_df,
        df_mola,
        left_on=['grid_lat', 'grid_lon'],
        right_on=['latitude', 'longitude'],
        how='left',
        suffixes=('_gcm', '_mola')
    )

    # --- Handle Missing Data ---
    print("Handling missing data (e.g., missions in 'bad' terrain)...")
    training_df['rank'] = training_df['rank'].fillna(3)
    training_df = training_df.fillna(0)

    print("Training data merged successfully.")
    return training_df


# --- 3. Train Model ---
def train_model(training_df):
    print("\n--- Step 3: Training Model ---")

    # --- Define our Features (X) and Target (y) ---
    GCM_FEATURES = [
        'extvar_29_min', 'extvar_29_max', 'extvar_29_mean',
        'extvar_44_min', 'extvar_44_max', 'extvar_44_mean',
        'temperature_min', 'temperature_max', 'temperature_mean',
        'windspeed_min', 'windspeed_max', 'windspeed_mean'
    ]
    MOLA_FEATURES = ['rank']
    FEATURES = GCM_FEATURES + MOLA_FEATURES
    TARGET = 'success_landing'

    # Filter for only the columns that actually exist
    existing_features = [col for col in FEATURES if col in training_df.columns]
    if not existing_features:
        print("!! ERROR: No features found in training data. Check GCM/MOLA file column names.")
        return None, []

    X_train = training_df[existing_features]
    y_train = training_df[TARGET]

    print(f"Training model on {len(X_train)} mission samples...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # --- Model Evaluation ---
    print("Evaluating model with Leave-One-Out Cross-Validation...")
    loo = LeaveOneOut()
    scores = cross_val_score(model, X_train, y_train, cv=loo, scoring='accuracy')
    print(f"Estimated Model Accuracy: {scores.mean() * 100:.2f}%")

    print("Model training complete.")
    return model, existing_features


# --- 4. Prepare Prediction Set (The whole planet) ---
def prepare_global_prediction_set(df_mola, df_gcm_prediction_file):
    print("\n--- Step 4: Preparing Global Prediction Set (All Mars) ---")

    # Start with the full GCM prediction file (e.g., out_grid1x1deg_0h_0sollon.csv)
    # We must merge the sparse MOLA map onto it.
    global_map_df = pd.merge(
        df_gcm_prediction_file,
        df_mola,
        on=['latitude', 'longitude'],
        how='left'  # Keep all GCM grid cells
    )

    # Just like in training, fill all non-optimal terrain with rank 3
    global_map_df['rank'] = global_map_df['rank'].fillna(3)

    # Fill any other missing data just in case
    global_map_df = global_map_df.fillna(0)

    # --- CRITICAL: Calculate windspeed for the prediction set ---
    # This was missing from your previous file.
    if 'zonal wind' in global_map_df.columns and 'meridional wind' in global_map_df.columns:
        print("Calculating windspeed for prediction set...")
        global_map_df['windspeed_mean'] = np.sqrt(
            global_map_df['zonal wind'] ** 2 + global_map_df['meridional wind'] ** 2)
        # We assume min/max are the same as mean for a single file
        global_map_df['windspeed_min'] = global_map_df['windspeed_mean']
        global_map_df['windspeed_max'] = global_map_df['windspeed_mean']
    else:
        print("!! WARNING: 'zonal wind' or 'meridional wind' not in prediction file!")

    print(f"Global prediction set prepared with {len(global_map_df)} grid cells.")
    return global_map_df


# --- 5. Predict and Save ---
def predict_and_save_map(model, features_list, global_map_df, output_file):
    if model is None:
        print("!! ERROR: Model was not trained. Aborting prediction.")
        return

    print("\n--- Step 5: Predicting Suitability for all of Mars ---")

    # --- CRITICAL: Rename columns to match model's features ---
    # The model was trained on 'temperature_mean', but the single GCM
    # file just has 'temperature'. We must rename them.
    # This is a robust way to handle the difference in file types.
    rename_map = {
        'extvar_29': 'extvar_29_mean',
        'extvar_44': 'extvar_44_mean',
        'temperature': 'temperature_mean',
        # (add min/max if your single file has them,
        #  but for now we assume it only provides the 'mean' value)
    }
    global_map_df_renamed = global_map_df.rename(columns=rename_map)

    # Fill in any missing 'min' or 'max' columns with the 'mean' value
    # This is a form of "test-time imputation"
    for col in features_list:
        if col not in global_map_df_renamed.columns:
            if '_min' in col:
                base_col = col.replace('_min', '')
                if base_col in global_map_df_renamed.columns:
                    global_map_df_renamed[col] = global_map_df_renamed[base_col]
            elif '_max' in col:
                base_col = col.replace('_max', '')
                if base_col in global_map_df_renamed.columns:
                    global_map_df_renamed[col] = global_map_df_renamed[base_col]

    # Fill any remaining NaNs with 0
    X_global = global_map_df_renamed[features_list].fillna(0)

    # 2. Predict!
    print(f"Using trained model to predict suitability for {len(X_global)} grid cells...")
    probabilities = model.predict_proba(X_global)[:, 1]

    # 3. Save the final map
    final_map = global_map_df[['latitude', 'longitude']].copy()
    final_map['suitability_score'] = probabilities

    final_map.to_csv(output_file, index=False)

    print(f"\n--- GLOBAL PREDICTION COMPLETE ---")
    print(f"Final prediction map saved to: {output_file}")

    print("\n--- Top 10 Most Suitable Landing Zones (from model) ---")
    print(final_map.sort_values(by='suitability_score', ascending=False).head(10))


# --- Main Execution ---
def main():
    missions_file = "../data/past_missions.csv"
    gcm_file = "../data/gcm/datasets/gcm_yearly_stats.csv"
    mola_file = "../data/mola/mars_landing_sites_1deg.csv"

    headers_file = "../data/gcm/gcm_headers.txt"

    final_prediction_map_file = "../output/final_suitability_map.csv"

    # 1. Load data
    df_missions, df_mola, df_gcm = load_all_datasets(missions_file, mola_file, gcm_file)
    if df_missions is None:
        print("Aborting run due to file loading error.")
        return

    # 2. Create the small, merged training dataset
    training_df = create_training_data(df_missions, df_mola, df_gcm)

    # 3. Train the model on the training dataset
    model, features_list = train_model(training_df)

    if model is None:
        print("Aborting run due to model training error.")
        return

    # 4c. Prepare the global prediction set by merging MOLA and the single GCM file
    prediction_input_file = "../data/gcm/every_month/out_grid1x1deg_0h_0sollon.csv"
    prediction_output_file = "../data/gcm/datasets/gcm_prediction_input_prepared.csv"
    prediction_df = gcm.prepare_single_gcm_file_for_prediction(headers_file, prediction_input_file, prediction_output_file)
    global_map_df = prepare_global_prediction_set(df_mola, prediction_df)

    # 5. Use the trained model to predict on the *entire planet*
    predict_and_save_map(model, features_list, global_map_df, final_prediction_map_file)


if __name__ == "__main__":
    main()