import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score


def load_all_datasets(missions_file, mola_file, gcm_file):
    print("\n--- Step 1: Loading Datasets ---")
    try:
        df_missions = load_dataset_from_file(missions_file)
        df_mola = load_dataset_from_file(mola_file)
        df_gcm = load_dataset_from_file(gcm_file)
        return df_missions, df_mola, df_gcm
    except FileNotFoundError as e:
        print(f"!! ERROR: {e}")
        print("Please check your file paths.")
        return None, None, None


def load_dataset_from_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"!! ERROR: File not found at: {file_path}")
        return None

def create_training_data(df_missions, df_mola, df_gcm):
    print("\n--- Step 2: Creating Training Dataset ---")

    # --- The "Nearest Grid Cell" Algorithm ---
    # We round the *exact* mission coordinates to the *nearest* 1-degree grid cell
    # (e.g., 18.4447 -> 18.0, 77.4508 -> 77.0)
    print("Matching exact mission coordinates to GCM/MOLA 1x1 grid...")
    df_missions['grid_lat'] = np.round(df_missions['latitude'])
    df_missions['grid_lon'] = np.round(df_missions['longitude'])

    # --- Merge 1: Missions + GCM Data ---
    # This links the mission to its climate data.
    # We use 'how=left' to ensure we keep ALL missions.
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
    # This links the mission to its terrain rank.
    # We use 'how=left' because the MOLA map is sparse (only good zones).
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
    # This is your key insight:
    # If a mission landed in a "bad" terrain zone (not in the mola_map file),
    # its 'rank' will be NaN (missing). We fill this with a new category, 3.
    print("Handling missing data (e.g., missions in 'bad' terrain)...")
    training_df['rank'] = training_df['rank'].fillna(3)

    # Also fill any potential missing climate data with 0 (a simple imputation)
    # This ensures the model can train, even if a mission's grid cell
    # was missing from the GCM file for some reason.
    training_df = training_df.fillna(0)

    print("Training data merged successfully.")
    return training_df


def train_model(training_df):
    print("\n--- Step 3: Training Model ---")

    # --- Define our Features (X) and Target (y) ---
    # Assumes your GCM stats file has these columns:
    GCM_FEATURES = [
        'extvar_29_min', 'extvar_29_max', 'extvar_29_mean',
        'extvar_44_min', 'extvar_44_max', 'extvar_44_mean',
        'temperature_min', 'temperature_max', 'temperature_mean',
        'windspeed_min', 'windspeed_max', 'windspeed_mean'
        # Add any other GCM min/max/mean columns here
    ]

    # The 'rank' from your MOLA file is now a critical feature
    MOLA_FEATURES = ['rank']

    FEATURES = GCM_FEATURES + MOLA_FEATURES
    TARGET = 'success_landing'

    # Filter for only the columns that actually exist in the merged file
    existing_features = [col for col in FEATURES if col in training_df.columns]

    # Check if we have any features to train on
    if not existing_features:
        print("!! ERROR: No features found in the training data. Check column names in your GCM/MOLA files.")
        return None, []

    X_train = training_df[existing_features]
    y_train = training_df[TARGET]

    print(f"Training model on {len(X_train)} mission samples...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # --- Model Evaluation (Cross-Validation) ---
    # Because our dataset is tiny (~16 missions), we use Leave-One-Out
    # Cross-Validation (LOOCV) to get an unbiased estimate of its accuracy.
    print("Evaluating model with Leave-One-Out Cross-Validation...")
    loo = LeaveOneOut()
    scores = cross_val_score(model, X_train, y_train, cv=loo, scoring='accuracy')
    print(f"Estimated Model Accuracy: {scores.mean() * 100:.2f}%")

    print("Model training complete.")
    return model, existing_features


def prepare_global_prediction_set(df_mola, df_gcm):
    print("\n--- Step 4: Preparing Global Prediction Set (All Mars) ---")

    # Start with the full GCM map (64,800+ rows)
    # Merge the sparse MOLA map onto it.
    global_map_df = pd.merge(
        df_gcm,
        df_mola,
        on=['latitude', 'longitude'],
        how='left'
    )

    # Just like in training, fill all non-optimal terrain with rank 3
    global_map_df['rank'] = global_map_df['rank'].fillna(3)

    # Fill any other missing data just in case
    global_map_df = global_map_df.fillna(0)

    print(f"Global prediction set prepared with {len(global_map_df)} grid cells.")
    return global_map_df


def predict_and_save_map(model, features_list, global_map_df, output_file):
    if model is None:
        print("!! ERROR: Model was not trained. Aborting prediction.")
        return

    print("\n--- Step 5: Predicting Suitability for all of Mars ---")


    X_global = global_map_df[features_list]

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


def main():
    missions_file = "../data/gcm/datasets/past_missions.csv"
    gcm_file = "../data/gcm/datasets/gcm_yearly_stats.csv"
    mola_file = "../data/mola/mars_landing_sites_1deg.csv"

    final_prediction_map_file = "../output/final_suitability_map.csv"

    prediction_dataset_file = "../data/gcm/every_month/out_grid1x1deg_0h_0sollon.csv"


    df_missions, df_mola, df_gcm = load_all_datasets(missions_file, mola_file, gcm_file)
    if df_missions is None:
        print("Aborting run due to file loading error.")
        return  # Stop if loading failed


    training_df = create_training_data(df_missions, df_mola, df_gcm)

    # 3. Train the model on the training dataset
    model, features_list = train_model(training_df)

    if model is None:
        print("Aborting run due to model training error.")
        return

    # 4. Prepare the full planet dataset for prediction
    prediction_df = load_dataset_from_file(prediction_dataset_file)
    global_map_df = prepare_global_prediction_set(df_mola, prediction_df)

    # 5. Use the trained model to predict on the *entire planet*
    predict_and_save_map(model, features_list, global_map_df, final_prediction_map_file)


if __name__ == "__main__":
    main()