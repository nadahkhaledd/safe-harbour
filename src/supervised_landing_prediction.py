import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import gcm.gcm_preprocessing as gcm  # Ensure this import matches your file name



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
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"!! ERROR: File not found at: {file_path}")
        return None


def load_single_gcm_file(file_path, column_names):
    print(f"Loading single GCM file from {file_path}...")
    try:
        df = pd.read_csv(file_path, sep=';', names=column_names, skiprows=[0])
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    except FileNotFoundError:
        print(f"!! ERROR: File not found at: {file_path}")
        return None


def load_all_datasets(missions_file, mola_file, gcm_file):
    print("\n--- Step 1: Loading Datasets ---")
    df_missions = load_clean_dataset(missions_file)
    df_mola = load_clean_dataset(mola_file)
    df_gcm = load_clean_dataset(gcm_file)

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

    print("Matching exact mission coordinates to GCM/MOLA 1x1 grid...")
    df_missions['grid_lat'] = np.round(df_missions['latitude'])
    df_missions['grid_lon'] = np.round(df_missions['longitude'])

    print("Merging mission data with GCM climate data...")
    training_df = pd.merge(
        df_missions, df_gcm,
        left_on=['grid_lat', 'grid_lon'], right_on=['latitude', 'longitude'],
        how='left', suffixes=('_mission', '_gcm')
    )

    print("Merging result with MOLA terrain data...")
    training_df = pd.merge(
        training_df, df_mola,
        left_on=['grid_lat', 'grid_lon'], right_on=['latitude', 'longitude'],
        how='left', suffixes=('_gcm', '_mola')
    )

    print("Handling missing data...")
    training_df['rank'] = training_df['rank'].fillna(3)
    training_df = training_df.fillna(0)

    print("Training data merged successfully.")
    return training_df


def train_and_evaluate_model(training_df):
    print("\n--- Step 3: Training & Evaluating Model ---")

    GCM_FEATURES = [
        'extvar_29_min', 'extvar_29_max', 'extvar_29_mean',
        'extvar_44_min', 'extvar_44_max', 'extvar_44_mean',
        'extvar_28_min', 'extvar_28_max', 'extvar_28_mean',
        'atm_pressure_min', 'atm_pressure_max', 'atm_pressure_mean',
        'temperature_min', 'temperature_max', 'temperature_mean',
        'windspeed_min', 'windspeed_max', 'windspeed_mean'
    ]
    MOLA_FEATURES = ['rank']
    FEATURES = GCM_FEATURES + MOLA_FEATURES
    TARGET = 'success_landing'

    existing_features = [col for col in FEATURES if col in training_df.columns]
    if not existing_features:
        print("!! ERROR: No features found. Check column names.")
        return None, []

    X = training_df[existing_features]
    y = training_df[TARGET]

    model = RandomForestClassifier(random_state=42, n_estimators=100)

    # 2. Leave-One-Out Cross-Validation (LOOCV)
    print("Running Leave-One-Out Cross-Validation...")
    loo = LeaveOneOut()

    # Get probability predictions (needed for ROC/AUC)
    y_probs = cross_val_predict(model, X, y, cv=loo, method='predict_proba')[:, 1]

    # Convert probabilities to binary predictions (0 or 1)
    y_pred = (y_probs >= 0.5).astype(int)

    # 3. Calculate Metrics
    print("\n--- Model Performance Metrics (LOOCV) ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    print(f"Accuracy: {accuracy_score(y, y_pred):.2%}")

    # 4. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("../output/roc_curve.png")
    print("Saved ROC curve to '../output/roc_curve.png'")
    plt.close()

    # 5. Feature Importance (Train on Full Data)
    print("\nRetraining model on full dataset for final export...")
    model.fit(X, y)

    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp_df['feature'], feat_imp_df['importance'])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Model Feature Importance")
    plt.tight_layout()
    plt.savefig("../output/feature_importance.png")
    print("Saved Feature Importance to '../output/feature_importance.png'")

    return model, existing_features


# --- 4. Prepare Prediction Set ---
def prepare_global_prediction_set(df_mola, df_gcm_prediction_file):
    print("\n--- Step 4: Preparing Global Prediction Set ---")
    global_map_df = pd.merge(
        df_gcm_prediction_file, df_mola,
        on=['latitude', 'longitude'], how='left'
    )
    global_map_df['rank'] = global_map_df['rank'].fillna(3)
    global_map_df = global_map_df.fillna(0)
    print(f"Global prediction set prepared with {len(global_map_df)} grid cells.")
    return global_map_df



def predict_and_save_map(model, features_list, global_map_df, output_file):
    if model is None: return

    print("\n--- Step 5: Predicting Suitability ---")

    rename_map = {
        'extvar_29': 'extvar_29_mean', 'extvar_44': 'extvar_44_mean',
        'temperature': 'temperature_mean', 'extvar_28': 'extvar_28_mean',
        'windspeed': 'windspeed_mean', 'atm pressure': 'atm_pressure_mean'
    }
    global_map_df_renamed = global_map_df.rename(columns=rename_map)

    # Impute min/max from mean
    for col in features_list:
        if col not in global_map_df_renamed.columns:
            base_col = col.replace('_min', '').replace('_max', '').replace('_mean', '')
            if '_mean' in col:
                pass
            else:  # We are looking for min/max
                # Construct the mean column name to copy from
                mean_col_name = col.replace('_min', '_mean').replace('_max', '_mean')
                if mean_col_name in global_map_df_renamed.columns:
                    global_map_df_renamed[col] = global_map_df_renamed[mean_col_name]

    # Prepare X_global
    X_global = global_map_df_renamed[features_list].fillna(0)

    print(f"Predicting suitability for {len(X_global)} grid cells...")
    probabilities = model.predict_proba(X_global)[:, 1]

    final_map = global_map_df[['latitude', 'longitude']].copy()
    final_map['suitability_score'] = probabilities

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_map.to_csv(output_file, index=False)

    print(f"\n--- GLOBAL PREDICTION COMPLETE ---")
    print(f"Final prediction map saved to: {output_file}")
    print("\n--- Top 10 Suitable Zones ---")
    print(final_map.sort_values(by='suitability_score', ascending=False).head(10))


# --- Main ---
def main():
    # Paths
    missions_file = "../data/past_missions.csv"
    gcm_yearly_file = "../data/gcm/datasets/gcm_yearly_stats.csv"
    mola_file = "../data/mola/mars_landing_sites_1deg.csv"

    headers_file = "../data/gcm/gcm_headers.txt"
    raw_prediction_input = "../data/gcm/every_month/out_grid1x1deg_0h_0sollon.csv"
    prepared_prediction_file = "../data/gcm/datasets/gcm_prediction_input_prepared.csv"

    final_map_file = "../output/final/final_suitability_map.csv"


    print("--- Ensuring Prediction Input Exists ---")
    if not os.path.exists(prepared_prediction_file):
        gcm.prepare_single_gcm_file_for_prediction(headers_file, raw_prediction_input, prepared_prediction_file)

    # 2. Load Data
    df_missions, df_mola, df_gcm = load_all_datasets(missions_file, mola_file, gcm_yearly_file)
    if df_missions is None: return

    # 3. Create Training Set & Train Model
    training_df = create_training_data(df_missions, df_mola, df_gcm)

    # Save training data for inspection
    os.makedirs("../output/final", exist_ok=True)
    training_df.to_csv("../output/training_dataset.csv", index=False)

    model, features = train_and_evaluate_model(training_df)
    if model is None: return

    # 4. Prepare Prediction Data
    df_prediction_input = load_clean_dataset(prepared_prediction_file)
    global_map_df = prepare_global_prediction_set(df_mola, df_prediction_input)

    # 5. Predict
    predict_and_save_map(model, features, global_map_df, final_map_file)


if __name__ == "__main__":
    main()