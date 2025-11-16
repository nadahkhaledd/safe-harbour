import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score
import gcm.gcm_preprocessing as gcm

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"!! ERROR: File not found: {file_path}")
        return None

def load_all_datasets(missions_file, mola_file, gcm_file):
    df_missions = load_clean_dataset(missions_file)
    df_mola = load_clean_dataset(mola_file)
    df_gcm = load_clean_dataset(gcm_file)
    if df_missions is None or df_mola is None or df_gcm is None:
        return None, None, None

    # Standardize MOLA columns
    if 'lat' in df_mola.columns and 'lon' in df_mola.columns:
        df_mola.rename(columns={'lat':'latitude', 'lon':'longitude'}, inplace=True)

    return df_missions, df_mola, df_gcm

# ============================================================
# 2. CREATE TRAINING DATASET
# ============================================================

def create_training_data(df_missions, df_mola, df_gcm):
    # Round to nearest grid for merge
    df_missions['grid_lat'] = df_missions['latitude'].round()
    df_missions['grid_lon'] = df_missions['longitude'].round()

    # Merge with GCM
    training_df = pd.merge(df_missions, df_gcm,
                           left_on=['grid_lat','grid_lon'],
                           right_on=['latitude','longitude'],
                           how='left')

    # Merge with MOLA
    training_df = pd.merge(training_df, df_mola,
                           left_on=['grid_lat','grid_lon'],
                           right_on=['latitude','longitude'],
                           how='left',
                           suffixes=('', '_mola'))

    # Fill rank missing
    training_df['rank'] = training_df['rank'].fillna(3)

    # Compute windspeed features
    wind_cols = ['zonal_wind_min','zonal_wind_max','zonal_wind_mean',
                 'meridional_wind_min','meridional_wind_max','meridional_wind_mean']

    if all(col in training_df.columns for col in wind_cols):
        training_df['windspeed_min'] = np.sqrt(training_df['zonal_wind_min']**2 + training_df['meridional_wind_min']**2)
        training_df['windspeed_max'] = np.sqrt(training_df['zonal_wind_max']**2 + training_df['meridional_wind_max']**2)
        training_df['windspeed_mean'] = np.sqrt(training_df['zonal_wind_mean']**2 + training_df['meridional_wind_mean']**2)
    else:
        training_df['windspeed_min'] = training_df['windspeed_max'] = training_df['windspeed_mean'] = 0

    # Fill missing values with -1 to differentiate from real zeros
    training_df.fillna(-1, inplace=True)

    return training_df

# ============================================================
# 3. FEATURE ANALYSIS
# ============================================================

def generate_correlation_matrix(df, features):
    plt.figure(figsize=(12,10))
    sns.heatmap(df[features].corr(), cmap='coolwarm', annot=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("../output/correlation_matrix.png")
    print("Saved correlation_matrix.png")

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'feature':features, 'importance':importances})
    feat_imp_df.sort_values('importance', ascending=True, inplace=True)

    plt.figure(figsize=(10,8))
    plt.barh(feat_imp_df['feature'], feat_imp_df['importance'])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("../output/feature_importance.png")
    print("Saved feature_importance.png")

# ============================================================
# 4. TRAIN MODEL WITH LOO AND CLASS_WEIGHT
# ============================================================

def train_model(training_df):
    TARGET = 'success_landing'
    FEATURES = ['extvar_29_min','extvar_29_max','extvar_29_mean',
                'extvar_44_min','extvar_44_max','extvar_44_mean',
                'temperature_min','temperature_max','temperature_mean',
                'windspeed_min','windspeed_max','windspeed_mean','rank']
    FEATURES = [f for f in FEATURES if f in training_df.columns]

    X = training_df[FEATURES]
    y = training_df[TARGET]

    # Fill NA with -1 for model
    X.fillna(-1, inplace=True)

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

    # Leave-One-Out CV
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo)
    print(f"Leave-One-Out CV accuracy: {scores.mean():.3f}")

    # Train on full dataset
    model.fit(X, y)

    # Plot feature importance
    plot_feature_importance(model, FEATURES)

    # Correlation matrix
    generate_correlation_matrix(training_df, FEATURES)

    return model, FEATURES

# ============================================================
# 5. GLOBAL PREDICTION
# ============================================================

def prepare_global_prediction_set(df_mola, gcm_prediction_df):
    df = gcm_prediction_df.merge(df_mola, on=['latitude','longitude'], how='left')
    df['rank'] = df['rank'].fillna(3)

    if 'zonal_wind_mean' in df.columns and 'meridional_wind_mean' in df.columns:
        df['windspeed_min'] = np.sqrt(df['zonal_wind_min']**2 + df['meridional_wind_min']**2)
        df['windspeed_max'] = np.sqrt(df['zonal_wind_max']**2 + df['meridional_wind_max']**2)
        df['windspeed_mean'] = np.sqrt(df['zonal_wind_mean']**2 + df['meridional_wind_mean']**2)
    else:
        df['windspeed_min'] = df['windspeed_max'] = df['windspeed_mean'] = 0

    df.fillna(-1, inplace=True)
    return df

def predict_global_map(model, features, df_global, output_file):
    X_global = df_global[features]
    df_global['suitability_score'] = model.predict_proba(X_global)[:,1]
    df_global[['latitude','longitude','suitability_score']].to_csv(output_file, index=False)
    print("Saved:", output_file)

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    missions_file = "../data/past_missions.csv"
    gcm_stats_file = "../data/gcm/datasets/gcm_yearly_stats.csv"
    mola_file = "../data/mola/mars_landing_sites_1deg.csv"
    single_gcm_file = "../data/gcm/every_month/out_grid1x1deg_0h_0sollon.csv"
    headers_file = "../data/gcm/gcm_headers.txt"

    output_pred_file = "../output/final_suitability_map_full2.csv"

    # 1. Load datasets
    df_missions, df_mola, df_gcm = load_all_datasets(missions_file, mola_file, gcm_stats_file)
    if df_missions is None: return

    # 2. Create training dataset
    training_df = create_training_data(df_missions, df_mola, df_gcm)

    # 3. Train model
    model, features = train_model(training_df)

    # 4. Prepare global prediction dataset
    prediction_df = gcm.prepare_single_gcm_file_for_prediction(headers_file, single_gcm_file,
                                                               "../data/gcm/datasets/gcm_prediction_input_prepared.csv")
    global_df = prepare_global_prediction_set(df_mola, prediction_df)

    # 5. Predict global suitability
    predict_global_map(model, features, global_df, output_pred_file)

if __name__ == "__main__":
    main()
