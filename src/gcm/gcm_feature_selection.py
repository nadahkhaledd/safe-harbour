import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def load_column_headers(headers_file_path):
    try:
        with open(headers_file_path, 'r') as f:
            header_line = f.readline().strip()
            return header_line.split(',')
    except FileNotFoundError:
        print(f"!! ERROR: Header file not found at: {headers_file_path}")
        return None


def create_full_feature_dataset(headers_file, gcm_folder, missions_file):
    column_names = load_column_headers(headers_file)
    if column_names is None: return None


    try:
        df_missions = pd.read_csv(missions_file)
        df_missions['grid_lat'] = np.round(df_missions['latitude'])
        df_missions['grid_lon'] = np.round(df_missions['longitude'])
    except FileNotFoundError:
        print(f"!! ERROR: Missions file not found: {missions_file}")
        return None

    # 3. Load ALL GCM files
    csv_files = glob.glob(os.path.join(gcm_folder, "*.csv"))
    if not csv_files:
        print(f"!! ERROR: No GCM files found in: {gcm_folder}")
        return None

    print(f"Loading {len(csv_files)} GCM files...")
    all_gcm_data = []
    for f in csv_files:
        df = pd.read_csv(f, sep=';', names=column_names, skiprows=[0])
        all_gcm_data.append(df)

    full_gcm = pd.concat(all_gcm_data)
    full_gcm = full_gcm.apply(pd.to_numeric, errors='coerce')

    print("Calculating yearly averages...")
    gcm_means = full_gcm.groupby(['latitude', 'longitude']).mean().reset_index()

    # 4. Merge
    analysis_df = pd.merge(
        df_missions,
        gcm_means,
        left_on=['grid_lat', 'grid_lon'],
        right_on=['latitude', 'longitude'],
        how='left'
    )

    return analysis_df


def analyze_features(df):
    print("\n--- Feature Analysis ---")

    # 1. Filter out non-climate columns
    ignore_cols = [
        'mission', 'country', 'year', 'notes', 'success_landing',
        'latitude', 'longitude', 'grid_lat', 'grid_lon',
        'latitude_x', 'latitude_y', 'longitude_x', 'longitude_y'
    ]

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols]

    # --- Method A: Correlation ---
    corr_matrix = df[feature_cols + ['success_landing']].corr()
    target_corr = corr_matrix['success_landing'].drop('success_landing')
    top_corr = target_corr.abs().sort_values(ascending=False).head(5)
    print(top_corr)

    print("\n[Method B] Random Forest Importance (Complex Relationships):")


    X = df[feature_cols].fillna(0)
    y = df['success_landing']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_importance = importances.sort_values(ascending=False).head(5)
    print(top_importance)

    # --- Visualization: Feature Importance ---
    plt.figure(figsize=(10, 6))
    top_importance.plot(kind='barh', color='skyblue')
    plt.title("Top 5 Most Important Climate Features (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("../../output/gcm/feature_importance_analysis.png")
    print("\nSaved feature importance plot to '../output/feature_importance_analysis.png'")


def main():
    headers_file = "../../data/gcm/gcm_headers.txt"
    gcm_folder = "../../data/gcm/every_month"
    missions_file = "../../data/past_missions.csv"

    full_data = create_full_feature_dataset(headers_file, gcm_folder, missions_file)

    if full_data is not None:
        analyze_features(full_data)


if __name__ == "__main__":
    main()