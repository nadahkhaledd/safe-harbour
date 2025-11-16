import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --- Import your existing pipeline helpers ---
import new_predict_traintest_data as pipeline

# ============================================================
# 1. BUILD TRAINING DATASET (SAME AS MAIN PIPELINE)
# ============================================================

# Same paths as in new_predict_traintest_data.main()
missions_file   = "./data/past_missions.csv"
gcm_stats_file  = "./data/gcm/datasets/gcm_yearly_stats.csv"
mola_file       = "./data/mola/mars_landing_sites_1deg.csv"

df_missions, df_mola, df_gcm = pipeline.load_all_datasets(
    missions_file,
    mola_file,
    gcm_stats_file
)

if df_missions is None or df_mola is None or df_gcm is None:
    raise RuntimeError("Could not load one or more input datasets.")

training_df = pipeline.create_training_data(df_missions, df_mola, df_gcm)

# Use the same feature set as your main model
FEATURES = [
    'extvar_29_min','extvar_29_max','extvar_29_mean',
    'extvar_44_min','extvar_44_max','extvar_44_mean',
    'temperature_min','temperature_max','temperature_mean',
    'windspeed_min','windspeed_max','windspeed_mean',
    'rank'
]
FEATURES = [f for f in FEATURES if f in training_df.columns]

X = training_df[FEATURES]
y = training_df['success_landing']

# Just in case: same missing-value handling as training
X = X.fillna(-1)

# ============================================================
# 2. STRATIFIED K-FOLD CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

print("=== Cross-Validation Results ===")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_proba)

    fold_metrics.append((acc, prec, rec, f1, roc))

    print(
        f"Fold {fold}: "
        f"Accuracy={acc:.3f}, Precision={prec:.3f}, "
        f"Recall={rec:.3f}, F1={f1:.3f}, ROC AUC={roc:.3f}"
    )

metrics_array = np.array(fold_metrics)
avg_acc, avg_prec, avg_rec, avg_f1, avg_roc = metrics_array.mean(axis=0)

print("\n=== Average Metrics Across 5 Folds ===")
print(
    f"Accuracy={avg_acc:.3f}, Precision={avg_prec:.3f}, "
    f"Recall={avg_rec:.3f}, F1={avg_f1:.3f}, ROC AUC={avg_roc:.3f}"
)