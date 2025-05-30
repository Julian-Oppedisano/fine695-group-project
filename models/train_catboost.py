import pandas as pd
import numpy as np
import catboost as cb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
import os
import csv
from datetime import datetime

# --- Configuration ---
MODEL_NAME = "CatBoost_IPCA_Factors"
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# Paths for original imputed data (to get target and identifiers)
TRAIN_ORIG_IMPUTED_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VAL_ORIG_IMPUTED_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_ORIG_IMPUTED_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

# Paths for IPCA factors
TRAIN_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'train_ipca_factors.parquet')
VAL_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'validation_ipca_factors.parquet')
TEST_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'test_ipca_factors.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_model.cbm')
PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, f'predictions_{MODEL_NAME.lower()}.parquet')

ORIG_TARGET_COLUMN = 'stock_exret'
NEW_TARGET_COLUMN = 'target_quintile'
DATE_COLUMN = 'date'
ID_COLUMN = 'permno'

# --- Define CSV Logging Function ---
RESULTS_DIR = 'results'
CSV_FILE = os.path.join(RESULTS_DIR, 'performance_summary.csv')

def log_metrics_to_csv(model_name, metrics_dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.isfile(CSV_FILE)
    
    metric_keys = sorted([k for k in metrics_dict.keys() if k not in ['timestamp', 'model_name']])
    fieldnames = ['timestamp', 'model_name'] + metric_keys
    
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writeheader()
            
        log_entry = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'model_name': model_name}
        log_entry.update(metrics_dict)
        writer.writerow(log_entry)
# --- End CSV Logging Function ---

def main():
    print(f"--- Training {MODEL_NAME} Model using IPCA Factors ---")

    print("Loading original imputed data for target generation and identifiers...")
    train_df_orig = pd.read_parquet(TRAIN_ORIG_IMPUTED_PATH)
    val_df_orig = pd.read_parquet(VAL_ORIG_IMPUTED_PATH)
    test_df_orig = pd.read_parquet(TEST_ORIG_IMPUTED_PATH)

    print("Loading IPCA factors...")
    train_factors_df = pd.read_parquet(TRAIN_FACTORS_PATH)
    val_factors_df = pd.read_parquet(VAL_FACTORS_PATH)
    test_factors_df = pd.read_parquet(TEST_FACTORS_PATH)

    # Define IPCA factor column names
    N_FACTORS_EXPECTED = 32
    predictor_cols = [f'ipca_factor_{i}' for i in range(N_FACTORS_EXPECTED)]

    # --- Create Quintile Target from original data ---
    print("\n--- Generating Quintile Target ---")
    temp_dfs_for_quintiles = []
    for df_part_name, df_part_orig in [('train', train_df_orig), ('validation', val_df_orig), ('test', test_df_orig)]:
        print(f"Processing {df_part_name} for quintiles...")
        current_df = df_part_orig[[ID_COLUMN, DATE_COLUMN, ORIG_TARGET_COLUMN]].copy()
        current_df[DATE_COLUMN] = pd.to_datetime(current_df[DATE_COLUMN].astype(str))
        current_df['year_month'] = current_df[DATE_COLUMN].dt.to_period('M')
        current_df[NEW_TARGET_COLUMN] = current_df.groupby('year_month')[ORIG_TARGET_COLUMN] \
                                            .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
        print(f"{df_part_name} {NEW_TARGET_COLUMN} distribution:\n{current_df[NEW_TARGET_COLUMN].value_counts(normalize=True, dropna=False).sort_index()}")
        if current_df[NEW_TARGET_COLUMN].isnull().any():
            print(f"Warning: NaNs found in {NEW_TARGET_COLUMN} for {df_part_name} after pd.qcut. Filling with mode.")
            mode_val = current_df[NEW_TARGET_COLUMN].mode()[0]
            current_df[NEW_TARGET_COLUMN].fillna(mode_val, inplace=True)
        current_df[NEW_TARGET_COLUMN] = current_df[NEW_TARGET_COLUMN].astype(int)
        temp_dfs_for_quintiles.append(current_df[[ID_COLUMN, DATE_COLUMN, NEW_TARGET_COLUMN]])
    
    train_targets_df, val_targets_df, test_targets_df = temp_dfs_for_quintiles
    print("--- Quintile Target Generation Complete ---\n")

    # Merge IPCA factors with corresponding target DataFrames
    print("Merging IPCA factors with target data...")
    train_df = pd.merge(train_factors_df, train_targets_df, on=[ID_COLUMN, DATE_COLUMN], how='inner')
    val_df = pd.merge(val_factors_df, val_targets_df, on=[ID_COLUMN, DATE_COLUMN], how='inner')
    test_df = pd.merge(test_factors_df, test_targets_df, on=[ID_COLUMN, DATE_COLUMN], how='inner')

    print(f"Shape after merging train: {train_df.shape} (Factors: {train_factors_df.shape}, Targets: {train_targets_df.shape})")
    print(f"Shape after merging val: {val_df.shape} (Factors: {val_factors_df.shape}, Targets: {val_targets_df.shape})")
    print(f"Shape after merging test: {test_df.shape} (Factors: {test_factors_df.shape}, Targets: {test_targets_df.shape})")

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Dataframe is empty after merging IPCA factors with targets. Check identifiers or merge logic.")
    if train_df[NEW_TARGET_COLUMN].isnull().any() or val_df[NEW_TARGET_COLUMN].isnull().any() or test_df[NEW_TARGET_COLUMN].isnull().any():
        raise ValueError("NaNs still present in target column after merge and quintile generation. This should not happen.")

    X_train = train_df[predictor_cols]
    y_train = train_df[NEW_TARGET_COLUMN]
    X_val = val_df[predictor_cols]
    y_val = val_df[NEW_TARGET_COLUMN]
    X_test = test_df[predictor_cols]
    y_test = test_df[NEW_TARGET_COLUMN]
    
    print(f"\n--- y_train (Quintiles from IPCA Factors) Diagnostics ---")
    print(y_train.describe())
    print(f"Number of unique values in y_train: {y_train.nunique()}")
    print(y_train.value_counts(normalize=True).sort_index())
    print("--- End y_train Diagnostics ---\n")
    
    categorical_features_names = [] 

    print("Initializing and training CatBoost model with IPCA Factors...")
    model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100,
        cat_features=categorical_features_names,
        early_stopping_rounds=50
    )

    eval_pool = cb.Pool(X_val, y_val, cat_features=categorical_features_names)
    
    model.fit(X_train, y_train, eval_set=eval_pool)

    print("Training finished.")
    print("Making predictions on the test set using IPCA Factors...")
    predictions = model.predict(X_test).flatten()

    print("\n--- Test Set Predictions (Quintiles from IPCA) Diagnostics ---")
    predictions_series = pd.Series(predictions)
    print(predictions_series.describe())
    print(f"Number of unique predictions: {predictions_series.nunique()}")
    print("Value counts of predicted quintiles:\n{predictions_series.value_counts(normalize=True).sort_index()}")
    print("--- End Test Set Predictions (Quintiles from IPCA) Diagnostics ---\n")

    oos_accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    print(f"Out-of-Sample Accuracy (IPCA Factors): {oos_accuracy:.6f}")
    print(f"Out-of-Sample F1 Macro (IPCA Factors): {f1_macro:.6f}")
    print(f"Out-of-Sample F1 Weighted (IPCA Factors): {f1_weighted:.6f}")

    report = classification_report(y_test, predictions, zero_division=0)
    print("\nClassification Report (IPCA Factors):\n", report)
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix (IPCA Factors):\n", cm)

    print("Saving model...")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    predictions_df = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        DATE_COLUMN: test_df[DATE_COLUMN],
        'predicted_quintile_ipca': predictions
    })
    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")

    metrics_to_log = {
        'accuracy': oos_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")

    print(f"--- {MODEL_NAME} Model Script Complete ---")

if __name__ == '__main__':
    main() 