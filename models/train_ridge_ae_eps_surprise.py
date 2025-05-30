import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler # AE factors might not need scaling, but let's keep for consistency first
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import csv
from datetime import datetime

# --- Configuration ---
MODEL_NAME = "Ridge_AE_eps_surprise" # Changed model name
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# Input feature files: AE factors + original files for target generation
TRAIN_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VAL_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

TRAIN_AE_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'train_ae_factors.parquet')
VAL_AE_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'validation_ae_factors.parquet')
TEST_AE_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'test_ae_factors.parquet')


SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
MODEL_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_model.joblib')
SCALER_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_scaler.joblib') # May remove if not scaling AE
PREDICTIONS_SAVE_PATH = os.path.join(PREDICTIONS_DIR, f'predictions_{MODEL_NAME.lower()}.parquet')

# Target definition (same as before)
NEW_TARGET_COL = 'eps_surprise_scaled_by_price'
EPS_ACTUAL_COL = 'eps_actual' 
EPS_MEANEST_COL = 'eps_meanest' 
PRICE_COL_FOR_SCALING = 'prc'   

DATE_COLUMN = 'date'
ID_COLUMN = 'permno'

# --- Define CSV Logging Function (same as before) --- 
RESULTS_DIR = 'results'
CSV_FILE = os.path.join(RESULTS_DIR, 'performance_summary.csv')

def log_metrics_to_csv(model_name, metrics_dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.isfile(CSV_FILE)
    # Include RMSE if present
    metric_keys = sorted([k for k in metrics_dict.keys() if k not in ['timestamp', 'model_name']])
    if any('rmse' in k for k in metrics_dict.keys()): # check if rmse is there
        if f'val_rmse_{NEW_TARGET_COL}' not in metric_keys : metric_keys.append(f'val_rmse_{NEW_TARGET_COL}')
        if f'test_rmse_{NEW_TARGET_COL}' not in metric_keys : metric_keys.append(f'test_rmse_{NEW_TARGET_COL}')
        metric_keys = sorted([k for k in metric_keys if metrics_dict.get(k) is not None])


    fieldnames = ['timestamp', 'model_name'] + metric_keys
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writeheader()
        
        row_to_write = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'model_name': model_name}
        for key in metric_keys:
            row_to_write[key] = metrics_dict.get(key)
        writer.writerow(row_to_write)

# --- Main Function ---
def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load original data for target creation and IDs/Dates
    print("Loading original data for target and IDs...")
    dfs_original = {}
    for split_name, path in zip(['train', 'val', 'test'], [TRAIN_FEATURES_PATH, VAL_FEATURES_PATH, TEST_FEATURES_PATH]):
        try:
            dfs_original[split_name] = pd.read_parquet(path, columns=[ID_COLUMN, DATE_COLUMN, EPS_ACTUAL_COL, EPS_MEANEST_COL, PRICE_COL_FOR_SCALING])
            print(f"Original {split_name} df shape (for target): {dfs_original[split_name].shape}")
        except Exception as e:
            print(f"Error loading original {split_name} data from {path}: {e}"); return

    # Load AE factors
    print("Loading AE factor data...")
    dfs_ae = {}
    for split_name, path in zip(['train', 'val', 'test'], [TRAIN_AE_FACTORS_PATH, VAL_AE_FACTORS_PATH, TEST_AE_FACTORS_PATH]):
        try:
            dfs_ae[split_name] = pd.read_parquet(path)
            print(f"AE {split_name} df shape: {dfs_ae[split_name].shape}")
        except Exception as e:
            print(f"Error loading AE {split_name} data from {path}: {e}"); return
            
    # --- Target Construction & Merging with AE Factors ---
    print(f"\nConstructing target: {NEW_TARGET_COL} and merging with AE factors")
    
    X_dfs = {}
    y_dfs = {}
    full_dfs_for_pred_saving = {}


    for split_name in ['train', 'val', 'test']:
        print(f"Processing {split_name} data...")
        original_df = dfs_original[split_name]
        ae_df = dfs_ae[split_name]

        # Construct target on original_df
        required_cols_for_target = [EPS_ACTUAL_COL, EPS_MEANEST_COL, PRICE_COL_FOR_SCALING]
        missing_cols = [col for col in required_cols_for_target if col not in original_df.columns]
        if missing_cols:
            print(f"Error: Missing cols for target in original_{split_name}_df: {missing_cols}"); return

        original_df['raw_eps_surprise'] = original_df[EPS_ACTUAL_COL] - original_df[EPS_MEANEST_COL]
        price_floor = 0.01
        original_df[NEW_TARGET_COL] = original_df['raw_eps_surprise'] / original_df[PRICE_COL_FOR_SCALING].clip(lower=price_floor)
        original_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Merge AE factors with original_df (containing target, ID, Date)
        # Important: AE factors parquet files should have ID_COLUMN and DATE_COLUMN for correct merging
        merged_df = pd.merge(original_df[[ID_COLUMN, DATE_COLUMN, NEW_TARGET_COL]], ae_df, on=[ID_COLUMN, DATE_COLUMN], how='inner')
        
        original_rows_merged = len(merged_df)
        merged_df.dropna(subset=[NEW_TARGET_COL], inplace=True) # Drop rows where target is NaN
        print(f"  {split_name} merged_df: {original_rows_merged - len(merged_df)} rows dropped due to NaN in {NEW_TARGET_COL}.")
        print(f"  {split_name} merged_df shape after target construction, merge & NaN drop: {merged_df.shape}")

        if merged_df.empty: 
            print(f"Warning: {split_name} DataFrame is empty after processing. This split will be skipped."); 
            X_dfs[split_name], y_dfs[split_name], full_dfs_for_pred_saving[split_name] = None, None, None
            if split_name in ['train', 'val']: # Critical if train or val is empty
                 print(f"CRITICAL ERROR: {split_name} is empty. Cannot proceed.")
                 return
            continue 

        print(f"  Descriptive stats for {NEW_TARGET_COL} in {split_name}_merged_df:")
        print(merged_df[NEW_TARGET_COL].describe())

        predictor_cols = [col for col in merged_df.columns if col.startswith('ae_factor_')]
        if not predictor_cols:
            print(f"Error: No AE factor columns found in {split_name}_merged_df."); return
            
        X_dfs[split_name] = merged_df[predictor_cols]
        y_dfs[split_name] = merged_df[NEW_TARGET_COL]
        full_dfs_for_pred_saving[split_name] = merged_df # Keep for saving predictions with ID/Date

    X_train, y_train = X_dfs['train'], y_dfs['train']
    X_val, y_val = X_dfs['val'], y_dfs['val']
    X_test, y_test = X_dfs['test'], y_dfs['test']
    
    test_df_for_saving = full_dfs_for_pred_saving['test']


    if X_train is None or y_train is None or X_val is None or y_val is None:
        raise ValueError("Train or Validation DataFrame is empty after processing. Cannot proceed.")

    # --- Preprocessing: Handle NaNs/Infs in predictors (AE factors should be clean) and Scale ---
    # AE factors from script are already scaled and imputed.
    # However, if we want to apply RobustScaler again (e.g. if they weren't scaled from 0-1 or robustly)
    # For now, assume AE factors are ready to use, or apply light scaling if needed.
    # Let's assume AE factors might not be on the same scale, so RobustScaler is still a good idea.
    print("Preprocessing AE factors (scaling)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    # Save the scaler used for AE factors
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler for AE factors saved to {SCALER_SAVE_PATH}")


    # --- Train Ridge Regression Model ---
    print("Training Ridge regression model on AE factors...")
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)

    # --- Evaluate Model ---
    print("Evaluating model...")
    y_pred_val = ridge_model.predict(X_val_scaled)
    val_r2 = r2_score(y_val, y_pred_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_rmse = np.sqrt(val_mse)
    print(f"Validation R-squared for {NEW_TARGET_COL} (AE): {val_r2:.6f}")
    print(f"Validation MSE for {NEW_TARGET_COL} (AE): {val_mse:.6f}")
    print(f"Validation RMSE for {NEW_TARGET_COL} (AE): {val_rmse:.6f}")

    test_r2, test_mse, test_rmse, y_pred_test = np.nan, np.nan, np.nan, None
    if X_test_scaled is not None and y_test is not None and not y_test.empty:
        y_pred_test = ridge_model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        print(f"Test R-squared for {NEW_TARGET_COL} (AE): {test_r2:.6f}")
        print(f"Test MSE for {NEW_TARGET_COL} (AE): {test_mse:.6f}")
        print(f"Test RMSE for {NEW_TARGET_COL} (AE): {test_rmse:.6f}")
    else:
        print("Skipping test set evaluation.")

    # --- Save Model and Predictions ---
    print("Saving model...")
    joblib.dump(ridge_model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    if y_pred_test is not None and test_df_for_saving is not None and not test_df_for_saving.empty:
        predictions_df = pd.DataFrame({
            ID_COLUMN: test_df_for_saving[ID_COLUMN],
            DATE_COLUMN: test_df_for_saving[DATE_COLUMN],
            f'predicted_{NEW_TARGET_COL}': y_pred_test,
            f'actual_{NEW_TARGET_COL}': y_test # y_test is already aligned with X_test from merged_df
        })
        predictions_df.to_parquet(PREDICTIONS_SAVE_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_SAVE_PATH}")
    else:
        print("Skipping saving of test predictions.")

    # --- Log Metrics ---
    metrics_to_log = {
        f'val_r2_{NEW_TARGET_COL}': val_r2, f'val_mse_{NEW_TARGET_COL}': val_mse, f'val_rmse_{NEW_TARGET_COL}': val_rmse,
    }
    if not np.isnan(test_r2): metrics_to_log[f'test_r2_{NEW_TARGET_COL}'] = test_r2
    if not np.isnan(test_mse): metrics_to_log[f'test_mse_{NEW_TARGET_COL}'] = test_mse
    if not np.isnan(test_rmse): metrics_to_log[f'test_rmse_{NEW_TARGET_COL}'] = test_rmse
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")

    # --- Feature Importance/Coefficients (for AE factors) ---
    print("\n--- Ridge Model Coefficients for AE Factors ---")
    if hasattr(ridge_model, 'coef_'):
        ae_predictor_cols = [f'ae_factor_{i}' for i in range(X_train_scaled.shape[1])] # Generate AE factor names
        coefficients = pd.Series(ridge_model.coef_, index=ae_predictor_cols)
        sorted_coefficients = coefficients.abs().sort_values(ascending=False)
        print("Top AE factors by absolute coefficient magnitude (max 32):")
        print(sorted_coefficients.head(32))
    else:
        print("Could not retrieve coefficients from the Ridge model.")
    print(f"--- {MODEL_NAME} Model Script Complete ---")

if __name__ == '__main__':
    np.random.seed(42)
    main() 