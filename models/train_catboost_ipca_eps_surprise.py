import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import csv
from datetime import datetime

# --- Configuration ---
MODEL_NAME = "CatBoost_IPCA_eps_surprise" # Changed model name
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# Input feature files: IPCA factors + original files for target generation
TRAIN_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VAL_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

TRAIN_IPCA_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'train_ipca_factors.parquet') # Changed path
VAL_IPCA_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'validation_ipca_factors.parquet') # Changed path
TEST_IPCA_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'test_ipca_factors.parquet') # Changed path

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
MODEL_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_model.cbm') 
PREDICTIONS_SAVE_PATH = os.path.join(PREDICTIONS_DIR, f'predictions_{MODEL_NAME.lower()}.parquet')

# Target definition (same as before)
NEW_TARGET_COL = 'eps_surprise_scaled_by_price'
EPS_ACTUAL_COL = 'eps_actual' 
EPS_MEANEST_COL = 'eps_meanest'
PRICE_COL_FOR_SCALING = 'prc'   

DATE_COLUMN = 'date'
ID_COLUMN = 'permno'

# --- CSV Logging Function (same as before) ---
RESULTS_DIR = 'results'
CSV_FILE = os.path.join(RESULTS_DIR, 'performance_summary.csv')

def log_metrics_to_csv(model_name, metrics_dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.isfile(CSV_FILE)
    metric_keys = sorted([k for k in metrics_dict.keys() if k not in ['timestamp', 'model_name']])
    has_rmse = any('rmse' in k for k in metrics_dict.keys())
    if has_rmse:
        for rmse_key_type in ['val', 'test']:
            full_rmse_key = f'{rmse_key_type}_rmse_{NEW_TARGET_COL}'
            if metrics_dict.get(full_rmse_key) is not None and full_rmse_key not in metric_keys:
                metric_keys.append(full_rmse_key)
        metric_keys = sorted(metric_keys) 
    
    fieldnames = ['timestamp', 'model_name'] + metric_keys
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writeheader()
        
        row_to_write = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'model_name': model_name}
        for key in fieldnames: 
            if key not in ('timestamp', 'model_name'):
                 row_to_write[key] = metrics_dict.get(key) 
        writer.writerow(row_to_write)
# --- End CSV Logging Function ---

# --- Main Function ---
def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading original data for target and IDs...")
    dfs_original = {}
    for split_name, path in zip(['train', 'val', 'test'], [TRAIN_FEATURES_PATH, VAL_FEATURES_PATH, TEST_FEATURES_PATH]):
        try:
            dfs_original[split_name] = pd.read_parquet(path, columns=[ID_COLUMN, DATE_COLUMN, EPS_ACTUAL_COL, EPS_MEANEST_COL, PRICE_COL_FOR_SCALING])
        except Exception as e:
            print(f"Error loading original {split_name} data from {path}: {e}"); return

    print("Loading IPCA factor data...") # Changed message
    dfs_ipca = {} # Changed variable name
    for split_name, path in zip(['train', 'val', 'test'], [TRAIN_IPCA_FACTORS_PATH, VAL_IPCA_FACTORS_PATH, TEST_IPCA_FACTORS_PATH]): # Changed paths
        try:
            dfs_ipca[split_name] = pd.read_parquet(path)
        except Exception as e:
            print(f"Error loading IPCA {split_name} data from {path}: {e}"); return
            
    X_dfs = {}
    y_dfs = {}
    full_dfs_for_pred_saving = {}

    for split_name in ['train', 'val', 'test']:
        print(f"Processing {split_name} data...")
        original_df = dfs_original[split_name]
        ipca_df = dfs_ipca[split_name] # Changed variable name

        original_df['raw_eps_surprise'] = original_df[EPS_ACTUAL_COL] - original_df[EPS_MEANEST_COL]
        price_floor = 0.01
        original_df[NEW_TARGET_COL] = original_df['raw_eps_surprise'] / original_df[PRICE_COL_FOR_SCALING].clip(lower=price_floor)
        original_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        merged_df = pd.merge(original_df[[ID_COLUMN, DATE_COLUMN, NEW_TARGET_COL]], ipca_df, on=[ID_COLUMN, DATE_COLUMN], how='inner') # Changed ipca_df
        
        original_rows_merged = len(merged_df)
        merged_df.dropna(subset=[NEW_TARGET_COL], inplace=True)
        print(f"  {split_name} merged_df: {original_rows_merged - len(merged_df)} rows dropped due to NaN in {NEW_TARGET_COL}.")
        print(f"  {split_name} merged_df shape after target construction, merge & NaN drop: {merged_df.shape}")

        if merged_df.empty: 
            print(f"Warning: {split_name} DataFrame is empty. This split will be skipped.");
            X_dfs[split_name], y_dfs[split_name], full_dfs_for_pred_saving[split_name] = None, None, None
            if split_name in ['train', 'val']: 
                 print(f"CRITICAL ERROR: {split_name} is empty. Cannot proceed."); return
            continue 

        print(f"  Descriptive stats for {NEW_TARGET_COL} in {split_name}_merged_df:")
        print(merged_df[NEW_TARGET_COL].describe())

        predictor_cols = [col for col in merged_df.columns if col.startswith('ipca_factor_')] # Changed prefix
        if not predictor_cols:
            print(f"Error: No IPCA factor columns found in {split_name}_merged_df."); return
            
        X_dfs[split_name] = merged_df[predictor_cols]
        y_dfs[split_name] = merged_df[NEW_TARGET_COL]
        full_dfs_for_pred_saving[split_name] = merged_df

    X_train, y_train = X_dfs['train'], y_dfs['train']
    X_val, y_val = X_dfs['val'], y_dfs['val']
    X_test, y_test = X_dfs['test'], y_dfs['test']
    test_df_for_saving = full_dfs_for_pred_saving['test']

    if X_train is None or y_train is None or X_val is None or y_val is None:
        raise ValueError("Train or Validation DataFrame is empty. Cannot proceed.")

    print("Training CatBoostRegressor model on IPCA factors...") # Changed message
    cb_model = CatBoostRegressor(
        iterations=5000, learning_rate=0.01, depth=5, 
        loss_function='RMSE', eval_metric='RMSE', 
        random_seed=42, verbose=200, early_stopping_rounds=100 
    )

    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    print("Evaluating model...")
    y_pred_val = cb_model.predict(X_val)
    val_r2 = r2_score(y_val, y_pred_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_rmse = np.sqrt(val_mse)
    print(f"Validation R-squared for {NEW_TARGET_COL} (IPCA): {val_r2:.6f}") # Changed label
    print(f"Validation MSE for {NEW_TARGET_COL} (IPCA): {val_mse:.6f}") # Changed label
    print(f"Validation RMSE for {NEW_TARGET_COL} (IPCA): {val_rmse:.6f}") # Changed label

    test_r2, test_mse, test_rmse, y_pred_test = np.nan, np.nan, np.nan, None
    if X_test is not None and y_test is not None and not y_test.empty:
        y_pred_test = cb_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        print(f"Test R-squared for {NEW_TARGET_COL} (IPCA): {test_r2:.6f}") # Changed label
        print(f"Test MSE for {NEW_TARGET_COL} (IPCA): {test_mse:.6f}") # Changed label
        print(f"Test RMSE for {NEW_TARGET_COL} (IPCA): {test_rmse:.6f}") # Changed label
    else:
        print("Skipping test set evaluation.")

    print("Saving model and predictions...")
    cb_model.save_model(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    if y_pred_test is not None and test_df_for_saving is not None and not test_df_for_saving.empty:
        predictions_df = pd.DataFrame({
            ID_COLUMN: test_df_for_saving[ID_COLUMN],
            DATE_COLUMN: test_df_for_saving[DATE_COLUMN],
            f'predicted_{NEW_TARGET_COL}': y_pred_test,
            f'actual_{NEW_TARGET_COL}': y_test
        })
        predictions_df.to_parquet(PREDICTIONS_SAVE_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_SAVE_PATH}")
    else:
        print("Skipping saving test predictions.")

    metrics_to_log = {
        f'val_r2_{NEW_TARGET_COL}': val_r2, f'val_mse_{NEW_TARGET_COL}': val_mse, f'val_rmse_{NEW_TARGET_COL}': val_rmse,
    }
    if not np.isnan(test_r2): metrics_to_log[f'test_r2_{NEW_TARGET_COL}'] = test_r2
    if not np.isnan(test_mse): metrics_to_log[f'test_mse_{NEW_TARGET_COL}'] = test_mse
    if not np.isnan(test_rmse): metrics_to_log[f'test_rmse_{NEW_TARGET_COL}'] = test_rmse
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")

    print("\n--- CatBoost Model Feature Importance for IPCA Factors ---") # Changed message
    if hasattr(cb_model, 'get_feature_importance'):
        ipca_predictor_cols = [f'ipca_factor_{i}' for i in range(X_train.shape[1])] # Changed prefix
        feature_importances = pd.Series(cb_model.get_feature_importance(), index=ipca_predictor_cols)
        sorted_importances = feature_importances.sort_values(ascending=False)
        print("Top IPCA factors by importance (max 32):") # Changed message
        print(sorted_importances.head(32))
    else:
        print("Could not retrieve feature importances from the CatBoost model.")
    print(f"--- {MODEL_NAME} Model Script Complete ---")

if __name__ == '__main__':
    np.random.seed(42)
    main() 