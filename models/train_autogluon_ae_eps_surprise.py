import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
from autogluon.tabular import TabularPredictor
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration ---
MODEL_NAME = "AutoGluon_AE_eps_surprise"
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# Paths for original data (to get target and IDs)
TRAIN_FEATURES_ORIG_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VAL_FEATURES_ORIG_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_FEATURES_ORIG_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

# Paths for AE factor data
TRAIN_AE_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'train_ae_factors.parquet')
VAL_AE_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'validation_ae_factors.parquet')
TEST_AE_FACTORS_PATH = os.path.join(PROCESSED_DIR, 'test_ae_factors.parquet')

# Output directories
SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models', MODEL_NAME.lower() + '_predictor') # AutoGluon saves a directory
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
PREDICTIONS_SAVE_PATH = os.path.join(PREDICTIONS_DIR, f'predictions_{MODEL_NAME.lower()}.parquet')

# Target definition
TARGET_COL = 'eps_surprise_scaled_by_price'
EPS_ACTUAL_COL = 'eps_actual' 
EPS_MEANEST_COL = 'eps_meanest' 
PRICE_COL_FOR_SCALING = 'prc'   

DATE_COLUMN = 'date'
ID_COLUMN = 'permno'

# AutoGluon settings
TIME_LIMIT_SECS = 3600  # 1 hour
PRESETS_QUALITY = 'good_quality' # 'best_quality', 'high_quality', 'good_quality', 'medium_quality'

# --- CSV Logging Function ---
RESULTS_DIR = 'results'
CSV_FILE = os.path.join(RESULTS_DIR, 'performance_summary.csv')

def log_metrics_to_csv(model_name, metrics_dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.isfile(CSV_FILE)
    metric_keys = sorted([k for k in metrics_dict.keys() if k not in ['timestamp', 'model_name']])
    if any('rmse' in k for k in metrics_dict.keys()): 
        for rmse_key_type in ['val', 'test']:
            full_rmse_key = f'{rmse_key_type}_rmse_{TARGET_COL}'
            if metrics_dict.get(full_rmse_key) is not None and full_rmse_key not in metric_keys:
                metric_keys.append(full_rmse_key)
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

# --- Helper function to load and merge data ---
def load_and_prepare_data(original_path, factors_path, split_name):
    print(f"Loading original {split_name} data for target and IDs from {original_path}...")
    try:
        df_original = pd.read_parquet(original_path, columns=[ID_COLUMN, DATE_COLUMN, EPS_ACTUAL_COL, EPS_MEANEST_COL, PRICE_COL_FOR_SCALING])
    except Exception as e:
        print(f"Error loading original {split_name} data: {e}"); return None

    print(f"Loading AE factors for {split_name} from {factors_path}...")
    try:
        df_factors = pd.read_parquet(factors_path)
    except Exception as e:
        print(f"Error loading AE factors for {split_name}: {e}"); return None

    print(f"Constructing target and merging for {split_name}...")
    df_original['raw_eps_surprise'] = df_original[EPS_ACTUAL_COL] - df_original[EPS_MEANEST_COL]
    price_floor = 0.01
    df_original[TARGET_COL] = df_original['raw_eps_surprise'] / df_original[PRICE_COL_FOR_SCALING].clip(lower=price_floor)
    df_original.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Select necessary columns from original_df for merge and target
    df_target_and_ids = df_original[[ID_COLUMN, DATE_COLUMN, TARGET_COL]]
    
    merged_df = pd.merge(df_target_and_ids, df_factors, on=[ID_COLUMN, DATE_COLUMN], how='inner')
    
    initial_rows = len(merged_df)
    merged_df.dropna(subset=[TARGET_COL], inplace=True) # Drop rows where target is NaN
    print(f"  {split_name} merged_df: {initial_rows - len(merged_df)} rows dropped due to NaN in {TARGET_COL}.")
    print(f"  {split_name} merged_df shape after merge & NaN drop: {merged_df.shape}")

    if merged_df.empty:
        print(f"Error: {split_name} DataFrame is empty after processing."); return None
    
    # Predictor columns are ae_factor_ prefixed, plus the TARGET_COL
    factor_cols = [col for col in merged_df.columns if col.startswith('ae_factor_')]
    if not factor_cols:
        print(f"Error: No AE factor columns found in {split_name} merged_df."); return None
        
    return merged_df[factor_cols + [TARGET_COL, ID_COLUMN, DATE_COLUMN]] # Keep ID, Date for test predictions

# --- Main Function ---
def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_data = load_and_prepare_data(TRAIN_FEATURES_ORIG_PATH, TRAIN_AE_FACTORS_PATH, "train")
    val_data = load_and_prepare_data(VAL_FEATURES_ORIG_PATH, VAL_AE_FACTORS_PATH, "validation")
    test_data = load_and_prepare_data(TEST_FEATURES_ORIG_PATH, TEST_AE_FACTORS_PATH, "test")

    if train_data is None or val_data is None:
        print("Critical error: Train or Validation data could not be prepared. Exiting.")
        return

    print(f"\nTraining AutoGluon TabularPredictor...")
    # Drop ID and Date from train and val before passing to predictor.fit
    train_fit_data = train_data.drop(columns=[ID_COLUMN, DATE_COLUMN])
    val_fit_data = val_data.drop(columns=[ID_COLUMN, DATE_COLUMN])

    predictor = TabularPredictor(
        label=TARGET_COL,
        path=SAVED_MODEL_DIR,
        problem_type='regression',
        eval_metric='r2' 
    )

    predictor.fit(
        train_fit_data,
        tuning_data=val_fit_data,
        time_limit=TIME_LIMIT_SECS,
        presets=PRESETS_QUALITY,
        use_bag_holdout=True,
        # ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'} # If memory issues with ensemble
    )

    print("\n--- AutoGluon Training Summary ---")
    predictor.fit_summary(show_plot=False) # show_plot=True can cause issues in non-interactive envs

    leaderboard = predictor.leaderboard(test_data.drop(columns=[ID_COLUMN, DATE_COLUMN]) if test_data is not None else None, silent=True)
    if leaderboard is not None:
        print("\n--- AutoGluon Leaderboard (on test data if available) ---")
        print(leaderboard)
    
    # Store validation performance from AutoGluon directly
    # AutoGluon's reported score_val is on the validation data using the specified eval_metric
    # It picks the best model based on this. If leaderboard is on test_data, get val score from best model.
    # For simplicity, we'll re-evaluate the best model on val_data if needed, or take from summary.
    # fit_summary() provides performance of models on validation data.
    # The default .score() uses the validation data used during fit if no data is provided.
    
    # Use a specific model if desired, e.g., predictor.predict(data, model='WeightedEnsemble_L2')
    # For now, default is fine (usually the best ensemble)

    print("\nEvaluating best model on validation data...")
    y_pred_val = predictor.predict(val_fit_data.drop(columns=[TARGET_COL]))
    val_r2 = r2_score(val_data[TARGET_COL], y_pred_val)
    val_mse = mean_squared_error(val_data[TARGET_COL], y_pred_val)
    val_rmse = np.sqrt(val_mse)
    print(f"Validation R-squared (best model): {val_r2:.6f}")
    print(f"Validation MSE (best model): {val_mse:.6f}")
    print(f"Validation RMSE (best model): {val_rmse:.6f}")

    test_r2, test_mse, test_rmse = np.nan, np.nan, np.nan
    if test_data is not None and not test_data.empty:
        print("\nEvaluating best model on test data...")
        X_test_predictors = test_data.drop(columns=[TARGET_COL, ID_COLUMN, DATE_COLUMN])
        y_pred_test = predictor.predict(X_test_predictors)
        y_test_actual = test_data[TARGET_COL]

        test_r2 = r2_score(y_test_actual, y_pred_test)
        test_mse = mean_squared_error(y_test_actual, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        print(f"Test R-squared (best model): {test_r2:.6f}")
        print(f"Test MSE (best model): {test_mse:.6f}")
        print(f"Test RMSE (best model): {test_rmse:.6f}")

        print("Saving test predictions...")
        predictions_df = pd.DataFrame({
            ID_COLUMN: test_data[ID_COLUMN],
            DATE_COLUMN: test_data[DATE_COLUMN],
            f'predicted_{TARGET_COL}': y_pred_test,
            f'actual_{TARGET_COL}': y_test_actual
        })
        predictions_df.to_parquet(PREDICTIONS_SAVE_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_SAVE_PATH}")
    else:
        print("Skipping test set evaluation and prediction saving as test data is unavailable.")

    metrics_to_log = {
        f'val_r2_{TARGET_COL}': val_r2, f'val_mse_{TARGET_COL}': val_mse, f'val_rmse_{TARGET_COL}': val_rmse,
    }
    if not np.isnan(test_r2): metrics_to_log[f'test_r2_{TARGET_COL}'] = test_r2
    if not np.isnan(test_mse): metrics_to_log[f'test_mse_{TARGET_COL}'] = test_mse
    if not np.isnan(test_rmse): metrics_to_log[f'test_rmse_{TARGET_COL}'] = test_rmse
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")

    print(f"\n--- {MODEL_NAME} Model Script Complete ---")
    print(f"AutoGluon models saved in: {SAVED_MODEL_DIR}")

if __name__ == '__main__':
    np.random.seed(42) # For reproducibility where possible with AutoGluon
    main() 