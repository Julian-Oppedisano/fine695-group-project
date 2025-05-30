import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import r2_score, mean_squared_error
import os
import joblib # For scaler if we decide to use external scaling, though AutoGluon handles it.
from datetime import datetime
import csv

# --- Configuration ---
MODEL_NAME = 'autogluon' # Or 'auto' to match pred_auto.csv, let's use 'autogluon' for clarity here
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
TEST_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models', MODEL_NAME) # AutoGluon saves models in its own directory
RESULTS_PRED_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RESULTS_DIR = RESULTS_PRED_DIR # For performance_summary.csv

TARGET_COLUMN = 'stock_exret'
ID_COLUMN = 'permno'
DATE_COLUMN = 'date'

# Time limit for AutoGluon training (in seconds)
# 3 hours = 3 * 60 * 60 = 10800 seconds
TIME_LIMIT_SECONDS = 10800 
# For quick testing, use a shorter time, e.g., 600 for 10 minutes
# TIME_LIMIT_SECONDS = 600 


# --- CSV Logging Function (copied from other scripts for consistency) ---
CSV_FILE = os.path.join(RESULTS_DIR, 'performance_summary.csv')

def log_metrics_to_csv(model_name, metrics_dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.isfile(CSV_FILE)
    
    # Ensure only expected metrics are in metrics_dict before this sort
    # Standard keys: 'mse', 'oos_r2'
    metric_keys = sorted([k for k in metrics_dict.keys() if k in ['mse', 'oos_r2']])
    fieldnames = ['timestamp', 'model_name'] + metric_keys
    
    row_to_log = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'model_name': model_name}
    row_to_log.update(metrics_dict)

    # Filter row_to_log to only include fieldnames to prevent extra columns
    filtered_row_to_log = {key: row_to_log[key] for key in fieldnames if key in row_to_log}

    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') # extrasaction='ignore' is safer
        
        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writeheader()
        writer.writerow(filtered_row_to_log)
# --- End CSV Logging Function ---

def get_predictor_columns(df, target_col):
    # Exclude IDs, date components, and the target itself.
    # AutoGluon can often infer, but explicit exclusion can be safer.
    identifier_cols = ['permno', 'stock_ticker', 'CUSIP', 'comp_name']
    date_related_cols = ['date', 'year', 'month', 'day', 'eom_date', 'month_num'] # Add common date parts
    # Any other known non-predictive columns or future leakage columns
    other_exclusions = ['ret_eom'] 
    
    cols_to_exclude = identifier_cols + date_related_cols + other_exclusions + [target_col]
    # Take all columns not in the exclusion list
    predictor_cols = [col for col in df.columns if col not in cols_to_exclude]
    
    # Sanity check: remove target if it accidentally got in
    if target_col in predictor_cols:
        predictor_cols.remove(target_col)
        
    print(f"Identified {len(predictor_cols)} predictor columns for AutoGluon.")
    return predictor_cols

def main():
    print(f"--- Training {MODEL_NAME} Model with AutoGluon ---")
    
    print("Loading imputed datasets...")
    try:
        train_df_full = pd.read_parquet(TRAIN_IMPUTED_INPUT_PATH)
        test_df_full = pd.read_parquet(TEST_IMPUTED_INPUT_PATH)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Data Preparation ---
    # AutoGluon prefers the target column to be part of the DataFrame passed to `fit`.
    # Drop rows where target is NaN for training, as AutoGluon might error or handle it suboptimally.
    train_df_full.dropna(subset=[TARGET_COLUMN], inplace=True)
    
    # Keep a copy of test permno and date for final output, before AutoGluon might drop/alter them
    test_ids_dates = test_df_full[[ID_COLUMN, DATE_COLUMN]].copy()

    # Select predictor columns + target for training data
    # AutoGluon will internally handle feature selection/preprocessing from these.
    # No explicit scaling or encoding needed outside AutoGluon for most cases.
    # Predictor columns identification could be simpler: just drop IDs and date.
    # For now, using a more explicit get_predictor_columns logic might be too restrictive.
    # AutoGluon can select from all columns if we just specify the target.
    
    # For AutoGluon, we provide the DataFrame and specify the label.
    # It's good practice to remove clear non-predictors or ID columns if they might confuse it,
    # or if they are not meant to be features.
    
    # Columns to keep for AutoGluon training and prediction: predictors + target
    # AutoGluon will ignore columns not present in training data during prediction if `feature_generator` is default.
    cols_to_drop_for_ag = ['stock_ticker', 'CUSIP', 'comp_name', 'year', 'month', 'day', 'eom_date', 'month_num', 'ret_eom']
    
    train_ag_df = train_df_full.drop(columns=[col for col in cols_to_drop_for_ag if col in train_df_full.columns], errors='ignore')
    test_ag_df = test_df_full.drop(columns=[col for col in cols_to_drop_for_ag if col in test_df_full.columns], errors='ignore')

    # --- Explicitly handle infinities and large numbers before passing to AutoGluon ---
    # Identify numeric columns excluding the target, permno, and date as they are handled separately or are IDs
    numeric_cols_train = train_ag_df.select_dtypes(include=np.number).columns.tolist()
    if TARGET_COLUMN in numeric_cols_train: numeric_cols_train.remove(TARGET_COLUMN)
    if ID_COLUMN in numeric_cols_train: numeric_cols_train.remove(ID_COLUMN)
    if DATE_COLUMN in numeric_cols_train: numeric_cols_train.remove(DATE_COLUMN) # Date is likely already converted or dropped
    
    numeric_cols_test = test_ag_df.select_dtypes(include=np.number).columns.tolist()
    if TARGET_COLUMN in numeric_cols_test: numeric_cols_test.remove(TARGET_COLUMN)
    if ID_COLUMN in numeric_cols_test: numeric_cols_test.remove(ID_COLUMN)
    if DATE_COLUMN in numeric_cols_test: numeric_cols_test.remove(DATE_COLUMN)

    print(f"Cleaning {len(numeric_cols_train)} numeric columns in training data.")
    for col in numeric_cols_train:
        # Replace inf with nan first, as nan_to_num handles nans by converting to zero by default
        train_ag_df[col] = np.nan_to_num(train_ag_df[col].replace([np.inf, -np.inf], np.nan), nan=0.0, posinf=1e9, neginf=-1e9)

    print(f"Cleaning {len(numeric_cols_test)} numeric columns in test data.")
    for col in numeric_cols_test:
        test_ag_df[col] = np.nan_to_num(test_ag_df[col].replace([np.inf, -np.inf], np.nan), nan=0.0, posinf=1e9, neginf=-1e9)
    print("Numeric data cleaned of infinities and large values.")
    # --- End data cleaning ---

    # Ensure target is not in test_ag_df if it exists, it will be used for evaluation by AG.
    # For final predictions for submission, AG's predict method doesn't need target in test data.
    y_test_true = None
    if TARGET_COLUMN in test_ag_df.columns:
        y_test_true = test_ag_df[TARGET_COLUMN].copy() # For our own evaluation later
        # test_ag_df_for_prediction = test_ag_df.drop(columns=[TARGET_COLUMN]) # AG predict doesn't need it
    # else:
        # test_ag_df_for_prediction = test_ag_df

    print(f"Training data shape for AutoGluon: {train_ag_df.shape}")
    print(f"Test data shape for AutoGluon: {test_ag_df.shape}")

    # --- AutoGluon Predictor Initialization and Training ---
    # Path where AutoGluon will save its models and artifacts
    autogluon_model_path = os.path.join(SAVED_MODEL_DIR, f"autogluon_{MODEL_NAME}_models")
    
    print(f"Initializing TabularPredictor. Models will be saved to: {autogluon_model_path}")
    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        problem_type='regression', # Assuming stock_exret is a continuous target
        eval_metric='r2',       # Optimize for R-squared. Other options: 'mean_squared_error', etc.
        path=autogluon_model_path
    )

    print(f"Starting AutoGluon training for a maximum of {TIME_LIMIT_SECONDS} seconds...")
    # We can specify presets like 'best_quality', 'high_quality', 'good_quality', 'medium_quality'
    # 'best_quality' can be very time-consuming. 'good_quality' is a reasonable balance.
    # Add verbosity for more detailed logs from AutoGluon
    predictor.fit(
        train_ag_df,
        time_limit=TIME_LIMIT_SECONDS,
        presets='good_quality', # Or 'medium_quality' for faster runs, 'best_quality' for more thoroughness
        # excluded_model_types=['FASTAI'], # Example: if some models cause issues
        # hyperparameter_tune_kwargs='auto', # Enables hyperparameter tuning
        verbosity=2 # 0 (silent), 1, 2 (default), 3, 4 (debug)
    )
    print("AutoGluon training complete.")

    # --- Leaderboard and Model Summary ---
    print("\nAutoGluon Model Leaderboard:")
    leaderboard = predictor.leaderboard(test_ag_df, silent=False) # Pass test data for out-of-sample scores
    print(leaderboard)

    print("\nPredictor summary:")
    print(predictor.fit_summary(verbosity=3)) # Higher verbosity for more details

    # --- Make Predictions on Test Set ---
    print("\nMaking predictions on the test set...")
    # If TARGET_COLUMN was in test_ag_df, AutoGluon's predict method ignores it.
    y_pred_test = predictor.predict(test_ag_df)
    print("Test predictions made.")

    # --- Evaluate Model (using our own metrics for consistency if needed) ---
    if y_test_true is not None and not y_test_true.isnull().all():
        # Drop NaNs from y_test_true and corresponding y_pred_test for fair comparison
        valid_idx = y_test_true.notnull()
        y_test_true_clean = y_test_true[valid_idx]
        y_pred_test_clean = y_pred_test[valid_idx]
        
        if len(y_test_true_clean) > 0:
            test_oos_r2 = r2_score(y_test_true_clean, y_pred_test_clean)
            test_mse = mean_squared_error(y_test_true_clean, y_pred_test_clean)
            print(f"Out-of-Sample (OOS) R-squared on Test Set (calculated manually): {test_oos_r2:.6f}")
            print(f"Mean Squared Error (MSE) on Test Set (calculated manually): {test_mse:.6f}")

            # --- Log Metrics ---
            metrics_to_log = {
                'oos_r2': test_oos_r2,
                'mse': test_mse,
            }
            # Use 'auto' for model name if outline specifies pred_auto.csv
            log_metrics_to_csv('auto', metrics_to_log) 
            print(f"Metrics logged to {CSV_FILE} for model 'auto'.")
        else:
            print("Test set has no valid (non-NaN) targets. Skipping manual R2/MSE calculation and logging.")
    else:
        print("No true target values in test set or all are NaNs; cannot calculate R2/MSE manually. Performance based on AG leaderboard.")


    # --- Save Test Predictions (CSV) ---
    # Ensure the order of predictions matches the original test_df_full for permno and date
    # AutoGluon predict should maintain row order if test_data is a DataFrame.
    
    if len(y_pred_test) != len(test_ids_dates):
        print(f"Warning: Length mismatch! Predictions: {len(y_pred_test)}, Original Test IDs: {len(test_ids_dates)}")
        print("Attempting to align based on index if test_ag_df retained original index.")
        # This part can be tricky if AutoGluon re-indexed or dropped rows.
        # A robust way is to ensure test_ag_df passed to predict has an index that maps to test_ids_dates
        if test_ag_df.index.equals(test_ids_dates.index):
             final_test_predictions_df = test_ids_dates.copy()
             final_test_predictions_df['prediction'] = y_pred_test.values # y_pred_test is a Series
        else: # Fallback, might be risky if order is not guaranteed
            print("Index mismatch. Saving predictions with potentially misaligned permno/date. Review carefully.")
            final_test_predictions_df = pd.DataFrame({
                 ID_COLUMN: test_ids_dates[ID_COLUMN].iloc[:len(y_pred_test)] if len(test_ids_dates) > len(y_pred_test) else test_ids_dates[ID_COLUMN], # Risky
                 DATE_COLUMN: test_ids_dates[DATE_COLUMN].iloc[:len(y_pred_test)] if len(test_ids_dates) > len(y_pred_test) else test_ids_dates[DATE_COLUMN], # Risky
                'prediction': y_pred_test.values
            })
            if len(final_test_predictions_df) != len(y_pred_test) : # If lengths still don't match after trying to slice
                 final_test_predictions_df = pd.DataFrame({'prediction': y_pred_test.values}) # Save only preds
                 print("Could not align permno/date. Saving only predictions.")

    else: # Lengths match, assume order is preserved
        final_test_predictions_df = test_ids_dates.copy()
        final_test_predictions_df['prediction'] = y_pred_test.values # y_pred_test is a pandas Series

    os.makedirs(RESULTS_PRED_DIR, exist_ok=True)
    csv_output_filename = "pred_auto.csv" # As per outline
    csv_predictions_save_path = os.path.join(RESULTS_PRED_DIR, csv_output_filename)
    
    final_test_predictions_df[[ID_COLUMN, DATE_COLUMN, 'prediction']].to_csv(csv_predictions_save_path, index=False)
    print(f"Test predictions saved to {csv_predictions_save_path}")

    print(f"\n--- {MODEL_NAME} Model Script (AutoGluon) Complete ---")

if __name__ == "__main__":
    main() 