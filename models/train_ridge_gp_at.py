import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import csv
from datetime import datetime

# --- Configuration ---
MODEL_NAME = "Ridge_gp_at_t_plus_1"
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# Input feature files (period t)
TRAIN_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VAL_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
MODEL_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_model.joblib')
SCALER_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_scaler.joblib')
PREDICTIONS_SAVE_PATH = os.path.join(PREDICTIONS_DIR, f'predictions_{MODEL_NAME.lower()}.parquet')

# Target fundamental and identifiers
FUNDAMENTAL_TARGET_ORIG_COL = 'gp_at' # The column to predict for t+1
NEW_TARGET_COL = 'gp_at_t_plus_1'
DATE_COLUMN = 'date'
ID_COLUMN = 'permno'

# --- Define CSV Logging Function --- (Copied from train_catboost.py)
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

# --- Helper function to get predictor columns (adapted from generate_ae_factors.py) ---
def get_predictor_feature_columns(df):
    excluded_cols = [
        'stock_exret', # Original stock return target, not a predictor for fundamental
        NEW_TARGET_COL, # The new target itself
        FUNDAMENTAL_TARGET_ORIG_COL, # Original fundamental column at time t if it's different from other predictors
        DATE_COLUMN, ID_COLUMN, 'year', 'month', 'day', 
        'eom_date', 'size_class', 'comb_code', 'month_num',
        'stock_exret_t_plus_1', 'stock_exret_t_plus_2', # Future returns
        'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12',
        'SHRCD', 
        'size_port', 'stock_ticker', 'CUSIP', 'comp_name', # Categorical/ID columns
        'target_quintile' # If present from other scripts
    ]
    # Ensure FUNDAMENTAL_TARGET_ORIG_COL is in excluded if it's one of the predictors by name
    if FUNDAMENTAL_TARGET_ORIG_COL not in excluded_cols:
        excluded_cols.append(FUNDAMENTAL_TARGET_ORIG_COL)
        
    predictor_cols = [col for col in df.columns if col not in excluded_cols and 
                           not col.startswith('month_') and not col.startswith('eps_')]
    # We should NOT exclude current 'gp_at' if it's a predictor for 'gp_at_t_plus_1'
    # The current logic above would exclude FUNDAMENTAL_TARGET_ORIG_COL ('gp_at')
    # We need to decide if current gp_at is a predictor for future gp_at.
    # For now, let's assume it IS a valid predictor (autoregressive component).
    # So, we will remove it from excluded_cols if it was added.
    if FUNDAMENTAL_TARGET_ORIG_COL in excluded_cols and FUNDAMENTAL_TARGET_ORIG_COL in df.columns:
        # Allow current value of the fundamental to be a predictor if it exists and is not the target itself.
        # The target is NEW_TARGET_COL.
        pass # Keep it excluded by default for now, can refine if gp_at (t) should predict gp_at (t+1)
             # The initial list construction above `if FUNDAMENTAL_TARGET_ORIG_COL not in excluded_cols:` handles this.
             # Actually, it's simpler: just ensure FUNDAMENTAL_TARGET_ORIG_COL is NOT in excluded_cols if it's different from NEW_TARGET_COL
             # The initial construction of predictor_cols already handles not including NEW_TARGET_COL.
             # We just need to make sure FUNDAMENTAL_TARGET_ORIG_COL (at time t) is included if desired.

    # Revised logic: Start with all columns, remove specific non-predictors and future-looking ones.
    # Keep current value of fundamental (FUNDAMENTAL_TARGET_ORIG_COL) as a potential predictor.
    final_excluded_cols = [
        'stock_exret', NEW_TARGET_COL, 
        DATE_COLUMN, ID_COLUMN, 'year', 'month', 'day', 'eom_date', 'size_class', 'comb_code', 'month_num',
        'stock_exret_t_plus_1', 'stock_exret_t_plus_2', 'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12',
        'SHRCD', 'size_port', 'stock_ticker', 'CUSIP', 'comp_name', 'target_quintile'
    ]
    # Remove eps_ related columns as per earlier decision for general predictors
    predictor_cols = [col for col in df.columns if col not in final_excluded_cols and not col.startswith('eps_') and not col.startswith('month_')]

    return predictor_cols

# --- Main Function ---
def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure results dir also exists

    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_FEATURES_PATH)
    val_df = pd.read_parquet(VAL_FEATURES_PATH)
    test_df = pd.read_parquet(TEST_FEATURES_PATH)

    print(f"Train shape original: {train_df.shape}")
    print(f"Val shape original: {val_df.shape}")
    print(f"Test shape original: {test_df.shape}")

    # --- Diagnostic: Inspect FUNDAMENTAL_TARGET_ORIG_COL before target generation ---
    print(f"\n--- Diagnostics for {FUNDAMENTAL_TARGET_ORIG_COL} before target generation ---")
    for df_name, df_part in [('train_df', train_df), ('val_df', val_df), ('test_df', test_df)]:
        if FUNDAMENTAL_TARGET_ORIG_COL in df_part.columns:
            print(f"Descriptive stats for {FUNDAMENTAL_TARGET_ORIG_COL} in {df_name}:")
            print(df_part[FUNDAMENTAL_TARGET_ORIG_COL].describe())
            print(f"NaN count in {FUNDAMENTAL_TARGET_ORIG_COL} in {df_name}: {df_part[FUNDAMENTAL_TARGET_ORIG_COL].isnull().sum()}")
            print(f"Number of unique values in {FUNDAMENTAL_TARGET_ORIG_COL} in {df_name}: {df_part[FUNDAMENTAL_TARGET_ORIG_COL].nunique()}\n")
        else:
            print(f"Warning: {FUNDAMENTAL_TARGET_ORIG_COL} not found in {df_name}.")
    print("--- End Pre-Target-Gen Diagnostics ---\n")
    # --- End Diagnostic ---

    # --- Generate Target Variable: FUNDAMENTAL_TARGET_ORIG_COL at t+1 ---
    print(f"Generating target variable: {NEW_TARGET_COL} (from {FUNDAMENTAL_TARGET_ORIG_COL})...")
    
    # Concatenate for easier group-wise shift, then split back
    train_df['_split'] = 'train'
    val_df['_split'] = 'val'
    test_df['_split'] = 'test'
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Ensure correct sorting for shift
    full_df[DATE_COLUMN] = pd.to_datetime(full_df[DATE_COLUMN])
    full_df.sort_values(by=[ID_COLUMN, DATE_COLUMN], inplace=True)
    
    # Generate t+1 target
    full_df[NEW_TARGET_COL] = full_df.groupby(ID_COLUMN)[FUNDAMENTAL_TARGET_ORIG_COL].shift(-1)
    
    # Split back
    train_df = full_df[full_df['_split'] == 'train'].drop(columns=['_split'])
    val_df = full_df[full_df['_split'] == 'val'].drop(columns=['_split'])
    test_df = full_df[full_df['_split'] == 'test'].drop(columns=['_split'])

    # Drop rows where the new target is NaN (these are the last observations for each stock)
    train_df.dropna(subset=[NEW_TARGET_COL], inplace=True)
    val_df.dropna(subset=[NEW_TARGET_COL], inplace=True)
    test_df.dropna(subset=[NEW_TARGET_COL], inplace=True)

    print(f"Train shape after target generation & NaN drop: {train_df.shape}")
    print(f"Val shape after target generation & NaN drop: {val_df.shape}")
    print(f"Test shape after target generation & NaN drop: {test_df.shape}")

    # --- Diagnostic: Inspect NEW_TARGET_COL after target generation ---
    print(f"\n--- Diagnostics for {NEW_TARGET_COL} after target generation and NaN drop ---")
    for df_name, df_part in [('train_df', train_df), ('val_df', val_df), ('test_df', test_df)]:
        if NEW_TARGET_COL in df_part.columns:
            print(f"Descriptive stats for {NEW_TARGET_COL} in {df_name}:")
            print(df_part[NEW_TARGET_COL].describe())
            print(f"NaN count in {NEW_TARGET_COL} in {df_name}: {df_part[NEW_TARGET_COL].isnull().sum()}") # Should be 0
            print(f"Number of unique values in {NEW_TARGET_COL} in {df_name}: {df_part[NEW_TARGET_COL].nunique()}\n")
        else:
            print(f"Warning: {NEW_TARGET_COL} not found in {df_name} after processing.")
    print("--- End Post-Target-Gen Diagnostics ---\n")
    # --- End Diagnostic ---

    if train_df.empty or val_df.empty:
        raise ValueError("Train or Validation DataFrame is empty after target generation. Check data or shift logic.")

    # --- Define Predictors (X) and Target (y) ---
    predictor_cols = get_predictor_feature_columns(train_df)
    print(f"Using {len(predictor_cols)} predictor columns.")
    # print(f"Predictors: {predictor_cols}") # Optional: for debugging

    X_train = train_df[predictor_cols].copy()
    y_train = train_df[NEW_TARGET_COL].copy()
    X_val = val_df[predictor_cols].copy()
    y_val = val_df[NEW_TARGET_COL].copy()
    X_test = test_df[predictor_cols].copy()
    y_test = test_df[NEW_TARGET_COL].copy()

    # --- Preprocessing: Handle NaNs in predictors and Scale ---
    print("Preprocessing predictors (handling NaNs, scaling)...")

    # Step 1: Replace infinities with NaN in predictor sets
    for X_df_part in [X_train, X_val, X_test]:
        X_df_part.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Step 2: Fill NaNs in predictors (e.g., with 0 or median). For Ridge, 0 is often fine after scaling.
    X_train.fillna(0, inplace=True)
    X_val.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # Scaling (RobustScaler might be better for financial data with outliers)
    # scaler = StandardScaler()
    scaler = RobustScaler() 
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Train Ridge Regression Model ---
    print("Training Ridge regression model...")
    # Alpha is the regularization strength. Can be tuned.
    ridge_model = Ridge(alpha=1.0, random_state=42) 
    ridge_model.fit(X_train_scaled, y_train)

    # --- Evaluate Model ---
    print("Evaluating model...")
    # Predictions on validation set
    y_pred_val = ridge_model.predict(X_val_scaled)
    val_r2 = r2_score(y_val, y_pred_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    print(f"Validation R-squared for {NEW_TARGET_COL}: {val_r2:.6f}")
    print(f"Validation MSE for {NEW_TARGET_COL}: {val_mse:.6f}")

    # Predictions on test set
    y_pred_test = ridge_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print(f"Test R-squared for {NEW_TARGET_COL}: {test_r2:.6f}")
    print(f"Test MSE for {NEW_TARGET_COL}: {test_mse:.6f}")

    # --- Save Model, Scaler, and Predictions ---
    print("Saving model, scaler, and predictions...")
    joblib.dump(ridge_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

    predictions_df = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        DATE_COLUMN: test_df[DATE_COLUMN],
        f'predicted_{NEW_TARGET_COL}': y_pred_test,
        f'actual_{NEW_TARGET_COL}': y_test
    })
    predictions_df.to_parquet(PREDICTIONS_SAVE_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_SAVE_PATH}")

    # --- Log Metrics ---
    metrics_to_log = {
        f'val_r2_{FUNDAMENTAL_TARGET_ORIG_COL}': val_r2,
        f'val_mse_{FUNDAMENTAL_TARGET_ORIG_COL}': val_mse,
        f'test_r2_{FUNDAMENTAL_TARGET_ORIG_COL}': test_r2,
        f'test_mse_{FUNDAMENTAL_TARGET_ORIG_COL}': test_mse,
    }
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")

    # --- Sanity Check: Variance of predictions ---
    print(f"\n--- Sanity Check on Test Predictions for {NEW_TARGET_COL} ---")
    print(f"Std dev of actual y_test: {y_test.std():.6f}")
    print(f"Std dev of predicted y_pred_test: {y_pred_test.std():.6f}")
    if y_pred_test.std() < 1e-6 and y_test.std() > 1e-6:
        print("Warning: Predictions have near-zero variance while actuals do not. Model might be predicting a constant.")
    else:
        print("Predictions show some variance.")
    print("--- End Sanity Check ---")

    print(f"--- {MODEL_NAME} Model Script Complete ---")

if __name__ == '__main__':
    # Set seed for reproducibility if any random operations were part of Ridge (e.g. solver related)
    np.random.seed(42) 
    main() 