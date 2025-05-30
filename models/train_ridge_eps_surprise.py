import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import csv
from datetime import datetime

# --- Configuration ---
MODEL_NAME = "Ridge_eps_surprise"
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# Input feature files (period t features, period t+1 eps info)
TRAIN_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VAL_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
MODEL_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_model.joblib')
SCALER_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_scaler.joblib')
PREDICTIONS_SAVE_PATH = os.path.join(PREDICTIONS_DIR, f'predictions_{MODEL_NAME.lower()}.parquet')

# Target definition
NEW_TARGET_COL = 'eps_surprise_scaled_by_price'
EPS_ACTUAL_COL = 'eps_actual' # Already t+1 in the data
EPS_MEANEST_COL = 'eps_meanest' # Already t+1 in the data
PRICE_COL_FOR_SCALING = 'prc'   # Price at time t (from features table)

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

# --- Helper function to get predictor columns ---
def get_predictor_feature_columns(df_columns):
    # Columns to exclude from being predictors
    excluded_cols = [
        NEW_TARGET_COL, EPS_ACTUAL_COL, EPS_MEANEST_COL, # Target and its components
        'raw_eps_surprise', # Explicitly exclude the intermediate raw surprise calculation
        'stock_exret', # Original stock return target
        DATE_COLUMN, ID_COLUMN, 'year', 'month', 'day', 
        'eom_date', 'size_class', 'comb_code', 'month_num',
        'stock_exret_t_plus_1', 'stock_exret_t_plus_2', # Future returns
        'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12',
        'SHRCD', 
        'size_port', 'stock_ticker', 'CUSIP', 'comp_name', # Categorical/ID columns
        'target_quintile', # If present from other scripts
        'earnings_surprise' # Unscaled version of target from input data
    ]
    # Also exclude all columns starting with 'eps_' as they are either targets or direct components/alternatives
    predictor_cols = [col for col in df_columns if col not in excluded_cols and 
                           not col.startswith('month_') and not col.startswith('eps_')]
    return predictor_cols

# --- Main Function ---
def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data for each split
    print("Loading data...")
    dfs = {}
    for split_name, path in zip(['train', 'val', 'test'], [TRAIN_FEATURES_PATH, VAL_FEATURES_PATH, TEST_FEATURES_PATH]):
        try:
            dfs[split_name] = pd.read_parquet(path)
            print(f"{split_name.capitalize()} df shape original: {dfs[split_name].shape}")
        except Exception as e:
            print(f"Error loading {split_name} data from {path}: {e}")
            print("CRITICAL: This script cannot proceed without valid data. Ensure the parquet files exist and are correct.")
            print("Especially check validation_features_imputed.parquet and test_features_imputed.parquet for constant zero features identified earlier.")
            return # Exit if data loading fails for any split

    # --- Target Construction & Handling --- 
    print(f"\nConstructing target: {NEW_TARGET_COL}")
    for split_name, df in dfs.items():
        print(f"Processing {split_name} data...")
        # Ensure necessary columns exist
        required_cols_for_target = [EPS_ACTUAL_COL, EPS_MEANEST_COL, PRICE_COL_FOR_SCALING]
        missing_cols = [col for col in required_cols_for_target if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns for target construction in {split_name}_df: {missing_cols}")
            print("Cannot proceed. Please ensure input Parquet files contain these columns.")
            return

        # Calculate raw surprise
        df['raw_eps_surprise'] = df[EPS_ACTUAL_COL] - df[EPS_MEANEST_COL]
        
        # Scale by price. Handle potential division by zero or small prices.
        # Replace 0 or negative prices with a small positive floor (e.g., 0.01) to avoid division errors / extreme values.
        # NaNs in price will also lead to NaN surprise, which is fine as they get dropped.
        price_floor = 0.01
        df[NEW_TARGET_COL] = df['raw_eps_surprise'] / df[PRICE_COL_FOR_SCALING].clip(lower=price_floor)
        
        # Handle infinities that might result from division if price_floor was too small or raw_eps_surprise too large
        df.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace in the whole df, will affect NEW_TARGET_COL

        # Drop rows where the new target is NaN 
        # (due to NaNs in eps_actual, eps_meanest, price, or after inf replacement)
        original_rows = len(df)
        df.dropna(subset=[NEW_TARGET_COL], inplace=True)
        print(f"  {split_name} df: {original_rows - len(df)} rows dropped due to NaN in {NEW_TARGET_COL}.")
        print(f"  {split_name} df shape after target construction & NaN drop: {df.shape}")

        if df.empty:
            print(f"Error: {split_name} DataFrame is empty after target construction. Check input data and target logic.")
            return

        # --- Diagnostic: Inspect NEW_TARGET_COL ---
        print(f"  Descriptive stats for {NEW_TARGET_COL} in {split_name}_df:")
        print(df[NEW_TARGET_COL].describe())
        print(f"  NaN count in {NEW_TARGET_COL}: {df[NEW_TARGET_COL].isnull().sum()}")
        print(f"  Number of unique values: {df[NEW_TARGET_COL].nunique()}\n")

    train_df, val_df, test_df = dfs['train'], dfs['val'], dfs['test']

    if train_df.empty or val_df.empty: # Test_df can be empty if no rows left, but train/val must exist
        raise ValueError("Train or Validation DataFrame is empty. Cannot proceed.")

    # --- Define Predictors (X) and Target (y) ---
    # Use columns from train_df to define the set of predictors
    predictor_cols = get_predictor_feature_columns(train_df.columns)
    print(f"Using {len(predictor_cols)} predictor columns (based on train_df).")
    # print(f"Predictors: {predictor_cols}") # Optional: for debugging

    X_train = train_df[predictor_cols].copy()
    y_train = train_df[NEW_TARGET_COL].copy()
    X_val = val_df[predictor_cols].copy()
    y_val = val_df[NEW_TARGET_COL].copy()
    
    # Test set might be empty if all rows had NaN target, handle this gracefully
    X_test, y_test = None, None
    if not test_df.empty:
        X_test = test_df[predictor_cols].copy()
        y_test = test_df[NEW_TARGET_COL].copy()
    else:
        print("Warning: Test DataFrame is empty after processing. Test set evaluation will be skipped.")

    # --- Preprocessing: Handle NaNs/Infs in predictors and Scale ---
    print("Preprocessing predictors (handling infinities, NaNs, scaling)...")
    for X_df_part in [X_train, X_val, X_test]:
        if X_df_part is not None:
            X_df_part.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_df_part.fillna(0, inplace=True)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    # --- Train Ridge Regression Model ---
    print("Training Ridge regression model...")
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)

    # --- Evaluate Model ---
    print("Evaluating model...")
    y_pred_val = ridge_model.predict(X_val_scaled)
    val_r2 = r2_score(y_val, y_pred_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    print(f"Validation R-squared for {NEW_TARGET_COL}: {val_r2:.6f}")
    print(f"Validation MSE for {NEW_TARGET_COL}: {val_mse:.6f}")

    test_r2, test_mse, y_pred_test = np.nan, np.nan, None
    if X_test_scaled is not None and y_test is not None and not y_test.empty:
        y_pred_test = ridge_model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        print(f"Test R-squared for {NEW_TARGET_COL}: {test_r2:.6f}")
        print(f"Test MSE for {NEW_TARGET_COL}: {test_mse:.6f}")
    else:
        print("Skipping test set evaluation as test data is unavailable or empty.")

    # --- Save Model, Scaler, and Predictions ---
    print("Saving model, scaler...")
    joblib.dump(ridge_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

    if y_pred_test is not None and not test_df.empty: # Ensure test_df still has ID/Date for saving
        predictions_df = pd.DataFrame({
            ID_COLUMN: test_df[ID_COLUMN],
            DATE_COLUMN: test_df[DATE_COLUMN],
            f'predicted_{NEW_TARGET_COL}': y_pred_test,
            f'actual_{NEW_TARGET_COL}': y_test
        })
        predictions_df.to_parquet(PREDICTIONS_SAVE_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_SAVE_PATH}")
    else:
        print("Skipping saving of test predictions as test data was empty or predictions not made.")

    # --- Log Metrics ---
    metrics_to_log = {
        f'val_r2_{NEW_TARGET_COL}': val_r2,
        f'val_mse_{NEW_TARGET_COL}': val_mse,
    }
    if not np.isnan(test_r2):
        metrics_to_log[f'test_r2_{NEW_TARGET_COL}'] = test_r2
    if not np.isnan(test_mse):
        metrics_to_log[f'test_mse_{NEW_TARGET_COL}'] = test_mse
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")

    # --- Sanity Check: Variance of predictions ---
    print(f"\n--- Sanity Check on Predictions for {NEW_TARGET_COL} ---")
    print(f"Std dev of actual y_train: {y_train.std():.6f}") # Check train target std
    print(f"Std dev of actual y_val: {y_val.std():.6f}")
    if y_pred_test is not None:
        print(f"Std dev of actual y_test: {y_test.std():.6f}")
        print(f"Std dev of predicted y_pred_test: {y_pred_test.std():.6f}")
        if y_test.std() > 1e-9 and y_pred_test.std() < 1e-6 :
            print("Warning: Test predictions have near-zero variance while actuals do not. Model might be predicting a constant.")
        else:
            print("Test predictions show some variance (or actuals also have low variance).")
    else:
        print("Test predictions not available for sanity check.")
    print("--- End Sanity Check ---")

    # --- Feature Importance/Coefficients ---
    print("\n--- Ridge Model Coefficients ---")
    if hasattr(ridge_model, 'coef_'):
        # Ensure predictor_cols used for training is available
        # Re-define predictor_cols if not in scope (it should be from main() context)
        # We need to load train_df again just to get columns if this part is run standalone
        # For simplicity, assuming predictor_cols is still in scope from training.
        
        # If X_train was scaled, coefficients are for scaled features.
        # For interpretation, it's often good to link them back to original feature names.
        coefficients = pd.Series(ridge_model.coef_, index=predictor_cols)
        sorted_coefficients = coefficients.abs().sort_values(ascending=False)
        print("Top 20 features by absolute coefficient magnitude:")
        print(sorted_coefficients.head(20))
        
        print("\nCoefficients for top 20 features (actual values):")
        top_20_names = sorted_coefficients.head(20).index
        print(coefficients[top_20_names])
    else:
        print("Could not retrieve coefficients from the Ridge model.")
    # --- End Feature Importance --- 

    print(f"--- {MODEL_NAME} Model Script Complete ---")

if __name__ == '__main__':
    np.random.seed(42)
    main() 