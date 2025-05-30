import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import csv
from datetime import datetime

# --- Configuration ---
MODEL_NAME = "CatBoost_eps_surprise"
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# Input feature files (period t features, period t+1 eps info)
TRAIN_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VAL_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
MODEL_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_model.cbm') # CatBoost model format
PREDICTIONS_SAVE_PATH = os.path.join(PREDICTIONS_DIR, f'predictions_{MODEL_NAME.lower()}.parquet')

# Target definition (same as Ridge script)
NEW_TARGET_COL = 'eps_surprise_scaled_by_price'
EPS_ACTUAL_COL = 'eps_actual' 
EPS_MEANEST_COL = 'eps_meanest'
PRICE_COL_FOR_SCALING = 'prc'   

DATE_COLUMN = 'date'
ID_COLUMN = 'permno'

# --- CSV Logging Function (same as Ridge script) ---
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

# --- Helper function to get predictor columns (same as Ridge script) ---
def get_predictor_feature_columns(df_columns):
    excluded_cols = [
        NEW_TARGET_COL, EPS_ACTUAL_COL, EPS_MEANEST_COL,
        'raw_eps_surprise',
        'stock_exret', 
        DATE_COLUMN, ID_COLUMN, 'year', 'month', 'day', 
        'eom_date', 'size_class', 'comb_code', 'month_num',
        'stock_exret_t_plus_1', 'stock_exret_t_plus_2', 
        'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12',
        'SHRCD', 
        'size_port', 'stock_ticker', 'CUSIP', 'comp_name',
        'target_quintile',
        'earnings_surprise'
    ]
    predictor_cols = [col for col in df_columns if col not in excluded_cols and 
                           not col.startswith('month_') and not col.startswith('eps_')]
    return predictor_cols

# --- Main Function ---
def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    dfs = {}
    for split_name, path in zip(['train', 'val', 'test'], [TRAIN_FEATURES_PATH, VAL_FEATURES_PATH, TEST_FEATURES_PATH]):
        try:
            dfs[split_name] = pd.read_parquet(path)
            print(f"{split_name.capitalize()} df shape original: {dfs[split_name].shape}")
        except Exception as e:
            print(f"Error loading {split_name} data from {path}: {e}"); return

    print(f"\nConstructing target: {NEW_TARGET_COL}")
    for split_name, df in dfs.items():
        print(f"Processing {split_name} data...")
        required_cols_for_target = [EPS_ACTUAL_COL, EPS_MEANEST_COL, PRICE_COL_FOR_SCALING]
        missing_cols = [col for col in required_cols_for_target if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing cols for target in {split_name}_df: {missing_cols}"); return

        df['raw_eps_surprise'] = df[EPS_ACTUAL_COL] - df[EPS_MEANEST_COL]
        price_floor = 0.01
        df[NEW_TARGET_COL] = df['raw_eps_surprise'] / df[PRICE_COL_FOR_SCALING].clip(lower=price_floor)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        original_rows = len(df)
        df.dropna(subset=[NEW_TARGET_COL], inplace=True)
        print(f"  {split_name} df: {original_rows - len(df)} rows dropped due to NaN in {NEW_TARGET_COL}.")
        print(f"  {split_name} df shape after target construction & NaN drop: {df.shape}")
        if df.empty: print(f"Error: {split_name} DataFrame empty after target processing."); return
        print(f"  Descriptive stats for {NEW_TARGET_COL} in {split_name}_df:")
        print(df[NEW_TARGET_COL].describe())

    train_df, val_df, test_df = dfs['train'], dfs['val'], dfs['test']
    if train_df.empty or val_df.empty: 
        raise ValueError("Train or Validation DataFrame is empty.")

    predictor_cols = get_predictor_feature_columns(train_df.columns)
    print(f"Using {len(predictor_cols)} predictor columns.")

    X_train = train_df[predictor_cols].copy()
    y_train = train_df[NEW_TARGET_COL].copy()
    X_val = val_df[predictor_cols].copy()
    y_val = val_df[NEW_TARGET_COL].copy()
    
    X_test, y_test = None, None
    if not test_df.empty:
        X_test = test_df[predictor_cols].copy()
        y_test = test_df[NEW_TARGET_COL].copy()
    else:
        print("Warning: Test DataFrame is empty.")

    # CatBoost handles NaNs internally if specified (though our data should be imputed by now)
    # No explicit scaling needed for CatBoost. 
    # The data from *_features_imputed.parquet is already normalized and imputed.

    print("Training CatBoostRegressor model...")
    cb_model = CatBoostRegressor(
        iterations=5000, # Increased iterations
        learning_rate=0.01, # Decreased learning rate
        depth=5, # Adjusted depth
        loss_function='RMSE',
        eval_metric='RMSE', # Changed eval_metric for early stopping
        random_seed=42,
        verbose=200, # Print evaluation metric every 200 iterations
        early_stopping_rounds=100 # Increased patience
    )

    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        # cat_features=None, # Specify if you have categorical features by name/index that are not preprocessed
    )

    print("Evaluating model...")
    y_pred_val = cb_model.predict(X_val)
    val_r2 = r2_score(y_val, y_pred_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_rmse = np.sqrt(val_mse) # Calculate RMSE for reporting
    print(f"Validation R-squared for {NEW_TARGET_COL}: {val_r2:.6f}")
    print(f"Validation MSE for {NEW_TARGET_COL}: {val_mse:.6f}")
    print(f"Validation RMSE for {NEW_TARGET_COL}: {val_rmse:.6f}")

    test_r2, test_mse, test_rmse, y_pred_test = np.nan, np.nan, np.nan, None
    if X_test is not None and y_test is not None and not y_test.empty:
        y_pred_test = cb_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse) # Calculate RMSE for reporting
        print(f"Test R-squared for {NEW_TARGET_COL}: {test_r2:.6f}")
        print(f"Test MSE for {NEW_TARGET_COL}: {test_mse:.6f}")
        print(f"Test RMSE for {NEW_TARGET_COL}: {test_rmse:.6f}")
    else:
        print("Skipping test set evaluation.")

    print("Saving model and predictions...")
    cb_model.save_model(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    if y_pred_test is not None and not test_df.empty:
        predictions_df = pd.DataFrame({
            ID_COLUMN: test_df[ID_COLUMN],
            DATE_COLUMN: test_df[DATE_COLUMN],
            f'predicted_{NEW_TARGET_COL}': y_pred_test,
            f'actual_{NEW_TARGET_COL}': y_test
        })
        predictions_df.to_parquet(PREDICTIONS_SAVE_PATH, index=False)
        print(f"Predictions saved to {PREDICTIONS_SAVE_PATH}")
    else:
        print("Skipping saving test predictions.")

    metrics_to_log = {
        f'val_r2_{NEW_TARGET_COL}': val_r2,
        f'val_mse_{NEW_TARGET_COL}': val_mse,
        f'val_rmse_{NEW_TARGET_COL}': val_rmse, # Log RMSE
    }
    if not np.isnan(test_r2): metrics_to_log[f'test_r2_{NEW_TARGET_COL}'] = test_r2
    if not np.isnan(test_mse): metrics_to_log[f'test_mse_{NEW_TARGET_COL}'] = test_mse
    if not np.isnan(test_rmse): metrics_to_log[f'test_rmse_{NEW_TARGET_COL}'] = test_rmse # Log RMSE
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")

    print(f"\n--- Sanity Check on Predictions for {NEW_TARGET_COL} ---")
    print(f"Std dev of actual y_train: {y_train.std():.6f}")
    print(f"Std dev of actual y_val: {y_val.std():.6f}")
    if y_pred_test is not None:
        print(f"Std dev of actual y_test: {y_test.std():.6f}")
        print(f"Std dev of predicted y_pred_test: {y_pred_test.std():.6f}")
        if y_test.std() > 1e-9 and y_pred_test.std() < 1e-6 :
            print("Warning: Test predictions have near-zero variance while actuals do not.")
        else:
            print("Test predictions show some variance.")
    else:
        print("Test predictions not available for sanity check.")
    print("--- End Sanity Check ---")

    # --- Feature Importance ---
    print("\n--- CatBoost Model Feature Importance ---")
    if hasattr(cb_model, 'get_feature_importance'):
        feature_importances = pd.Series(cb_model.get_feature_importance(), index=X_train.columns) # X_train.columns are predictor_cols
        sorted_importances = feature_importances.sort_values(ascending=False)
        print("Top 20 features by importance:")
        print(sorted_importances.head(20))
    else:
        print("Could not retrieve feature importances from the CatBoost model.")
    # --- End Feature Importance ---

    print(f"--- {MODEL_NAME} Model Script Complete ---")

if __name__ == '__main__':
    np.random.seed(42)
    main() 