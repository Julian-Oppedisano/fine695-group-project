import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import csv
from datetime import datetime

# --- Configuration ---
MODEL_NAME = "XGBoost"
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_model.joblib') # Using joblib for consistency
SCALER_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_scaler.joblib')
PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, f'predictions_{MODEL_NAME.lower()}.parquet')

TARGET_COLUMN = 'stock_exret'
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

# --- Helper Functions ---
def get_predictor_columns(df):
    excluded_cols = [TARGET_COLUMN, DATE_COLUMN, ID_COLUMN, 'year', 'month', 'day', 
                     'eom_date', 'size_class', 'comb_code', 'month_num',
                     'stock_exret_t_plus_1', 'stock_exret_t_plus_2', 
                     'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12']
    predictor_cols = [col for col in df.columns if col not in excluded_cols and not col.startswith('month_')]
    # XGBoost can handle categorical features if they are label encoded or of type 'category'
    # For now, ensure all are numeric or handle categorical encoding explicitly if needed.
    return predictor_cols

def main():
    print(f"--- Training {MODEL_NAME} Model ---")

    print("Loading imputed datasets...")
    train_df = pd.read_parquet(TRAIN_IMPUTED_INPUT_PATH)
    # val_df = pd.read_parquet(VALIDATION_IMPUTED_INPUT_PATH) # XGBoost can use an eval set
    test_df = pd.read_parquet(TEST_IMPUTED_INPUT_PATH)

    print(f"Train df shape: {train_df.shape}")
    # print(f"Validation df shape: {val_df.shape}")
    print(f"Test df shape: {test_df.shape}")

    for df_name, df_content in [('train', train_df), ('test', test_df)]: # ('val', val_df)
        if df_content[TARGET_COLUMN].isnull().all():
            raise ValueError(f"Target column '{TARGET_COLUMN}' in {df_name}_df is all NaNs.")
        if TARGET_COLUMN in get_predictor_columns(df_content):
             raise ValueError(f"Target column '{TARGET_COLUMN}' found in predictor list for {df_name}_df.")

    predictor_cols = get_predictor_columns(train_df)
    
    # Ensure all predictor columns are numeric for XGBoost or handle encoding
    # For simplicity, this script assumes imputation and prior processing handled non-numeric types appropriately
    # or that XGBoost's internal handling is sufficient.
    # Consider explicit label encoding for categorical features if not already done and if XGBoost version requires it.

    X_train = train_df[predictor_cols]
    y_train = train_df[TARGET_COLUMN]
    # X_val = val_df[predictor_cols]
    # y_val = val_df[TARGET_COLUMN]
    X_test = test_df[predictor_cols]
    y_test = test_df[TARGET_COLUMN]

    # Scaling continuous features (optional but often good practice)
    # XGBoost is somewhat robust to feature scaling, but it doesn't hurt.
    # We will only scale based on train_df to prevent data leakage.
    # Identify continuous columns heuristically (non-object, non-int with many unique values)
    # This step might need refinement based on your actual feature types.
    
    # For now, let's assume all predictor_cols are continuous or appropriately encoded.
    # If you have explicit categorical columns that are not one-hot encoded, 
    # XGBoost might require them to be label encoded or of pandas 'category' dtype.
    # The current get_predictor_columns filters out month dummies, assuming other columns are numeric.

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    # X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print("Initializing and training XGBoost model...")
    # Basic XGBoost parameters - these can be tuned extensively
    model = xgb.XGBRegressor(
        objective='reg:squarederror', # for regression
        n_estimators=100,             # Number of boosting rounds
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        # tree_method='gpu_hist' if torch.cuda.is_available() else 'hist', # Use GPU if available
        # enable_categorical=True # If using pandas category dtypes for categoricals
        # For GPU, ensure XGBoost is compiled with GPU support. Might need 'cuda' for tree_method
    )

    # For early stopping, you would typically use an eval_set:
    # model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], early_stopping_rounds=10, verbose=True)
    model.fit(X_train_scaled, y_train, verbose=True) # Simpler fit for now

    print("Training finished.")

    print("Making predictions on the test set...")
    predictions = model.predict(X_test_scaled)

    oos_r2 = r2_score(y_test, predictions)
    oos_mse = mean_squared_error(y_test, predictions)

    print(f"Out-of-Sample R2 Score: {oos_r2:.6f}")
    print(f"Out-of-Sample MSE: {oos_mse:.6f}")

    print("Saving model, scaler, and predictions...")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

    predictions_df = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN], # Make sure test_df still has ID and Date
        DATE_COLUMN: test_df[DATE_COLUMN],
        'prediction': predictions
    })
    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")

    # --- Log Metrics ---
    metrics_to_log = {
        'out_of_sample_r2': oos_r2,
        'mse': oos_mse,
    }
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    if 'CSV_FILE' in globals():
        print(f"Metrics logged to {CSV_FILE}")
    else:
        print("Metrics logged (CSV_FILE path not found for message).")
    # --- End Log Metrics ---

    print(f"--- {MODEL_NAME} Model Script Complete ---")

if __name__ == '__main__':
    main() 