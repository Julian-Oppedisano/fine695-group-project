import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_model.joblib')
SCALER_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_scaler.joblib')
LABEL_ENCODERS_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_label_encoders.joblib') # For explicit categoricals
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
                     'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12',
                     'SHRCD']
    
    # Start with all columns
    predictor_cols = df.columns.tolist()
    
    # Remove explicitly excluded columns
    predictor_cols = [col for col in predictor_cols if col not in excluded_cols]
    
    # Remove columns starting with 'month_' or 'eps_'
    predictor_cols = [col for col in predictor_cols if not (col.startswith('month_') or col.startswith('eps_'))]
    
    return predictor_cols

def main():
    print(f"--- Training {MODEL_NAME} Model ---")

    print("Loading imputed datasets...")
    train_df = pd.read_parquet(TRAIN_IMPUTED_INPUT_PATH)
    val_df = pd.read_parquet(VALIDATION_IMPUTED_INPUT_PATH)
    test_df = pd.read_parquet(TEST_IMPUTED_INPUT_PATH)

    print(f"Train df shape: {train_df.shape}")
    print(f"Validation df shape: {val_df.shape}")
    print(f"Test df shape: {test_df.shape}")

    for df_name, df_content in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        if TARGET_COLUMN not in df_content.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {df_name}_df.")
        if df_content[TARGET_COLUMN].isnull().all():
            if df_name in ['train', 'validation']:
                raise ValueError(f"Target column '{TARGET_COLUMN}' in {df_name}_df is all NaNs.")
            else:
                print(f"Warning: Target column '{TARGET_COLUMN}' in {df_name}_df is all NaNs.")

    predictor_cols = get_predictor_columns(train_df)
    if TARGET_COLUMN in predictor_cols:
        print(f"Warning: Target column '{TARGET_COLUMN}' was found in predictor_cols. Removing it.")
        predictor_cols.remove(TARGET_COLUMN)
    
    X_train = train_df[predictor_cols].copy()
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[predictor_cols].copy()
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df[predictor_cols].copy()
    y_test = test_df[TARGET_COLUMN]

    # --- Start y_train Diagnostics ---
    print("\n--- y_train Diagnostics ---")
    print(y_train.describe())
    print(f"Number of unique values in y_train: {y_train.nunique()}")
    print("--- End y_train Diagnostics ---\n")
    # --- End y_train diagnostics ---

    # --- Preprocessing for specific categorical columns like 'size_port' ---
    categorical_to_encode = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            categorical_to_encode.append(col)
            print(f"Identified column '{col}' for label encoding.")

    label_encoders = {}
    for col in categorical_to_encode:
        print(f"Label encoding column: {col}")
        # Fill NaN/None with a placeholder string before encoding
        X_train[col] = X_train[col].fillna('MISSING').astype(str)
        X_val[col] = X_val[col].fillna('MISSING').astype(str)
        X_test[col] = X_test[col].fillna('MISSING').astype(str)
        
        le = LabelEncoder()
        # Fit on combined unique values from train, validation, and test to ensure consistency
        all_unique_values = pd.concat([X_train[col], X_val[col], X_test[col]]).unique()
        le.fit(all_unique_values)
        
        X_train[col] = le.transform(X_train[col])
        X_val[col] = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
    # --- End Categorical Preprocessing ---

    # Identify columns for scaling (all predictor_cols MINUS the categoricals we just label encoded)
    cols_to_scale = [col for col in predictor_cols if col not in categorical_to_encode]

    if cols_to_scale:
        print(f"Scaling features: {cols_to_scale}")
        # Replace infinities with NaN, then fill NaNs with 0 before scaling
        X_train[cols_to_scale] = X_train[cols_to_scale].replace([np.inf, -np.inf], np.nan)
        X_val[cols_to_scale] = X_val[cols_to_scale].replace([np.inf, -np.inf], np.nan)
        X_test[cols_to_scale] = X_test[cols_to_scale].replace([np.inf, -np.inf], np.nan)
        
        # Fill any NaNs (including those from replaced infinities) with 0
        X_train[cols_to_scale] = X_train[cols_to_scale].fillna(0)
        X_val[cols_to_scale] = X_val[cols_to_scale].fillna(0)
        X_test[cols_to_scale] = X_test[cols_to_scale].fillna(0)

        scaler = StandardScaler()
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    else:
        print("No columns identified for scaling.")
        scaler = None

    # --- Start X_train Diagnostics (after preprocessing) ---
    print("\n--- X_train Diagnostics (after preprocessing) ---")
    low_variance_cols = X_train.nunique() == 1
    print(f"Number of columns in X_train with only 1 unique value: {low_variance_cols.sum()}")
    if low_variance_cols.sum() > 0:
        print(f"Columns with 1 unique value: {X_train.columns[low_variance_cols].tolist()}")
    print("--- End X_train Diagnostics ---\n")
    # --- End X_train diagnostics ---

    print("Initializing and training XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]

    model.fit(X_train, y_train, eval_set=eval_set, verbose=100)

    print("Training finished.")

    print("Making predictions on the test set...")
    predictions = model.predict(X_test)

    oos_r2 = r2_score(y_test, predictions)
    oos_mse = mean_squared_error(y_test, predictions)

    print(f"Out-of-Sample R2 Score: {oos_r2:.6f}")
    print(f"Out-of-Sample MSE: {oos_mse:.6f}")

    print("Saving model, scaler, and label encoders...")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    if scaler:
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler saved to {SCALER_PATH}")
    if label_encoders:
        joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
        print(f"Label encoders saved to {LABEL_ENCODERS_PATH}")
    print(f"Model saved to {MODEL_PATH}")

    predictions_df = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        DATE_COLUMN: test_df[DATE_COLUMN],
        'prediction': predictions
    })
    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")

    metrics_to_log = {
        'out_of_sample_r2': oos_r2,
        'mse': oos_mse,
    }
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")

    print(f"--- {MODEL_NAME} Model Script Complete ---")

if __name__ == '__main__':
    main() 