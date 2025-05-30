import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import csv # Added for CSV logging
from datetime import datetime # Added for timestamp

# --- Configuration ---
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')

MODEL_NAME = 'linear_regression_ols'
TARGET_COLUMN = 'stock_exret' # Assuming this is the primary target

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

def get_predictor_columns(df, target_col):
    """Helper function to identify numeric predictor columns, excluding target and identifiers."""
    identifier_cols = ['permno', 'stock_ticker', 'CUSIP', 'comp_name'] 
    # Also exclude other potential targets or date columns not used as direct features
    cols_to_exclude = identifier_cols + [target_col, 'ret_eom', 'date', 'year', 'month'] 
    # Include seasonality dummies if they exist and are desired as predictors
    # For now, assume all other numeric columns are predictors
    
    potential_predictor_cols = [col for col in df.columns if col not in cols_to_exclude]
    numeric_predictor_cols = df[potential_predictor_cols].select_dtypes(include=np.number).columns.tolist()
    print(f"Identified {len(numeric_predictor_cols)} numeric predictor columns for model training.")
    return numeric_predictor_cols

def main():
    print(f"--- Training {MODEL_NAME} Model ---")

    # --- Load Data ---
    print("Loading imputed datasets...")
    try:
        train_df = pd.read_parquet(TRAIN_IMPUTED_INPUT_PATH)
        validation_df = pd.read_parquet(VALIDATION_IMPUTED_INPUT_PATH)
        print("Datasets loaded successfully.")
        print(f"Train shape: {train_df.shape}, Validation shape: {validation_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Prepare X and y ---
    predictor_cols = get_predictor_columns(train_df, TARGET_COLUMN)
    if not predictor_cols:
        print("Error: No predictor columns identified. Aborting training.")
        return

    X_train = train_df[predictor_cols]
    y_train = train_df[TARGET_COLUMN]
    
    X_validation = validation_df[predictor_cols]
    y_validation = validation_df[TARGET_COLUMN]

    # Ensure no NaNs in target variable (should have been handled, but good check)
    if y_train.isnull().any() or y_validation.isnull().any():
        print("Warning: NaNs found in target column. Dropping rows with NaN target.")
        # This requires aligning X and y again if rows are dropped
        train_df_cleaned = train_df.dropna(subset=[TARGET_COLUMN])
        X_train = train_df_cleaned[predictor_cols]
        y_train = train_df_cleaned[TARGET_COLUMN]
        
        validation_df_cleaned = validation_df.dropna(subset=[TARGET_COLUMN])
        X_validation = validation_df_cleaned[predictor_cols]
        y_validation = validation_df_cleaned[TARGET_COLUMN]
        
        print(f"Shapes after NaN target drop: X_train: {X_train.shape}, X_val: {X_validation.shape}")

    # --- Check for NaNs in predictors AFTER final selection for the model ---
    if X_train.isnull().sum().sum() > 0:
        print("ERROR: NaNs found in X_train predictor columns before fitting model.")
        print("NaN counts per column in X_train:")
        print(X_train.isnull().sum()[X_train.isnull().sum() > 0])
        print("Filling these NaNs with 0 for now...")
        X_train = X_train.fillna(0)
    
    if X_validation.isnull().sum().sum() > 0:
        print("ERROR: NaNs found in X_validation predictor columns before fitting model.")
        print("NaN counts per column in X_validation:")
        print(X_validation.isnull().sum()[X_validation.isnull().sum() > 0])
        print("Filling these NaNs with 0 for now...")
        X_validation = X_validation.fillna(0)

    # --- Handle potential infinities ---
    # Replace inf with a large finite number or 0, and -inf with a small finite number or 0
    # Using np.nan_to_num which also handles NaNs if any slipped through, though fillna(0) should have caught them.
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e9, neginf=-1e9) # Arbitrary large numbers for inf
    X_validation = np.nan_to_num(X_validation, nan=0.0, posinf=1e9, neginf=-1e9)
    print("Applied np.nan_to_num to handle NaNs and infinities in predictors.")

    # --- Train Model ---
    print("\nTraining OLS model...")
    model = LinearRegression()
    try:
        model.fit(X_train, y_train)
        print("Model training complete.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # --- Make Predictions ---
    print("\nMaking predictions on validation set...")
    try:
        y_pred_validation = model.predict(X_validation)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # --- Evaluate Model ---
    print("\nEvaluating model performance...")
    oos_r2 = r2_score(y_validation, y_pred_validation)
    mse = mean_squared_error(y_validation, y_pred_validation)

    print(f"Out-of-Sample (OOS) R-squared on Validation: {oos_r2:.4f}")
    print(f"Mean Squared Error (MSE) on Validation: {mse:.6f}")

    # --- Save Model ---
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}.joblib")
    try:
        joblib.dump(model, model_save_path)
        print(f"Model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # --- Save Predictions ---
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    predictions_df = validation_df[['permno', 'date']].copy() # Include year, month if needed from 'date'
    predictions_df['actual_' + TARGET_COLUMN] = y_validation.values # Use .values to avoid index issues if y_validation was modified
    predictions_df['predicted_' + TARGET_COLUMN] = y_pred_validation
    
    predictions_save_path = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_validation_predictions.parquet")
    try:
        predictions_df.to_parquet(predictions_save_path, index=False)
        print(f"Validation predictions saved to {predictions_save_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        
    # --- Log Metrics ---
    metrics_to_log = {
        'oos_r2': oos_r2,
        'mse': mse,
        # Add any other metrics you want to log, e.g., training_time
    }
    if 'log_metrics_to_csv' in globals(): # Check if function is defined
        log_metrics_to_csv(MODEL_NAME, metrics_to_log)
        if 'CSV_FILE' in globals():
            print(f"Metrics logged to {CSV_FILE}")
        else:
            print("Metrics logged (CSV_FILE path not found for message).")
    else:
        print("log_metrics_to_csv function not found. Skipping CSV logging.")
    # --- End Log Metrics ---

    print(f"\n--- {MODEL_NAME} Model Script Complete ---")

if __name__ == "__main__":
    main() 