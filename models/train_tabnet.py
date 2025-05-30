import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler # Optional, but can help
import torch # For device management
import joblib # For saving scaler, model can be saved with its own method or joblib
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import shutil # For cleaning up tensorboard logs
import csv # Added for CSV logging
from datetime import datetime # Added for timestamp

# --- Configuration ---
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')

MODEL_NAME = 'tabnet'
TARGET_COLUMN = 'stock_exret'

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
    identifier_cols = ['permno', 'stock_ticker', 'CUSIP', 'comp_name']
    cols_to_exclude = identifier_cols + [target_col, 'ret_eom', 'date', 'year', 'month']
    potential_predictor_cols = [col for col in df.columns if col not in cols_to_exclude]
    # TabNet can handle categorical features, but for now, treat all as numeric
    # and rely on its internal handling or prior normalization.
    numeric_predictor_cols = df[potential_predictor_cols].select_dtypes(include=np.number).columns.tolist()
    print(f"Identified {len(numeric_predictor_cols)} numeric predictor columns for TabNet.")
    return numeric_predictor_cols

def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading imputed datasets...")
    try:
        train_df = pd.read_parquet(TRAIN_IMPUTED_INPUT_PATH)
        validation_df = pd.read_parquet(VALIDATION_IMPUTED_INPUT_PATH)
        print("Datasets loaded successfully.")
    except Exception as e: print(f"Error loading data: {e}"); return

    predictor_cols = get_predictor_columns(train_df, TARGET_COLUMN)
    if not predictor_cols: print("Error: No predictors. Aborting."); return

    X_train = train_df[predictor_cols].copy()
    y_train = train_df[[TARGET_COLUMN]].copy() # TabNet expects y to be 2D (n_samples, n_tasks)
    X_validation = validation_df[predictor_cols].copy()
    y_validation = validation_df[[TARGET_COLUMN]].copy()

    # Handle NaNs in target
    if y_train[TARGET_COLUMN].isnull().any():
        print("Warning: NaNs in y_train. Dropping corresponding rows in X_train and y_train.")
        train_valid_target_idx = y_train[TARGET_COLUMN].notnull()
        X_train = X_train[train_valid_target_idx]
        y_train = y_train[train_valid_target_idx]
    if y_validation[TARGET_COLUMN].isnull().any():
        print("Warning: NaNs in y_validation. Dropping corresponding rows in X_validation and y_validation.")
        val_valid_target_idx = y_validation[TARGET_COLUMN].notnull()
        X_validation = X_validation[val_valid_target_idx]
        y_validation = y_validation[val_valid_target_idx]

    # Convert to NumPy arrays as expected by TabNet
    X_train_np = X_train.fillna(0).to_numpy() # Fill NaNs just in case, then convert
    y_train_np = y_train.to_numpy()
    X_validation_np = X_validation.fillna(0).to_numpy()
    y_validation_np = y_validation.to_numpy() # This is for evaluation, y_validation for fit needs to be 2D

    # Handle potential infinities after fillna(0) (though unlikely if data is from imputation script)
    X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=1e9, neginf=-1e9)
    X_validation_np = np.nan_to_num(X_validation_np, nan=0.0, posinf=1e9, neginf=-1e9)
    print("Converted data to NumPy, filled NaNs with 0, and handled infinities.")

    # --- TabNet Model Training ---
    print("\nTraining TabNetRegressor model...")
    # TabNet parameters - these often require tuning
    tabnet_params = dict(
        n_d=32, n_a=32,       # Width of the decision prediction layer and attention layer
        n_steps=3,            # Number of sequential attention steps
        gamma=1.3,            # Coefficient for feature reusage in subsequent steps
        lambda_sparse=1e-3,   # Sparsity regularization coefficient
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(mode="min", patience=10, min_lr=1e-5, factor=0.5), # Learning rate scheduler
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='sparsemax', # 'entmax' or 'sparsemax'
        verbose=10,            # TabNet verbosity for fitting, 0 for less, 1 for more
        device_name=device
    )
    model = TabNetRegressor(**tabnet_params)

    try:
        model.fit(
            X_train_np, y_train_np,
            eval_set=[(X_validation_np, y_validation_np)],
            eval_metric=['mae'], # Mean Absolute Error, or 'mse', 'rmse'
            max_epochs=100,       # Max number of epochs for training
            patience=20,          # Early stopping patience
            batch_size=1024*4,    # Adjust based on memory
            virtual_batch_size=128*4 # Adjust based on memory
        )
        print("TabNet training complete.")
    except Exception as e:
        print(f"Error during TabNet training: {e}"); return

    # --- Make Predictions ---
    print("\nMaking predictions on validation set...")
    y_pred_validation = model.predict(X_validation_np)

    # --- Evaluate Model ---
    print("\nEvaluating model performance...")
    # y_validation_np is already (n_samples, 1), y_pred_validation is also (n_samples, 1)
    oos_r2 = r2_score(y_validation_np, y_pred_validation)
    mse = mean_squared_error(y_validation_np, y_pred_validation)
    print(f"Out-of-Sample (OOS) R-squared on Validation: {oos_r2:.4f}")
    print(f"Mean Squared Error (MSE) on Validation: {mse:.6f}")

    # --- Save Model ---
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    # TabNet models are typically saved via a .zip file that includes model architecture and weights
    model_save_path_zip = os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}_model") # Will append .zip automatically
    model.save_model(model_save_path_zip)
    print(f"TabNet model saved to {model_save_path_zip}.zip")

    # --- Save Predictions ---
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    # Create a DataFrame for predictions using original validation_df for permno and date
    # Ensure y_validation (original Series/DataFrame column) and y_pred_validation (numpy array) align
    predictions_df = validation_df[['permno', 'date']].iloc[y_validation.index] # Align if rows were dropped from y_validation
    predictions_df['actual_' + TARGET_COLUMN] = y_validation_np.flatten()
    predictions_df['predicted_' + TARGET_COLUMN] = y_pred_validation.flatten()
    predictions_save_path = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_validation_predictions.parquet")
    predictions_df.to_parquet(predictions_save_path, index=False)
    print(f"Validation predictions saved to {predictions_save_path}")
        
    # --- Log Metrics ---
    metrics_to_log = {
        'oos_r2': oos_r2, # Assuming test set is the hold-out validation
        'mse': mse,
    }
    if 'log_metrics_to_csv' in globals() and 'MODEL_NAME' in globals():
        log_metrics_to_csv(MODEL_NAME, metrics_to_log)
        if 'CSV_FILE' in globals():
            print(f"Metrics logged to {CSV_FILE}")
        else:
            print("Metrics logged (CSV_FILE path not found for message).")
    else:
        print("log_metrics_to_csv function or MODEL_NAME not found. Skipping CSV logging.")
    # --- End Log Metrics ---

    print(f"\n--- {MODEL_NAME} Model Script Complete ---")

if __name__ == "__main__":
    main() 