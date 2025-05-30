import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# --- Configuration ---
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')

MODEL_NAME = 'lightgbm' # Can add tuning info later if needed, e.g., lightgbm_tuned
TARGET_COLUMN = 'stock_exret'

def get_predictor_columns(df, target_col):
    identifier_cols = ['permno', 'stock_ticker', 'CUSIP', 'comp_name']
    cols_to_exclude = identifier_cols + [target_col, 'ret_eom', 'date', 'year', 'month']
    potential_predictor_cols = [col for col in df.columns if col not in cols_to_exclude]
    numeric_predictor_cols = df[potential_predictor_cols].select_dtypes(include=np.number).columns.tolist()
    print(f"Identified {len(numeric_predictor_cols)} numeric predictor columns for model training.")
    return numeric_predictor_cols

def main():
    print(f"--- Training {MODEL_NAME} Model ---")

    print("Loading imputed datasets...")
    try:
        train_df = pd.read_parquet(TRAIN_IMPUTED_INPUT_PATH)
        validation_df = pd.read_parquet(VALIDATION_IMPUTED_INPUT_PATH)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}"); return

    predictor_cols = get_predictor_columns(train_df, TARGET_COLUMN)
    if not predictor_cols: print("Error: No predictors. Aborting."); return

    X_train = train_df[predictor_cols]
    y_train = train_df[TARGET_COLUMN]
    X_validation = validation_df[predictor_cols]
    y_validation = validation_df[TARGET_COLUMN]

    # Handle NaNs in target
    if y_train.isnull().any():
        print("Warning: NaNs in y_train. Dropping.")
        train_df_cleaned = train_df.dropna(subset=[TARGET_COLUMN])
        X_train = train_df_cleaned[predictor_cols]
        y_train = train_df_cleaned[TARGET_COLUMN]
    if y_validation.isnull().any():
        print("Warning: NaNs in y_validation. Dropping.")
        validation_df_cleaned = validation_df.dropna(subset=[TARGET_COLUMN])
        X_validation = validation_df_cleaned[predictor_cols]
        y_validation = validation_df_cleaned[TARGET_COLUMN]

    # Handle NaNs and infinities in predictors (as done for OLS/ElasticNet)
    # LightGBM can handle NaNs natively if configured, but for consistency with other models
    # and to ensure our preprocessed data is used as intended, we'll clean them.
    X_train = X_train.fillna(0)
    X_validation = X_validation.fillna(0)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e9, neginf=-1e9)
    X_validation = np.nan_to_num(X_validation, nan=0.0, posinf=1e9, neginf=-1e9)
    print("Filled NaNs with 0 and handled infinities in predictors.")

    # --- LightGBM Model Training (with basic parameters and early stopping) ---
    # For a more thorough tuning, a grid search or Bayesian optimization (e.g., with Optuna) would be used.
    # Here, we set some reasonable defaults and use early stopping based on validation performance.
    print("\nTraining LightGBM model...")
    
    lgb_params = {
        'objective': 'regression_l1', # L1 loss (MAE), often better for noisy financial data
        'metric': 'mae',             # Evaluation metric for early stopping
        'n_estimators': 1000,       # Max number of trees, will be cut by early stopping
        'learning_rate': 0.05,      # Typical learning rate
        'num_leaves': 31,           # Default, balance between speed and accuracy
        'max_depth': -1,            # No limit on depth
        'min_child_samples': 20,    # Default
        'subsample': 0.8,           # Row subsampling
        'colsample_bytree': 0.8,    # Feature subsampling
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,              # Suppress LightGBM's own verbosity
        'boosting_type': 'gbdt'
    }

    model = lgb.LGBMRegressor(**lgb_params)

    try:
        model.fit(X_train, y_train,
                  eval_set=[(X_validation, y_validation)],
                  eval_metric='mae',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=-1)])
        print("LightGBM training complete.")
        print(f"Best iteration: {model.best_iteration_}")
    except Exception as e:
        print(f"Error during LightGBM training: {e}"); return

    # --- Make Predictions ---
    print("\nMaking predictions on validation set...")
    # Predictions are made using the model at its best iteration (due to early stopping)
    y_pred_validation = model.predict(X_validation, num_iteration=model.best_iteration_)

    # --- Evaluate Model ---
    print("\nEvaluating model performance...")
    oos_r2 = r2_score(y_validation, y_pred_validation)
    mse = mean_squared_error(y_validation, y_pred_validation)
    print(f"Out-of-Sample (OOS) R-squared on Validation: {oos_r2:.4f}")
    print(f"Mean Squared Error (MSE) on Validation: {mse:.6f}")

    # --- Save Model ---
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}.joblib") # Using joblib for consistency
    joblib.dump(model, model_save_path)
    print(f"LightGBM model saved to {model_save_path}")

    # --- Save Predictions ---
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    predictions_df = validation_df[['permno', 'date']].copy()
    predictions_df['actual_' + TARGET_COLUMN] = y_validation.values
    predictions_df['predicted_' + TARGET_COLUMN] = y_pred_validation
    predictions_save_path = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_validation_predictions.parquet")
    predictions_df.to_parquet(predictions_save_path, index=False)
    print(f"Validation predictions saved to {predictions_save_path}")
        
    print(f"\n--- {MODEL_NAME} Model Script Complete ---")

if __name__ == "__main__":
    main() 