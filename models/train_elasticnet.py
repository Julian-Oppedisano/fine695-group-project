import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- Configuration ---
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')

MODEL_NAME = 'elasticnet_cv'
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

    X_train_raw = train_df[predictor_cols]
    y_train = train_df[TARGET_COLUMN]
    X_validation_raw = validation_df[predictor_cols]
    y_validation = validation_df[TARGET_COLUMN]

    # Handle NaNs in target
    if y_train.isnull().any():
        print("Warning: NaNs in y_train. Dropping.")
        train_df_cleaned = train_df.dropna(subset=[TARGET_COLUMN])
        X_train_raw = train_df_cleaned[predictor_cols]
        y_train = train_df_cleaned[TARGET_COLUMN]
    if y_validation.isnull().any():
        print("Warning: NaNs in y_validation. Dropping.")
        validation_df_cleaned = validation_df.dropna(subset=[TARGET_COLUMN])
        X_validation_raw = validation_df_cleaned[predictor_cols]
        y_validation = validation_df_cleaned[TARGET_COLUMN]

    # Handle NaNs and infinities in predictors (as done for OLS)
    X_train_raw = X_train_raw.fillna(0)
    X_validation_raw = X_validation_raw.fillna(0)
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=1e9, neginf=-1e9)
    X_validation_raw = np.nan_to_num(X_validation_raw, nan=0.0, posinf=1e9, neginf=-1e9)
    print("Filled NaNs with 0 and handled infinities in predictors.")

    # --- Feature Scaling (Important for Regularized Models) ---
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_validation_scaled = scaler.transform(X_validation_raw)
    # Save the scaler
    scaler_path = os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # --- Train ElasticNetCV Model ---
    # ElasticNetCV performs cross-validation to find the best alpha and l1_ratio.
    # We are using the full training set here to find these params, then fitting on it.
    # For financial data, time-series CV (e.g., TimeSeriesSplit) is often preferred.
    # Here, using default CV (k-fold) for simplicity as per typical sklearn examples.
    # Consider TimeSeriesSplit for more rigorous financial CV.
    print("\nTraining ElasticNetCV model (tuning alpha and l1_ratio)...")
    # Define l1_ratios to search. Common values.
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0] # 1.0 is Lasso, close to 0 is Ridge
    # n_alphas: number of alphas along the regularization path
    # cv: number of folds for cross-validation
    # max_iter: to ensure convergence, especially with many features
    # n_jobs=-1: use all available CPU cores
    elastic_cv_model = ElasticNetCV(
        l1_ratio=l1_ratios,
        n_alphas=100, 
        cv=3, # Using 3 folds for speed. Consider 5 for more robustness.
        random_state=42,
        max_iter=2000, # Increased max_iter
        n_jobs=-1,
        selection='cyclic' # Default, can be 'random' for faster on large datasets
    )
    
    try:
        elastic_cv_model.fit(X_train_scaled, y_train)
        print("ElasticNetCV training complete.")
        print(f"Best alpha: {elastic_cv_model.alpha_:.6f}")
        print(f"Best l1_ratio: {elastic_cv_model.l1_ratio_:.2f}")
    except Exception as e:
        print(f"Error during ElasticNetCV training: {e}"); return

    # --- Make Predictions using the best estimator from CV ---
    print("\nMaking predictions on scaled validation set...")
    y_pred_validation = elastic_cv_model.predict(X_validation_scaled)

    # --- Evaluate Model ---
    print("\nEvaluating model performance...")
    oos_r2 = r2_score(y_validation, y_pred_validation)
    mse = mean_squared_error(y_validation, y_pred_validation)
    print(f"Out-of-Sample (OOS) R-squared on Validation: {oos_r2:.4f}")
    print(f"Mean Squared Error (MSE) on Validation: {mse:.6f}")
    num_selected_coeffs = np.sum(elastic_cv_model.coef_ != 0)
    print(f"Number of selected coefficients: {num_selected_coeffs} out of {X_train_raw.shape[1]}")

    # --- Save Model (The best estimator found by CV) ---
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}.joblib")
    joblib.dump(elastic_cv_model, model_save_path)
    print(f"ElasticNetCV model saved to {model_save_path}")

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