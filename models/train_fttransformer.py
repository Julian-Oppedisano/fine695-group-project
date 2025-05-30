import pandas as pd
import numpy as np
import torch
import os
import csv
from datetime import datetime
import typing

# Add this to handle PyTorch 2.6+ loading issues with omegaconf
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata
if hasattr(torch.serialization, 'add_safe_globals') and callable(torch.serialization.add_safe_globals):
    torch.serialization.add_safe_globals([DictConfig, ContainerMetadata, typing.Any])

from pytorch_tabular.tabular_model import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig, ExperimentConfig
from pytorch_tabular.feature_extractor import DeepFeatureExtractor # Changed from FeatureExtractor
from pytorch_tabular.models.common.heads.config import LinearHeadConfig # Added for head configuration

from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration ---
MODEL_NAME = "FTTransformer"
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet') # Added for test set evaluation

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')

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

def get_raw_predictor_columns(df): # Renamed to avoid confusion with Pytorch Tabular's own feature handling
    excluded_cols = [TARGET_COLUMN, DATE_COLUMN, ID_COLUMN, 'year', 'month', 'day', 
                     'eom_date', 'size_class', 'comb_code', 'month_num',
                     'stock_exret_t_plus_1', 'stock_exret_t_plus_2', 
                     'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12']
    # Do not exclude month_ dummies here, let Pytorch Tabular decide if they are cat or cont
    predictor_cols = [col for col in df.columns if col not in excluded_cols]
    return predictor_cols

def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    accelerator_setting = 'gpu' if device == 'cuda' else device # Map 'cuda' to 'gpu' for accelerator

    print("Loading imputed datasets...")
    try:
        train_df = pd.read_parquet(TRAIN_IMPUTED_INPUT_PATH)
        validation_df = pd.read_parquet(VALIDATION_IMPUTED_INPUT_PATH)
        test_df = pd.read_parquet(TEST_IMPUTED_INPUT_PATH) # Load test_df
        print("Datasets loaded successfully.")
    except Exception as e: 
        print(f"Error loading data: {e}"); return

    # Get initial list of potential predictors
    raw_predictor_cols = get_raw_predictor_columns(train_df)

    # Prepare data for PyTorch Tabular (it expects all relevant columns in the DataFrame)
    # PyTorch Tabular will internally identify categorical and continuous features
    
    # Ensure target is not all NaNs and handle NaNs in target for train and validation
    for df_name, df_content in [('train', train_df), ('validation', validation_df), ('test', test_df)]:
        if df_content[TARGET_COLUMN].isnull().all():
            raise ValueError(f"Target column '{TARGET_COLUMN}' in {df_name}_df is all NaNs.")
        if df_name != 'test': # Only drop from train/val, test target NaNs will be handled by predict or during eval
            if df_content[TARGET_COLUMN].isnull().any():
                print(f"Warning: NaNs in y_{df_name}. Dropping corresponding rows.")
                valid_target_idx = df_content[TARGET_COLUMN].notnull()
                if df_name == 'train': train_df = train_df[valid_target_idx]
                if df_name == 'validation': validation_df = validation_df[valid_target_idx]
    
    # Impute NaNs and infinities in predictor columns (PyTorch Tabular prefers this)
    print("Imputing NaNs and infinities in predictor columns...")
    for col in raw_predictor_cols:
        for df_ref in [train_df, validation_df, test_df]:
            if df_ref[col].dtype == 'object':
                df_ref[col] = df_ref[col].fillna('MISSING').astype(str) # Ensure objects are strings
            else:
                # For numerical columns, replace inf with NaN, then fill NaN with 0
                df_ref[col] = df_ref[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- PyTorch Tabular Configs ---
    # Let DataConfig infer categorical and continuous columns.
    # We will provide all raw_predictor_cols initially as continuous and let it adjust.
    # Object columns will be treated as categorical by default by Pytorch Tabular if not specified.
    
    # Identify potential categorical columns based on dtype for initial guidance (optional, but can help)
    # Pytorch Tabular will make the final decision based on cardinalities etc.
    potential_categorical_cols = [col for col in raw_predictor_cols if train_df[col].dtype == 'object']
    potential_continuous_cols = [col for col in raw_predictor_cols if col not in potential_categorical_cols]

    print(f"Identified potential categorical columns (dtype object): {potential_categorical_cols}")
    print(f"Identified potential continuous columns (other dtypes): {potential_continuous_cols}")

    data_config = DataConfig(
        target=[TARGET_COLUMN],
        continuous_cols=potential_continuous_cols, # Pass all non-object columns as continuous initially
        categorical_cols=potential_categorical_cols, # Pass object columns as categorical
        # PyTorch Tabular will also apply its own logic for type detection and cardinality checks.
        # num_workers=4 # Adjust based on your system
    )

    trainer_config = TrainerConfig(
        auto_lr_find=False, # Can be set to True to find optimal LR
        batch_size=1024, # Increased batch size
        max_epochs=5, # Reduced for first run, was 100
        accelerator=accelerator_setting, # Use the mapped setting
        devices=1 if device == 'cuda' else 1, # Changed 'auto' to 1 for CPU
        # gpus=1 if device == 'cuda' else 0, # Deprecated
        # early_stopping_patience=10, # From Pytorch Tabular config
        # check_val_every_n_epoch=1,
    )

    optimizer_config = OptimizerConfig() # Use defaults for Adam

    # FTTransformer specific config
    # Refer to FTTransformerConfig documentation for all parameters

    # Define the head_config for the output feed-forward layers
    head_config_params = {
        "layers": "128-64",  # Was out_ff_layers
        "activation": "ReLU", # Was out_ff_activation
        "dropout": 0.1,       # Was out_ff_dropout
        "use_batch_norm": False # Default for LinearHeadConfig, can be True if needed
    }

    model_config = FTTransformerConfig(
        task="regression",
        # Common FT-Transformer parameters:
        input_embed_dim=32,      # Dimension of embeddings
        embedding_dropout=0.1,
        # For categorical features (if any identified by Pytorch Tabular)
        # share_embedding = True, 
        # share_embedding_strategy = "add",
        # shared_embedding_fraction = 0.25,
        # Transformer params:
        num_attn_blocks=3,       # Number of attention blocks (like depth)
        num_heads=4,             # Number of attention heads
        transformer_activation="GEGLU", # GELU, ReLU, LeakyReLU, GEGLU, ReGLU
        ff_dropout=0.1,
        attn_dropout=0.1,
        add_norm_dropout=0.1,
        # Head params - now moved to head_config
        head="LinearHead", # Explicitly state LinearHead, though it's default
        head_config=head_config_params,
        # General params
        batch_norm_continuous_input=True,
        learning_rate=1e-4, # Common learning rate, adjust as needed
        loss="MSELoss" # For regression
    )

    experiment_config = ExperimentConfig(
        project_name="fine695_fttransformer", 
        run_name=MODEL_NAME,
        log_target="tensorboard", # or wandb
        exp_watch="gradients", # "gradients", "parameters", "all", or None
        log_logits=False,
        # checkpoint_monitor="valid_loss" # Pytorch Tabular uses valid_loss or a metric
    )
    
    # Initialize TabularModel
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        experiment_config=experiment_config,
        # model_name=MODEL_NAME, # Not a param here
        # verbose=False # Control verbosity from TrainerConfig or ExperimentConfig
    )

    print("\nFitting the TabularModel (FTTransformer)...")
    try:
        tabular_model.fit(train=train_df, validation=validation_df)
        print("FTTransformer training complete.")
    except Exception as e:
        print(f"Error during FTTransformer training: {e}"); return

    # --- Evaluate Model on Test Set ---
    print("\nEvaluating model performance on Test Set...")
    test_results = tabular_model.evaluate(test_df, verbose=False) # test_df should have target for evaluation
    oos_r2 = test_results[0].get(f"{MODEL_NAME}_test_r2_score", np.nan) # Metric name might vary
    mse = test_results[0].get(f"{MODEL_NAME}_test_mean_squared_error", np.nan)
    
    # Fallback if specific metric names are not found (Pytorch Tabular might use generic names)
    if np.isnan(oos_r2) and 'test_r2_score' in test_results[0]:
        oos_r2 = test_results[0]['test_r2_score']
    if np.isnan(mse) and 'test_mean_squared_error' in test_results[0]:
         mse = test_results[0]['test_mean_squared_error']
    if np.isnan(mse) and 'test_loss' in test_results[0]: # If MSE not directly there, use test_loss if it is MSE
        if model_config.loss == "MSELoss":
            mse = test_results[0]['test_loss']

    print(f"Out-of-Sample (OOS) R-squared on Test Set: {oos_r2:.6f}")
    print(f"Mean Squared Error (MSE) on Test Set: {mse:.6f}")

    # --- Make Predictions for Saving (if needed) ---
    print("\nMaking predictions on Test Set for saving...")
    # Ensure test_df for prediction does not have the target column if predict method requires it
    # However, Pytorch Tabular's predict usually handles it or can take df with target.
    predictions_test_df = tabular_model.predict(test_df, verbose=False)
    # The output is a DataFrame with predictions, usually named <TARGET_COLUMN>_prediction
    final_predictions = predictions_test_df[f'{TARGET_COLUMN}_prediction'].values

    # --- Save Model ---
    print("Saving FTTransformer model...")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    # Pytorch Tabular saves the whole experiment run directory
    # For a single model file, you might need to look into tabular_model.save_weights or similar
    # or save the specific checkpoint from the experiment directory.
    # For now, we'll save the experiment directory path conceptually.
    model_save_path = tabular_model.config_fitted.output_dir # This dir contains model, configs, etc.
    print(f"FTTransformer experiment saved to: {model_save_path}")
    # To load: TabularModel.load_from_checkpoint(path_to_checkpoint_in_model_save_path)
    # Or TabularModel.load_model(model_save_path) if it saves a specific model file we identify.
    # Let's also save the best_model_path specifically if available
    if hasattr(tabular_model.trainer, 'checkpoint_callback') and tabular_model.trainer.checkpoint_callback:
        best_ckpt_path = tabular_model.trainer.checkpoint_callback.best_model_path
        print(f"Best checkpoint path: {best_ckpt_path}")
        # We can copy this best_ckpt_path to our SAVED_MODEL_DIR with a standard name if needed
        # For now, the user should note this path or the experiment dir.

    # --- Save Predictions ---
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    # Re-attach permno and date for predictions file
    # Ensure test_df used here has original permno and date, and is in the same order as predictions
    # The test_df passed to predict should be the original one for ID purposes
    predictions_output_df = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        DATE_COLUMN: test_df[DATE_COLUMN],
        'prediction': final_predictions
    })
    predictions_save_path = os.path.join(PREDICTIONS_DIR, f"predictions_{MODEL_NAME.lower()}.parquet")
    predictions_output_df.to_parquet(predictions_save_path, index=False)
    print(f"Test predictions saved to {predictions_save_path}")
        
    # --- Log Metrics ---
    metrics_to_log = {
        'out_of_sample_r2': oos_r2,
        'mse': mse,
    }
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")
    # --- End Log Metrics ---

    print(f"\n--- {MODEL_NAME} Model Script Complete ---")

if __name__ == "__main__":
    main() 