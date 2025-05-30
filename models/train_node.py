import pandas as pd
import numpy as np
import torch
import os
import csv
from datetime import datetime
import typing
from collections import defaultdict

# Add this to handle PyTorch 2.6+ loading issues with omegaconf
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata
if hasattr(torch.serialization, 'add_safe_globals') and callable(torch.serialization.add_safe_globals):
    torch.serialization.add_safe_globals([DictConfig, ContainerMetadata, typing.Any, dict, defaultdict])

from pytorch_tabular.tabular_model import TabularModel
from pytorch_tabular.models import NodeConfig # Changed from FTTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig, ExperimentConfig
# from pytorch_tabular.feature_extractor import DeepFeatureExtractor # Likely not needed for NODE in this basic setup
from pytorch_tabular.models.common.heads.config import LinearHeadConfig # May not be explicitly used but kept for now

from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration ---
MODEL_NAME = "NODE" # Changed
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

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

def get_raw_predictor_columns(df):
    excluded_cols = [TARGET_COLUMN, DATE_COLUMN, ID_COLUMN, 'year', 'month', 'day', 
                     'eom_date', 'size_class', 'comb_code', 'month_num',
                     'stock_exret_t_plus_1', 'stock_exret_t_plus_2', 
                     'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12']
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
        test_df = pd.read_parquet(TEST_IMPUTED_INPUT_PATH)
        print("Datasets loaded successfully.")
    except Exception as e: 
        print(f"Error loading data: {e}"); return

    raw_predictor_cols = get_raw_predictor_columns(train_df)

    for df_name, df_content in [('train', train_df), ('validation', validation_df), ('test', test_df)]:
        if df_content[TARGET_COLUMN].isnull().all():
            raise ValueError(f"Target column '{TARGET_COLUMN}' in {df_name}_df is all NaNs.")
        if df_name != 'test':
            if df_content[TARGET_COLUMN].isnull().any():
                print(f"Warning: NaNs in y_{df_name}. Dropping corresponding rows.")
                valid_target_idx = df_content[TARGET_COLUMN].notnull()
                if df_name == 'train': train_df = train_df[valid_target_idx]
                if df_name == 'validation': validation_df = validation_df[valid_target_idx]
    
    print("Imputing NaNs and infinities in predictor columns...")
    for col in raw_predictor_cols:
        for df_ref in [train_df, validation_df, test_df]:
            if df_ref[col].dtype == 'object':
                df_ref[col] = df_ref[col].fillna('MISSING').astype(str)
            else:
                df_ref[col] = df_ref[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    potential_categorical_cols = [col for col in raw_predictor_cols if train_df[col].dtype == 'object']
    potential_continuous_cols = [col for col in raw_predictor_cols if col not in potential_categorical_cols]

    print(f"Identified potential categorical columns (dtype object): {potential_categorical_cols}")
    print(f"Identified potential continuous columns (other dtypes): {potential_continuous_cols}")

    data_config = DataConfig(
        target=[TARGET_COLUMN],
        continuous_cols=potential_continuous_cols,
        categorical_cols=potential_categorical_cols,
    )

    trainer_config = TrainerConfig(
        auto_lr_find=False,
        batch_size=1024, 
        max_epochs=5, # Reduced for first run
        accelerator=accelerator_setting, # Use the mapped setting
        devices=1 if device == 'cuda' else 1,
    )

    optimizer_config = OptimizerConfig()

    # NODE specific config
    # Refer to NodeConfig documentation for all parameters
    model_config = NodeConfig(
        task="regression",
        num_layers=2,                # Number of ODST Layers. Default is 1.
        num_trees=1024,              # Number of ODSTrees in each layer. Default is 2048.
        depth=4,                     # Depth of each tree. Default is 6.
        # choice_function="entmax15",  # Default
        # bin_function="entmoid15",    # Default
        # batch_norm_continuous_input=True, # Default for ModelConfig, NODE uses it.
        learning_rate=1e-3,          # Default for ModelConfig, can be tuned.
        loss="MSELoss"
    )

    experiment_config = ExperimentConfig(
        project_name="fine695_node", # Changed
        run_name=MODEL_NAME,
        log_target="tensorboard",
        exp_watch="gradients",
        log_logits=False,
    )
    
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        experiment_config=experiment_config,
    )

    print(f"\nFitting the TabularModel ({MODEL_NAME})...")
    try:
        tabular_model.fit(train=train_df, validation=validation_df)
        print(f"{MODEL_NAME} training complete.")
    except Exception as e:
        print(f"Error during {MODEL_NAME} training: {e}"); return

    print(f"\nEvaluating model performance on Test Set ({MODEL_NAME})...")
    test_results = tabular_model.evaluate(test_df, verbose=False)
    oos_r2 = test_results[0].get(f"{MODEL_NAME}_test_r2_score", np.nan)
    mse = test_results[0].get(f"{MODEL_NAME}_test_mean_squared_error", np.nan)
    
    if np.isnan(oos_r2) and 'test_r2_score' in test_results[0]:
        oos_r2 = test_results[0]['test_r2_score']
    if np.isnan(mse) and 'test_mean_squared_error' in test_results[0]:
         mse = test_results[0]['test_mean_squared_error']
    if np.isnan(mse) and 'test_loss' in test_results[0]:
        if model_config.loss == "MSELoss":
            mse = test_results[0]['test_loss']

    print(f"Out-of-Sample (OOS) R-squared on Test Set: {oos_r2:.6f}")
    print(f"Mean Squared Error (MSE) on Test Set: {mse:.6f}")

    print(f"\nMaking predictions on Test Set ({MODEL_NAME}) for saving...")
    predictions_test_df = tabular_model.predict(test_df, verbose=False)
    final_predictions = predictions_test_df[f'{TARGET_COLUMN}_prediction'].values

    print(f"Saving {MODEL_NAME} model...")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    model_save_path = tabular_model.config_fitted.output_dir
    print(f"{MODEL_NAME} experiment saved to: {model_save_path}")
    if hasattr(tabular_model.trainer, 'checkpoint_callback') and tabular_model.trainer.checkpoint_callback:
        best_ckpt_path = tabular_model.trainer.checkpoint_callback.best_model_path
        print(f"Best checkpoint path: {best_ckpt_path}")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    predictions_output_df = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        DATE_COLUMN: test_df[DATE_COLUMN],
        'prediction': final_predictions
    })
    predictions_save_path = os.path.join(PREDICTIONS_DIR, f"predictions_{MODEL_NAME.lower()}.parquet")
    predictions_output_df.to_parquet(predictions_save_path, index=False)
    print(f"Test predictions saved to {predictions_save_path}")
        
    metrics_to_log = {
        'out_of_sample_r2': oos_r2,
        'mse': mse,
    }
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")

    print(f"\n--- {MODEL_NAME} Model Script Complete ---")

if __name__ == "__main__":
    main() 