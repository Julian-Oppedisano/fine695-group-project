import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
# from lit_saint import SAINT # Assuming this is the correct import from the installed package
from lit_saint.model import Saint as SAINT # Try direct import
from lit_saint.config import SaintConfig, TransformerConfig, NetworkConfig, OptimizerConfig as LitSaintOptimizerConfig, TrainConfig as LitSaintTrainConfig # Import config classes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import shutil
import csv # Added for CSV logging
from datetime import datetime # Added for timestamp

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

# --- Configuration ---
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet') # Added test path

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions') # For parquet validation predictions
RESULTS_PRED_DIR = os.path.join(os.path.dirname(__file__), '..', 'results') # For CSV test predictions

MODEL_NAME = 'saint'
TARGET_COLUMN = 'stock_exret'

# --- Data Module ---
class StockDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=1024): # Added X_test, y_test
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test # Store test data
        self.y_test = y_test # Store test targets
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(torch.tensor(self.X_train, dtype=torch.float32),
                                           torch.tensor(self.y_train, dtype=torch.float32))
        self.val_dataset = TensorDataset(torch.tensor(self.X_val, dtype=torch.float32),
                                         torch.tensor(self.y_val, dtype=torch.float32))
        self.test_dataset = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32), # Create test_dataset
                                          torch.tensor(self.y_test, dtype=torch.float32))


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    def test_dataloader(self): # Added test_dataloader
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

# --- Lightning Module for SAINT ---
class SAINTLightningModule(pl.LightningModule):
    def __init__(self, saint_model_init_args: dict, num_continuous_features: int, cat_feature_indices: list,
                 lr: float = 1e-3, scheduler_patience: int = 5, scheduler_factor: float = 0.5):
        super().__init__()
        self.save_hyperparameters('saint_model_init_args', 'num_continuous_features', 'cat_feature_indices',
                                  'lr', 'scheduler_patience', 'scheduler_factor')
        self.model = SAINT(**self.hparams.saint_model_init_args)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x_combined):
        batch_size = x_combined.shape[0]
        device = x_combined.device
        dtype = x_combined.dtype

        # Continuous features
        if self.hparams.num_continuous_features > 0:
            x_cont = x_combined[:, :self.hparams.num_continuous_features]
        else:
            x_cont = torch.empty(batch_size, 0, dtype=dtype, device=device)

        # Categorical features
        num_expected_cat_features = len(self.hparams.saint_model_init_args['categories'])

        if num_expected_cat_features > 0 and self.hparams.cat_feature_indices:
            # Slice the part of x_combined that should contain categorical features
            x_categ_raw = x_combined[:, self.hparams.num_continuous_features:]
            # Ensure this slice actually has columns, matching num_expected_cat_features
            if x_categ_raw.shape[1] == num_expected_cat_features:
                x_categ = x_categ_raw.long()
            else:
                # This case indicates a mismatch between data prep and model expectation.
                # For safety, create an empty tensor, but this signals a problem.
                print(f"SAINTLightningModule Warning: Mismatch in expected categorical features. \n" \
                      f"Model configured for {num_expected_cat_features} categorical features (from saint_model_init_args[\'categories\']). \n" \
                      f"cat_feature_indices (from hparams): {self.hparams.cat_feature_indices}. \n" \
                      f"num_continuous_features (from hparams): {self.hparams.num_continuous_features}. \n" \
                      f"Shape of x_combined: {x_combined.shape}. \n" \
                      f"Resulting x_categ_raw slice shape: {x_categ_raw.shape}. \n" \
                      f"Creating empty tensor for x_categ.")
                x_categ = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        else:
            # No categorical features expected by the model or configured in cat_feature_indices.
            x_categ = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            
        model_output = self.model(x_categ=x_categ, x_cont=x_cont)
        if isinstance(model_output, tuple):
            # This might happen if compute_feature_importance was True, 
            # or if the model's forward path for pretraining was somehow activated.
            # Assuming the first element is the primary prediction.
            # print("SAINTLightningModule Info: model output is a tuple. Taking the first element as prediction.")
            return model_output[0]
        return model_output


    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'predictions': y_hat, 'targets': y}
    
    def test_step(self, batch, batch_idx): # Added test_step
        x, y = batch 
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'predictions': y_hat.detach(), 'targets': y.detach()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                               patience=self.hparams.scheduler_patience, 
                                                               factor=self.hparams.scheduler_factor)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

def get_predictor_and_categorical_columns(df, target_col):
    identifier_cols = ['permno', 'stock_ticker', 'CUSIP', 'comp_name'] 
    date_related_cols = ['date', 'year', 'month']
    categorical_feature_names = [f'month_{i}' for i in range(2, 13)]
    
    cols_to_exclude_for_numeric = identifier_cols + [target_col, 'ret_eom'] + date_related_cols + categorical_feature_names
    numeric_potential_predictors = [col for col in df.columns if col not in cols_to_exclude_for_numeric]
    numeric_predictor_names = df[numeric_potential_predictors].select_dtypes(include=np.number).columns.tolist()
    
    actual_categorical_features = [col for col in categorical_feature_names if col in df.columns]

    print(f"Identified {len(numeric_predictor_names)} numeric predictors.")
    print(f"Identified {len(actual_categorical_features)} categorical features: {actual_categorical_features}")
    return numeric_predictor_names, actual_categorical_features

def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    pl.seed_everything(42, workers=True)
    
    accelerator_device = 'cpu'
    devices_val = 'auto'
    if torch.cuda.is_available():
        accelerator_device = 'cuda'
        devices_val = 1
        print("Using device: cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # For Apple Silicon
        accelerator_device = 'mps'
        devices_val = 1
        print("Using device: mps")
    else:
        print("Using device: cpu")

    print("Loading imputed datasets...")
    try:
        train_df_orig = pd.read_parquet(TRAIN_IMPUTED_INPUT_PATH)
        validation_df_orig = pd.read_parquet(VALIDATION_IMPUTED_INPUT_PATH)
        test_df_orig = pd.read_parquet(TEST_IMPUTED_INPUT_PATH) # Load original test_df
        print("Datasets loaded successfully.")
    except Exception as e: print(f"Error loading data: {e}"); return

    numeric_cols, categ_cols = get_predictor_and_categorical_columns(train_df_orig, TARGET_COLUMN)
    if not numeric_cols and not categ_cols: print("Error: No predictors. Aborting."); return
    
    if train_df_orig[TARGET_COLUMN].isnull().any():
        print("Warning: NaNs in y_train. Dropping.")
        train_df_orig.dropna(subset=[TARGET_COLUMN], inplace=True)
    if validation_df_orig[TARGET_COLUMN].isnull().any():
        print("Warning: NaNs in y_validation. Dropping.")
        validation_df_orig.dropna(subset=[TARGET_COLUMN], inplace=True)
    if test_df_orig[TARGET_COLUMN].isnull().any(): # Handle NaNs in test target
        print("Warning: NaNs in y_test. Dropping.")
        test_df_orig.dropna(subset=[TARGET_COLUMN], inplace=True)

    y_train_df = train_df_orig[[TARGET_COLUMN]]
    y_validation_df = validation_df_orig[[TARGET_COLUMN]]
    y_test_df = test_df_orig[[TARGET_COLUMN]] # Create y_test_df

    scaler = StandardScaler()
    X_train_num_scaled = np.empty((len(train_df_orig), 0))
    X_validation_num_scaled = np.empty((len(validation_df_orig), 0))
    X_test_num_scaled = np.empty((len(test_df_orig), 0)) # For test numeric features
    num_actual_continuous_features = 0

    if numeric_cols:
        X_train_num_raw = train_df_orig[numeric_cols].fillna(0)
        X_train_num_raw = np.nan_to_num(X_train_num_raw, nan=0.0, posinf=1e9, neginf=-1e9)
        X_train_num_scaled = scaler.fit_transform(X_train_num_raw)
        joblib.dump(scaler, os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}_scaler.joblib"))
        
        X_validation_num_raw = validation_df_orig[numeric_cols].fillna(0)
        X_validation_num_raw = np.nan_to_num(X_validation_num_raw, nan=0.0, posinf=1e9, neginf=-1e9)
        X_validation_num_scaled = scaler.transform(X_validation_num_raw)
        
        X_test_num_raw = test_df_orig[numeric_cols].fillna(0) # Process test numeric features
        X_test_num_raw = np.nan_to_num(X_test_num_raw, nan=0.0, posinf=1e9, neginf=-1e9)
        X_test_num_scaled = scaler.transform(X_test_num_raw)

        num_actual_continuous_features = X_train_num_scaled.shape[1]
        print(f"Numeric features scaled. Count: {num_actual_continuous_features}")

    X_train_cat = np.empty((len(train_df_orig), 0))
    X_validation_cat = np.empty((len(validation_df_orig), 0))
    X_test_cat = np.empty((len(test_df_orig), 0)) # For test categorical features
    cat_dims = [] 

    if categ_cols:
        X_train_cat_list = []
        X_validation_cat_list = []
        X_test_cat_list = [] # For test categorical features
        for col_idx, col in enumerate(categ_cols):
            train_df_orig[col] = train_df_orig[col].fillna(0) 
            validation_df_orig[col] = validation_df_orig[col].fillna(0)
            test_df_orig[col] = test_df_orig[col].fillna(0) # Fill NaNs in test categorical
            le = LabelEncoder()
            # Fit LabelEncoder on combined train, validation, and test data for robustness
            combined_col_data = pd.concat([
                train_df_orig[col], 
                validation_df_orig[col],
                test_df_orig[col] # Include test data for fitting LE
            ], axis=0).astype(str)
            le.fit(combined_col_data)
            
            X_train_cat_list.append(le.transform(train_df_orig[col].astype(str)).reshape(-1,1))
            X_validation_cat_list.append(le.transform(validation_df_orig[col].astype(str)).reshape(-1,1))
            X_test_cat_list.append(le.transform(test_df_orig[col].astype(str)).reshape(-1,1)) # Transform test categorical

            cat_dims.append(len(le.classes_))
            joblib.dump(le, os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}_le_cat_{col_idx}.joblib"))

        X_train_cat = np.hstack(X_train_cat_list) if X_train_cat_list else X_train_cat
        X_validation_cat = np.hstack(X_validation_cat_list) if X_validation_cat_list else X_validation_cat
        X_test_cat = np.hstack(X_test_cat_list) if X_test_cat_list else X_test_cat # Hstack test categorical
        print(f"Categorical features processed. Count: {X_train_cat.shape[1]}, Dims: {cat_dims}")


    if X_train_cat.shape[1] > 0:
        X_train_final = np.concatenate((X_train_num_scaled, X_train_cat), axis=1)
        X_validation_final = np.concatenate((X_validation_num_scaled, X_validation_cat), axis=1)
        X_test_final = np.concatenate((X_test_num_scaled, X_test_cat), axis=1) # Create X_test_final
        cat_idxs = list(range(num_actual_continuous_features, X_train_final.shape[1]))
    else: 
        X_train_final = X_train_num_scaled
        X_validation_final = X_validation_num_scaled
        X_test_final = X_test_num_scaled # Create X_test_final (no cat features)
        cat_idxs = [] 
    
    y_train_np = y_train_df.to_numpy()
    y_validation_np = y_validation_df.to_numpy()
    y_test_np = y_test_df.to_numpy() # Create y_test_np

    # --- SAINT Model and DataModule Initialization ---
    # SAINT uses specific config objects from lit_saint, ensure these are correctly set up
    # Default values or values derived from data properties (like cat_dims, num_actual_continuous_features)
    transformer_cfg = TransformerConfig(
        num_attention_heads=8, 
        num_transformer_blocks=6, 
        dropout_rate=0.1,
        attention_dropout_rate=0.1, # Added for completeness if model supports
        # Other params as needed by lit_saint's TransformerConfig
    )
    network_cfg = NetworkConfig(
        embedding_dim=32, # Example, adjust as needed
        # Other params as needed by lit_saint's NetworkConfig
    )
    optimizer_cfg = LitSaintOptimizerConfig(
        name='AdamW', # Example, check lit_saint for options
        lr=1e-3, # Example
        # Other params as needed
    )
    saint_train_cfg = LitSaintTrainConfig( # Renamed to avoid conflict if `train_cfg` means something else
        pretrain=False, # Assuming supervised training
        run_name=MODEL_NAME,
        batch_size=1024 * 4, # Increased batch_size
        max_epochs=100, # Example
        # Other params
    )
    
    saint_model_args = dict(
        categories=tuple(cat_dims) if cat_dims else tuple(),
        num_continuous=num_actual_continuous_features,
        dim=32,  # Example: Dimension of embeddings/transformer, ensure consistency
        depth=6, # Example: Number of transformer layers
        heads=8, # Example: Number of attention heads
        attn_dropout=0.1,
        ff_dropout=0.1,
        # Removed direct use of config objects here, pass parameters directly
        # transformer_config=transformer_cfg, 
        # network_config=network_cfg,
        # optimizer_config=optimizer_cfg,
        # train_config=saint_train_cfg,
        # Expects parameters like 'categories', 'num_continuous', 'dim', 'depth', 'heads', etc.
        # Make sure these match the lit_saint.model.Saint constructor
        # For example, based on a typical SAINT setup:
        # dim_out = 1, for regression
        # Other specific params lit-saint's SAINT model might need
    )

    model = SAINTLightningModule(
        saint_model_init_args=saint_model_args,
        num_continuous_features=num_actual_continuous_features,
        cat_feature_indices=cat_idxs, # These are indices in the combined X data
        lr=optimizer_cfg.lr, # Use lr from config
        scheduler_patience=10, # Example
        scheduler_factor=0.5   # Example
    )

    data_module = StockDataModule(
        X_train=X_train_final, y_train=y_train_np,
        X_val=X_validation_final, y_val=y_validation_np,
        X_test=X_test_final, y_test=y_test_np, # Pass test data to DataModule
        batch_size=saint_train_cfg.batch_size
    )
    
    # --- Training Callbacks ---
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, verbose=True, mode='min' # Increased patience
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=SAVED_MODEL_DIR,
        filename=f'{MODEL_NAME}_best_model-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=saint_train_cfg.max_epochs, # Use max_epochs from config
        accelerator=accelerator_device, devices=devices_val,
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger(save_dir=os.path.join(SAVED_MODEL_DIR, 'tb_logs'), name=MODEL_NAME),
        deterministic=True, # For reproducibility
        enable_progress_bar=True
    )

    print("Starting SAINT model training...")
    try:
        trainer.fit(model, datamodule=data_module)
        print("SAINT training complete.")
        best_model_path = early_stopping_callback.best_model_path
        print(f"Best model path: {best_model_path}")
        if best_model_path and os.path.exists(best_model_path):
            best_model = SAINTLightningModule.load_from_checkpoint(
                best_model_path # PyTorch Lightning handles hparams loading if saved correctly
            )
        else:
            print("Warning: Best model checkpoint not found or path invalid. Using last model state.")
            best_model = model
        best_model.to(accelerator_device) 
        best_model.eval() 

    except Exception as e:
        print(f"Error during SAINT training or loading best model: {e}"); return

    print("\nMaking predictions on validation set...")
    # Manually iterate through val_dataloader to get predictions
    val_predictions_list = [] # Renamed to avoid conflict
    val_targets_list = []   # Renamed to avoid conflict
    with torch.no_grad():
        for batch in data_module.val_dataloader():
            x_batch, y_batch = batch
            if accelerator_device != 'cpu':
                 x_batch = x_batch.to(best_model.device)
                 y_batch = y_batch.to(best_model.device)
            
            y_hat_batch = best_model(x_batch)
            val_predictions_list.append(y_hat_batch.cpu()) # Use new list name
            val_targets_list.append(y_batch.cpu())       # Use new list name

    if not val_predictions_list: # Check new list name
        print("No validation predictions were made. Cannot evaluate or save validation preds.")
    else:
        y_pred_validation = torch.cat(val_predictions_list).numpy().flatten() # Use new list name
        y_true_validation = torch.cat(val_targets_list).numpy().flatten()   # Use new list name
        
        # Save validation predictions (Parquet)
        val_preds_df = validation_df_orig.loc[y_validation_df.index, ['permno', 'date']].copy()
        val_preds_df['actual_' + TARGET_COLUMN] = y_true_validation
        val_preds_df['predicted_' + TARGET_COLUMN] = y_pred_validation

        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        val_predictions_save_path = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_validation_predictions.parquet")
        val_preds_df.to_parquet(val_predictions_save_path, index=False)
        print(f"Validation predictions saved to {val_predictions_save_path}")

    # --- Make Predictions on Test Set ---
    print("\nMaking predictions on test set...")
    test_predictions_list = []
    test_targets_list = [] # To calculate metrics if needed
    
    # Use the test_dataloader from the data_module
    # Ensure data_module.setup() has been called if it wasn't through trainer.fit() or trainer.validate()
    # trainer.fit() calls setup(stage='fit'), trainer.validate() calls setup(stage='validate')
    # We might need to call data_module.setup(stage='test') if not already done.
    # However, PyTorch Lightning's trainer.test() handles this.
    
    # Option 1: Use trainer.test() - preferred if SAINTLightningModule.test_step is well defined
    print("Using trainer.test() for test predictions...")
    try:
        # Before calling test, ensure the model is on the correct device,
        # although trainer usually handles this.
        if accelerator_device != 'cpu': best_model.to(accelerator_device)
        
        test_results = trainer.test(model=best_model, datamodule=data_module, verbose=True)
        # test_results will be a list of dicts, one for each test_dataloader
        # Each dict contains metrics logged in test_step, e.g., 'test_loss'
        # Predictions themselves need to be collected if test_step returns them
        # and PTL is configured to aggregate them, or we do it manually.
        
        # Let's assume test_step appends to a list in the model like `model.test_outputs`
        # Or, we can get them from the callback if one is designed for it.
        # For now, let's try to get predictions manually after trainer.test() if it doesn't directly return them
        # The `test_step` in `SAINTLightningModule` returns {'predictions': y_hat, 'targets': y}
        # We need to see how PTL aggregates these.
        # A common way is to collect them in `on_test_epoch_end` or access a list populated by `test_step`.
        # For simplicity, let's try manual iteration first if `trainer.test` doesn't easily yield preds.

        # Manual iteration for test predictions (fallback or primary if trainer.test() is complex for pred extraction)
        print("Manually iterating test_dataloader for test predictions...")
        best_model.eval() # Ensure eval mode
        with torch.no_grad():
            for batch in data_module.test_dataloader():
                x_batch, y_batch = batch
                if accelerator_device != 'cpu':
                    x_batch = x_batch.to(best_model.device)
                    # y_batch can stay on CPU if only used for metrics later
                
                y_hat_batch = best_model(x_batch)
                test_predictions_list.append(y_hat_batch.cpu())
                test_targets_list.append(y_batch.cpu()) # Collect targets for metrics

    except Exception as e:
        print(f"Error during test set prediction: {e}")
        # Fallback to manual iteration if trainer.test() fails or is complex
        # This code block is now part of the manual iteration above.
        # Consider re-raising or specific error handling.


    if not test_predictions_list:
        print("No test predictions were made. Cannot evaluate or save test preds.")
        return # Exit if no test predictions

    y_pred_test = torch.cat(test_predictions_list).numpy().flatten()
    y_true_test = torch.cat(test_targets_list).numpy().flatten()

    # --- Save Test Predictions (CSV) ---
    os.makedirs(RESULTS_PRED_DIR, exist_ok=True)
    # Align with original test_df_orig using y_test_df.index (which has NaNs dropped)
    test_preds_df = test_df_orig.loc[y_test_df.index, ['permno', 'date']].copy()
    test_preds_df['prediction'] = y_pred_test
    
    csv_output_filename = f"pred_{MODEL_NAME}.csv"
    csv_predictions_save_path = os.path.join(RESULTS_PRED_DIR, csv_output_filename)
    test_preds_df.to_csv(csv_predictions_save_path, index=False)
    print(f"Test predictions saved to {csv_predictions_save_path}")

    # --- Log Test Metrics ---
    print("\nEvaluating model performance on Test Set...")
    test_oos_r2 = r2_score(y_true_test, y_pred_test)
    test_mse = mean_squared_error(y_true_test, y_pred_test)
    print(f"Out-of-Sample (OOS) R-squared on Test Set: {test_oos_r2:.4f}")
    print(f"Mean Squared Error (MSE) on Test Set: {test_mse:.6f}")

    metrics_to_log = {
        'oos_r2': test_oos_r2, 
        'mse': test_mse,
    }
    if 'log_metrics_to_csv' in globals() and 'MODEL_NAME' in globals():
        log_metrics_to_csv(MODEL_NAME, metrics_to_log) # Log test metrics
        if 'CSV_FILE' in globals():
            print(f"Test metrics logged to {CSV_FILE}")
    else:
        print("log_metrics_to_csv function or MODEL_NAME not found. Skipping CSV logging for test metrics.")
    
    # --- (Original Validation Metrics Logging - can be kept or removed if test metrics are primary) ---
    # print("\nEvaluating model performance on Validation Set (using loaded best model)...")
    # val_oos_r2 = r2_score(y_true_validation, y_pred_validation) # Requires y_true_validation and y_pred_validation
    # val_mse = mean_squared_error(y_true_validation, y_pred_validation)
    # print(f"Out-of-Sample (OOS) R-squared on Validation: {val_oos_r2:.4f}")
    # print(f"Mean Squared Error (MSE) on Validation: {val_mse:.6f}")
    # metrics_to_log_val = { # Log validation as well, perhaps with different naming or context
    #     'val_oos_r2': val_oos_r2, 
    #     'val_mse': val_mse,
    # }
    # log_metrics_to_csv(f"{MODEL_NAME}_val", metrics_to_log_val) # Example: log with suffix
    # print(f"Validation metrics logged for {MODEL_NAME}_val")


    # --- Clean up TensorBoard logs ---
    tb_log_dir = os.path.join(SAVED_MODEL_DIR, "tb_logs", MODEL_NAME)
    if os.path.exists(tb_log_dir):
        try:
            shutil.rmtree(tb_log_dir)
            print(f"Cleaned up TensorBoard log directory: {tb_log_dir}")
        except OSError as e:
            print(f"Error cleaning up TensorBoard logs: {e}")


    print(f"\n--- {MODEL_NAME} Model Script Complete ---")

if __name__ == "__main__":
    main() 