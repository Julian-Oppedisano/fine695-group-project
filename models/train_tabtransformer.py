import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import TabTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import csv # Added for CSV logging
from datetime import datetime # Added for timestamp

# --- Configuration ---
MODEL_NAME = "TabTransformer"
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_model.ckpt') # For PyTorch Lightning checkpoint
SCALER_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_scaler.joblib')
LABEL_ENCODERS_PATH = os.path.join(SAVED_MODEL_DIR, f'{MODEL_NAME.lower()}_label_encoders.joblib')
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
                     # Future returns that might have been kept for other analyses
                     'stock_exret_t_plus_1', 'stock_exret_t_plus_2', 
                     'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12',
                     # Month dummy prefix if used (e.g. 'month_')
                    ] 
    # Filter out columns that are explicitly excluded or are month dummies
    predictor_cols = [col for col in df.columns if col not in excluded_cols and not col.startswith('month_')]
    return predictor_cols

def get_categorical_and_continuous_columns(df, predictor_cols):
    categorical_cols = []
    continuous_cols = []
    # Heuristic: identify categorical columns (low unique values, object/int type)
    # For now, let's assume 'month_num' if it wasn't excluded, and other potential low-cardinality integer columns
    # This part may need refinement based on actual data.
    # For TabTransformer, we will also need cardinalities.
    
    # A more robust approach (for now simplified):
    # Check for object types and integer types with few unique values
    for col in predictor_cols:
        if df[col].dtype == 'object' or df[col].nunique() < 20 and df[col].dtype == 'int64': # Heuristic
             if col not in ['year', 'month', 'day']: # Ensure date parts are not treated as categorical here
                categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            continuous_cols.append(col)
            
    # Ensure 'month_num' (if present and intended as categorical) is handled
    # For this project, month dummies are created, so 'month_num' itself is not a direct feature.
    # We will use month dummies as continuous for now, or handle them separately if needed.

    print(f"Identified {len(categorical_cols)} categorical columns: {categorical_cols}")
    print(f"Identified {len(continuous_cols)} continuous columns: {continuous_cols}")
    return categorical_cols, continuous_cols

# --- PyTorch Lightning DataModule ---
class TabularDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, categorical_cols, continuous_cols, target_col, batch_size=1024): # Increased batch size
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.target_col = target_col
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.label_encoders = {col: LabelEncoder() for col in self.categorical_cols}
        self.cat_cardinalities = []

    def setup(self, stage=None):
        # Handle infinities and NaNs in continuous columns before scaling
        if self.continuous_cols:
            for df_ref in [self.train_df, self.val_df, self.test_df]:
                # Use .loc to ensure modification of the original DataFrame slice
                df_ref.loc[:, self.continuous_cols] = df_ref[self.continuous_cols].replace([np.inf, -np.inf], np.nan)
                df_ref.loc[:, self.continuous_cols] = df_ref[self.continuous_cols].fillna(0) # Fill NaNs with 0 after handling infs

        # Fit scaler on training data and transform
        if self.continuous_cols:
            self.train_df.loc[:, self.continuous_cols] = self.scaler.fit_transform(self.train_df[self.continuous_cols]).astype(np.float32)
            self.val_df.loc[:, self.continuous_cols] = self.scaler.transform(self.val_df[self.continuous_cols]).astype(np.float32)
            self.test_df.loc[:, self.continuous_cols] = self.scaler.transform(self.test_df[self.continuous_cols]).astype(np.float32)

        # Fit label encoders
        self.cat_cardinalities = []
        for col in self.categorical_cols:
            all_unique_values = pd.concat([
                self.train_df[col].astype(str),
                self.val_df[col].astype(str),
                self.test_df[col].astype(str)
            ]).unique()
            
            self.label_encoders[col].fit(all_unique_values)
            
            self.train_df.loc[:, col] = self.label_encoders[col].transform(self.train_df[col].astype(str)).astype(np.int64)
            self.val_df.loc[:, col] = self.label_encoders[col].transform(self.val_df[col].astype(str)).astype(np.int64)
            self.test_df.loc[:, col] = self.label_encoders[col].transform(self.test_df[col].astype(str)).astype(np.int64)
            self.cat_cardinalities.append(len(self.label_encoders[col].classes_))
        
        print(f"Categorical cardinalities: {self.cat_cardinalities}")

        # Prepare tensors
        if self.categorical_cols:
            train_cat_arrays = [self.train_df[col].values.astype(np.int64) for col in self.categorical_cols]
            self.X_train_cat = torch.tensor(np.stack(train_cat_arrays, axis=1), dtype=torch.long)
            val_cat_arrays = [self.val_df[col].values.astype(np.int64) for col in self.categorical_cols]
            self.X_val_cat = torch.tensor(np.stack(val_cat_arrays, axis=1), dtype=torch.long)
            test_cat_arrays = [self.test_df[col].values.astype(np.int64) for col in self.categorical_cols]
            self.X_test_cat = torch.tensor(np.stack(test_cat_arrays, axis=1), dtype=torch.long)
        else:
            self.X_train_cat = torch.empty(len(self.train_df), 0, dtype=torch.long)

        self.X_train_cont = torch.tensor(self.train_df[self.continuous_cols].values, dtype=torch.float32) if self.continuous_cols else torch.empty(len(self.train_df), 0, dtype=torch.float32)
        self.y_train = torch.tensor(self.train_df[self.target_col].values, dtype=torch.float32).unsqueeze(1)

        self.X_val_cont = torch.tensor(self.val_df[self.continuous_cols].values, dtype=torch.float32) if self.continuous_cols else torch.empty(len(self.val_df), 0, dtype=torch.float32)
        self.y_val = torch.tensor(self.val_df[self.target_col].values, dtype=torch.float32).unsqueeze(1)

        self.X_test_cont = torch.tensor(self.test_df[self.continuous_cols].values, dtype=torch.float32) if self.continuous_cols else torch.empty(len(self.test_df), 0, dtype=torch.float32)
        self.y_test = torch.tensor(self.test_df[self.target_col].values, dtype=torch.float32).unsqueeze(1)

    def train_dataloader(self):
        train_dataset = TensorDataset(self.X_train_cat, self.X_train_cont, self.y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True) # Added num_workers

    def val_dataloader(self):
        val_dataset = TensorDataset(self.X_val_cat, self.X_val_cont, self.y_val)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True) # Added num_workers

    def test_dataloader(self):
        test_dataset = TensorDataset(self.X_test_cat, self.X_test_cont, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True) # Added num_workers

# --- PyTorch Lightning Model ---
class TabTransformerLightning(pl.LightningModule):
    def __init__(self, categories, num_continuous, dim, depth, heads, dim_out=1, lr=1e-3, att_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_out=dim_out,
            attn_dropout=att_dropout,
            ff_dropout=ff_dropout,
            # mlp_hidden_mults = (4, 2), # Default in TabTransformer source
            # mlp_act = nn.ReLU() # Default in TabTransformer source
        )
        self.criterion = torch.nn.MSELoss()
        self.lr = lr
        self.test_step_outputs = [] # Added to manually collect outputs

    def forward(self, x_cat, x_cont):
        return self.model(x_cat, x_cont)

    def training_step(self, batch, batch_idx):
        x_cat, x_cont, y = batch
        y_hat = self(x_cat, x_cont)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_cat, x_cont, y = batch
        y_hat = self(x_cat, x_cont)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate R2 for validation set for monitoring
        y_np = y.cpu().numpy()
        y_hat_np = y_hat.cpu().numpy()
        # Ensure y_np is not all zeros or constant
        if np.var(y_np) > 1e-6 : # Check if variance is non-negligible
             val_r2 = r2_score(y_np, y_hat_np)
             self.log('val_r2', val_r2, on_epoch=True, prog_bar=True, logger=True)
        else:
             self.log('val_r2', 0.0, on_epoch=True, prog_bar=True, logger=True) # Or log NaN, or skip

        return loss
        
    def test_step(self, batch, batch_idx):
        x_cat, x_cont, y = batch
        y_hat = self(x_cat, x_cont)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        output = {'test_loss': loss, 'predictions': y_hat.detach(), 'targets': y.detach()}
        self.test_step_outputs.append(output) # Manually append output
        return output # Still return for PTL's internal logging/callbacks

    def on_test_epoch_end(self):
        # Clear the collected outputs after the test epoch ends
        # This is important if trainer.test() might be called multiple times on the same model instance
        # if hasattr(self, 'test_step_outputs'): # Check if attribute exists
        #     self.test_step_outputs.clear() # Temporarily removed to allow main() to access outputs
        pass # Or remove the method if it does nothing else

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr) # Changed to AdamW
        return optimizer

# --- Main Training Script ---
def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    pl.seed_everything(42, workers=True)
    
    # Check for MPS (Apple Silicon) or CUDA availability
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
    train_df = pd.read_parquet(TRAIN_IMPUTED_INPUT_PATH)
    val_df = pd.read_parquet(VALIDATION_IMPUTED_INPUT_PATH)
    test_df = pd.read_parquet(TEST_IMPUTED_INPUT_PATH)
    
    print(f"Train df shape: {train_df.shape}")
    print(f"Validation df shape: {val_df.shape}")
    print(f"Test df shape: {test_df.shape}")

    # Ensure target is not all NaNs
    for df_name, df_content in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if df_content[TARGET_COLUMN].isnull().all():
            raise ValueError(f"Target column '{TARGET_COLUMN}' in {df_name}_df is all NaNs.")
        # For TabTransformer, ensure target is not used in predictors
        if TARGET_COLUMN in get_predictor_columns(df_content):
             raise ValueError(f"Target column '{TARGET_COLUMN}' found in predictor list for {df_name}_df.")


    # Identify predictor columns
    predictor_cols_train = get_predictor_columns(train_df)
    
    # Ensure all predictor columns are present in val and test
    missing_cols_val = [col for col in predictor_cols_train if col not in val_df.columns]
    if missing_cols_val:
        raise ValueError(f"Missing predictor columns in validation set: {missing_cols_val}")
    missing_cols_test = [col for col in predictor_cols_train if col not in test_df.columns]
    if missing_cols_test:
        raise ValueError(f"Missing predictor columns in test set: {missing_cols_test}")

    # Use only predictor_cols_train for consistency
    val_df = val_df[predictor_cols_train + [TARGET_COLUMN, DATE_COLUMN, ID_COLUMN]]
    test_df = test_df[predictor_cols_train + [TARGET_COLUMN, DATE_COLUMN, ID_COLUMN]]
    
    categorical_cols, continuous_cols = get_categorical_and_continuous_columns(train_df, predictor_cols_train)
    
    # Create DataModule
    data_module = TabularDataModule(
        train_df=train_df[predictor_cols_train + [TARGET_COLUMN]].copy(), # Pass only necessary columns
        val_df=val_df[predictor_cols_train + [TARGET_COLUMN]].copy(),
        test_df=test_df[predictor_cols_train + [TARGET_COLUMN]].copy(),
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        target_col=TARGET_COLUMN,
        batch_size=2048 # Further increased batch size
    )
    data_module.setup() # This fits scalers/encoders and prepares tensors

    # Initialize model
    # These are example parameters, may need tuning
    tabtransformer_model = TabTransformerLightning(
        categories=tuple(data_module.cat_cardinalities), # Important: must be a tuple
        num_continuous=len(continuous_cols),
        dim=32,        # Dimension of embeddings
        depth=6,       # Number of transformer layers
        heads=8,       # Number of attention heads
        dim_out=1,     # Output dimension (1 for regression)
        lr=1e-4,       # Adjusted learning rate
        att_dropout=0.2, # Increased dropout
        ff_dropout=0.2   # Increased dropout
    )

    # Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        min_delta=0.0001,    # Minimum change to qualify as an improvement
        patience=10,         # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=SAVED_MODEL_DIR,
        filename=f'{MODEL_NAME.lower()}_best_model', # Save best model
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=5, # Reduced for smoke test (Task: "smoke test") 
        accelerator=accelerator_device, 
        devices=devices_val,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger(save_dir=os.path.join(SAVED_MODEL_DIR, 'tb_logs'), name=MODEL_NAME.lower()),
        deterministic=True, # For reproducibility
        # precision='16-mixed' if accelerator_device != 'cpu' else 32 # Mixed precision if not on CPU
    )

    print("Starting model training...")
    trainer.fit(tabtransformer_model, datamodule=data_module)
    print("Training finished.")

    # Save scaler and label encoders
    print("Saving scaler and label encoders...")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    joblib.dump(data_module.scaler, SCALER_PATH)
    joblib.dump(data_module.label_encoders, LABEL_ENCODERS_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Label encoders saved to {LABEL_ENCODERS_PATH}")

    # Load best model for testing
    print(f"Loading best model from {checkpoint_callback.best_model_path} for testing...")
    best_model = TabTransformerLightning.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    # Test the model
    print("Evaluating model on the test set...")
    trainer.test(best_model, datamodule=data_module) # Call test
    
    # Aggregate predictions and targets from manually collected outputs
    batch_outputs = best_model.test_step_outputs # Use manually collected outputs
                                                 # No longer test_results[0]

    if not batch_outputs:
        raise RuntimeError("test_step_outputs was empty after running trainer.test(). Check test_step and on_test_epoch_end.")

    all_predictions = torch.cat([r['predictions'] for r in batch_outputs]).cpu().numpy().flatten()
    all_targets = torch.cat([r['targets'] for r in batch_outputs]).cpu().numpy().flatten()

    # Calculate final metrics
    oos_r2 = r2_score(all_targets, all_predictions)
    oos_mse = mean_squared_error(all_targets, all_predictions)
    print(f"Out-of-Sample R2 Score: {oos_r2:.6f}")
    print(f"Out-of-Sample MSE: {oos_mse:.6f}")
    
    # Save predictions
    print("Saving predictions...")
    # Re-attach permno and date for predictions file
    # Important: Ensure test_df used here has original permno and date, and is in the same order as test_dataloader
    # The data_module.test_df is a copy and is modified. Reload original test_df or use its index.
    original_test_df_ids = pd.read_parquet(TEST_IMPUTED_INPUT_PATH, columns=[ID_COLUMN, DATE_COLUMN])
    
    if len(original_test_df_ids) != len(all_predictions):
        print(f"Warning: Length of original test IDs ({len(original_test_df_ids)}) does not match predictions ({len(all_predictions)}). Using test_df from DataModule for IDs.")
        # This might happen if test_df in DataModule was filtered/sampled.
        # Fallback to test_df from datamodule (less ideal as it's processed)
        # Ensure correct permno and date are present:
        pred_df_ids = data_module.test_df[[ID_COLUMN, DATE_COLUMN]].copy() # Assuming they were kept in test_df
        pred_df_ids.reset_index(drop=True, inplace=True)
    else:
         pred_df_ids = original_test_df_ids.iloc[:len(all_predictions)].copy()


    predictions_df = pd.DataFrame({
        ID_COLUMN: pred_df_ids[ID_COLUMN],
        DATE_COLUMN: pred_df_ids[DATE_COLUMN],
        'prediction': all_predictions
    })
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")
    print(f"Model checkpoint saved to: {checkpoint_callback.best_model_path}")
    print(f"--- {MODEL_NAME} Model Training Complete ---")

    # Log metrics to CSV
    metrics_dict = {
        'out_of_sample_r2': oos_r2,
        'out_of_sample_mse': oos_mse
    }
    log_metrics_to_csv(MODEL_NAME, metrics_dict)

    # --- Log Metrics ---
    metrics_to_log = {
        'oos_r2': oos_r2, # Assuming test set is the hold-out validation
        'mse': oos_mse,
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

if __name__ == '__main__':
    main() 