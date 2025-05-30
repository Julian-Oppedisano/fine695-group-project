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

# --- Configuration ---
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')

MODEL_NAME = 'saint'
TARGET_COLUMN = 'stock_exret'

# --- Data Module ---
class StockDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=1024):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(torch.tensor(self.X_train, dtype=torch.float32),
                                           torch.tensor(self.y_train, dtype=torch.float32))
        self.val_dataset = TensorDataset(torch.tensor(self.X_val, dtype=torch.float32),
                                         torch.tensor(self.y_val, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

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
    except Exception as e: print(f"Error loading data: {e}"); return

    numeric_cols, categ_cols = get_predictor_and_categorical_columns(train_df_orig, TARGET_COLUMN)
    if not numeric_cols and not categ_cols: print("Error: No predictors. Aborting."); return
    
    if train_df_orig[TARGET_COLUMN].isnull().any():
        print("Warning: NaNs in y_train. Dropping.")
        train_df_orig.dropna(subset=[TARGET_COLUMN], inplace=True)
    if validation_df_orig[TARGET_COLUMN].isnull().any():
        print("Warning: NaNs in y_validation. Dropping.")
        validation_df_orig.dropna(subset=[TARGET_COLUMN], inplace=True)

    y_train_df = train_df_orig[[TARGET_COLUMN]]
    y_validation_df = validation_df_orig[[TARGET_COLUMN]]

    scaler = StandardScaler()
    X_train_num_scaled = np.empty((len(train_df_orig), 0))
    X_validation_num_scaled = np.empty((len(validation_df_orig), 0))
    num_actual_continuous_features = 0

    if numeric_cols:
        X_train_num_raw = train_df_orig[numeric_cols].fillna(0)
        X_train_num_raw = np.nan_to_num(X_train_num_raw, nan=0.0, posinf=1e9, neginf=-1e9)
        X_train_num_scaled = scaler.fit_transform(X_train_num_raw)
        joblib.dump(scaler, os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}_scaler.joblib"))
        
        X_validation_num_raw = validation_df_orig[numeric_cols].fillna(0)
        X_validation_num_raw = np.nan_to_num(X_validation_num_raw, nan=0.0, posinf=1e9, neginf=-1e9)
        X_validation_num_scaled = scaler.transform(X_validation_num_raw)
        num_actual_continuous_features = X_train_num_scaled.shape[1]
        print(f"Numeric features scaled. Count: {num_actual_continuous_features}")

    X_train_cat = np.empty((len(train_df_orig), 0))
    X_validation_cat = np.empty((len(validation_df_orig), 0))
    cat_dims = [] 

    if categ_cols:
        X_train_cat_list = []
        X_validation_cat_list = []
        for col_idx, col in enumerate(categ_cols):
            train_df_orig[col] = train_df_orig[col].fillna(0) 
            validation_df_orig[col] = validation_df_orig[col].fillna(0)
            le = LabelEncoder()
            combined_col_data = pd.concat([train_df_orig[col], validation_df_orig[col]], axis=0).astype(str)
            le.fit(combined_col_data)
            X_train_cat_list.append(le.transform(train_df_orig[col].astype(str)).reshape(-1,1))
            X_validation_cat_list.append(le.transform(validation_df_orig[col].astype(str)).reshape(-1,1))
            cat_dims.append(len(le.classes_))
            # Save label encoder for this categorical column
            joblib.dump(le, os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}_le_cat_{col_idx}.joblib"))

        X_train_cat = np.hstack(X_train_cat_list) if X_train_cat_list else X_train_cat
        X_validation_cat = np.hstack(X_validation_cat_list) if X_validation_cat_list else X_validation_cat
        print(f"Categorical features processed. Count: {X_train_cat.shape[1]}, Dims: {cat_dims}")


    if X_train_cat.shape[1] > 0:
        X_train_final = np.concatenate((X_train_num_scaled, X_train_cat), axis=1)
        X_validation_final = np.concatenate((X_validation_num_scaled, X_validation_cat), axis=1)
        cat_idxs = list(range(num_actual_continuous_features, X_train_final.shape[1]))
    else: 
        X_train_final = X_train_num_scaled
        X_validation_final = X_validation_num_scaled
        cat_idxs = [] 
    
    y_train_np = y_train_df.to_numpy()
    y_validation_np = y_validation_df.to_numpy()

    data_module = StockDataModule(X_train_final, y_train_np, X_validation_final, y_validation_np, batch_size=1024*2) # Increased batch size

    # --- SAINT Model and Config Initialization ---
    saint_config = SaintConfig() # Use default config first, then customize
    
    # Customize network parameters
    saint_config.network.embedding_size = 32 # Example: was 'dim'
    saint_config.network.transformer.depth = 3 # Example: default is 3, prev used 6
    saint_config.network.transformer.heads = 4 # Example: default is 1, prev used 8
    saint_config.network.transformer.dropout = 0.1 
    # saint_config.network.transformer.attention_type = 'colrow' # Example

    # For the lit_saint.model.Saint class constructor:
    # continuous: List of indices with continuous columns --> this needs to be indices WITHIN the continuous block itself
    # So if we have N continuous features, this list should be list(range(N))
    # The SAINTLightningModule's forward pass will handle slicing the combined X_final
    
    actual_saint_model_init_args = {
        'categories': tuple(cat_dims) if cat_dims else (), # Cardinalities of categorical features
        'continuous': list(range(num_actual_continuous_features)) if num_actual_continuous_features > 0 else [], # Indices of continuous features
        'dim_target': 1, # For regression
        'config': saint_config,
        # 'metrics': None, # Optional: Can add custom torchmetrics here
        # 'loss_fn': None, # Optional: Defaults to MSE for regression / CrossEntropy for classification
        # 'optimizer': torch.optim.Adam # Optional: Handled by LightningModule
    }
    
    # Lightning Module hyperparams
    lr_val = 2e-3 # Example: prev 1e-3
    scheduler_patience_val = 10 # Example: prev 5
    scheduler_factor_val = 0.5 

    saint_lightning_model = SAINTLightningModule(
        saint_model_init_args=actual_saint_model_init_args,
        num_continuous_features=num_actual_continuous_features,
        cat_feature_indices=cat_idxs, # Indices of categorical features in the combined X_final
        lr=lr_val,
        scheduler_patience=scheduler_patience_val,
        scheduler_factor=scheduler_factor_val
    )

    print("\nSetting up PyTorch Lightning Trainer...")
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00005, patience=15, verbose=True, mode="min") # Adjusted min_delta & patience
    
    trainer = pl.Trainer(
        max_epochs=100, # Adjust as needed, prev 50
        accelerator=accelerator_device, devices=devices_val,
        callbacks=[early_stop_callback],
        logger=pl.loggers.TensorBoardLogger(save_dir=os.path.join(SAVED_MODEL_DIR, 'tb_logs'), name=MODEL_NAME),
        deterministic=True, # For reproducibility
        enable_progress_bar=True
    )

    print("Starting SAINT model training...")
    try:
        trainer.fit(saint_lightning_model, datamodule=data_module)
        print("SAINT training complete.")
        best_model_path = early_stop_callback.best_model_path
        print(f"Best model path: {best_model_path}")
        if best_model_path and os.path.exists(best_model_path):
            eval_model = SAINTLightningModule.load_from_checkpoint(
                best_model_path # PyTorch Lightning handles hparams loading if saved correctly
            )
        else:
            print("Warning: Best model checkpoint not found or path invalid. Using last model state.")
            eval_model = saint_lightning_model
        eval_model.to(accelerator_device) 
        eval_model.eval() 

    except Exception as e:
        print(f"Error during SAINT training or loading best model: {e}"); return

    print("\nMaking predictions on validation set...")
    val_preds_list = []
    with torch.no_grad():
        for batch in data_module.val_dataloader():
            x_batch, _ = batch
            x_batch = x_batch.to(accelerator_device)
            y_hat_batch = eval_model(x_batch) 
            val_preds_list.append(y_hat_batch.cpu().numpy())
    
    y_pred_validation = np.concatenate(val_preds_list, axis=0)

    print("\nEvaluating model performance...")
    oos_r2 = r2_score(y_validation_np, y_pred_validation)
    mse = mean_squared_error(y_validation_np, y_pred_validation)
    print(f"Out-of-Sample (OOS) R-squared on Validation: {oos_r2:.4f}")
    print(f"Mean Squared Error (MSE) on Validation: {mse:.6f}")

    final_model_save_path = os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}_final.ckpt")
    try:
        trainer.save_checkpoint(final_model_save_path)
        print(f"Final SAINT model state saved to {final_model_save_path}")
    except Exception as e:
        print(f"Error saving final model checkpoint: {e}")


    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    # Ensure original_validation_df aligns with y_validation_np if rows were dropped due to NaNs in target
    # Re-index validation_df_orig based on y_validation_df's index (which reflects dropped NaN target rows)
    predictions_base_df = validation_df_orig.loc[y_validation_df.index]
    
    predictions_df = predictions_base_df[['permno', 'date']].copy() 
    predictions_df['actual_' + TARGET_COLUMN] = y_validation_np.flatten()
    predictions_df['predicted_' + TARGET_COLUMN] = y_pred_validation.flatten()
    predictions_save_path = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_validation_predictions.parquet")
    predictions_df.to_parquet(predictions_save_path, index=False)
    print(f"Validation predictions saved to {predictions_save_path}")
        
    print(f"\n--- {MODEL_NAME} Model Script Complete ---")

if __name__ == "__main__":
    main() 