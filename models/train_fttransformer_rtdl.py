import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import csv
from datetime import datetime
import rtdl # Revisiting Tabular Deep Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import collections

# --- Configuration ---
MODEL_NAME = "FTTransformer_RTDL"
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_IMPUTED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results') # Adjusted path
CSV_FILE = os.path.join(RESULTS_DIR, 'performance_summary.csv')

TARGET_COLUMN = 'stock_exret'
DATE_COLUMN = 'date'
ID_COLUMN = 'permno'

# FT-Transformer Hyperparameters (can be tuned)
D_TOKEN = 192       # Token embedding dimensionality
N_BLOCKS = 3        # Number of Transformer blocks
ATTENTION_N_HEADS = 8 # Number of attention heads
ATTENTION_DROPOUT = 0.2
FFN_D_HIDDEN_FACTOR = 4 / 3 # Factor for hidden layer size in FFN
FFN_DROPOUT = 0.1
RESIDUAL_DROPOUT = 0.0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256 # Smaller batch size might be needed if memory is an issue
EPOCHS = 5 # Reduced for initial testing, was 50-100

# --- Define CSV Logging Function ---
def log_metrics_to_csv(model_name, metrics_dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.isfile(CSV_FILE)
    # Ensure keys are consistently ordered
    metric_keys = sorted([k for k in metrics_dict.keys() if k not in ['timestamp', 'model_name']])
    fieldnames = ['timestamp', 'model_name'] + metric_keys
    
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writeheader()
        log_entry = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'model_name': model_name}
        log_entry.update(metrics_dict)
        writer.writerow(log_entry)

# --- Helper Functions ---
def get_predictor_columns(df):
    excluded_cols = [TARGET_COLUMN, DATE_COLUMN, ID_COLUMN, 'year', 'month', 'day', 
                     'eom_date', 'size_class', 'comb_code', 'month_num',
                     'stock_exret_t_plus_1', 'stock_exret_t_plus_2', 
                     'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12']
    predictor_cols = [col for col in df.columns if col not in excluded_cols]
    return predictor_cols

def preprocess_data(df, predictor_cols, fit_scalers=False, num_scaler=None, cat_encoders=None):
    df_processed = df.copy()

    # Identify categorical and numerical columns
    categorical_cols = [col for col in predictor_cols if df_processed[col].dtype == 'object' or df_processed[col].nunique() < 20] # Heuristic for categorical
    numerical_cols = [col for col in predictor_cols if col not in categorical_cols]
    
    # Impute NaNs and infinities
    for col in predictor_cols:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].fillna('MISSING').astype(str)
        else:
            df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Numerical features
    x_num = df_processed[numerical_cols].values.astype(np.float32)
    if fit_scalers:
        num_scaler = StandardScaler()
        x_num = num_scaler.fit_transform(x_num)
    elif num_scaler:
        x_num = num_scaler.transform(x_num)
    
    # Categorical features
    x_cat = None
    cat_cardinalities = []
    if cat_encoders is None:
        cat_encoders = collections.defaultdict(LabelEncoder)

    if categorical_cols:
        x_cat_list = []
        for i, col in enumerate(categorical_cols):
            if fit_scalers: # Fit label encoders only on training data
                encoded_col = cat_encoders[col].fit_transform(df_processed[col].values)
            else: # Transform using existing encoders
                # Handle unseen labels by mapping them to a new category (e.g., len of classes)
                # Or more simply, try/except to map to a default 'unknown' index if pre-fit
                try:
                    encoded_col = cat_encoders[col].transform(df_processed[col].values)
                except ValueError: # Handle unseen labels during transform
                    # Option 1: Map to a specific index (e.g., 0 or max_existing_label + 1)
                    # For simplicity, we'll try to map to max known + 1, requires storing this info
                    # A simpler robust way is to add a 'UNK' category to all encoders during fit
                    # For now, let's use a placeholder or skip if a new label is truly problematic
                    print(f"Warning: Unseen labels in {col} for test/val. Mapping to -1 (will become 0 after +1).")
                    # Create a mask for known labels
                    known_mask = df_processed[col].isin(cat_encoders[col].classes_)
                    encoded_col = np.full(len(df_processed), -1, dtype=int) # Placeholder for unknown
                    encoded_col[known_mask] = cat_encoders[col].transform(df_processed[col][known_mask])


            x_cat_list.append(encoded_col.reshape(-1, 1))
        
        x_cat = np.concatenate(x_cat_list, axis=1).astype(np.int64)
        # rtdl expects 0-based category indices, ensure no negatives from unseen labels
        x_cat = np.maximum(x_cat, 0) 


    if fit_scalers: # Get cardinalities after fitting encoders
      for col in categorical_cols:
          cat_cardinalities.append(len(cat_encoders[col].classes_))
            
    y = df_processed[TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)
    
    return x_num, x_cat, y, num_scaler, cat_encoders, numerical_cols, categorical_cols, cat_cardinalities


# --- Main Training Script ---
def main():
    print(f"--- Training {MODEL_NAME} Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading datasets...")
    try:
        train_df_raw = pd.read_parquet(TRAIN_IMPUTED_INPUT_PATH)
        val_df_raw = pd.read_parquet(VALIDATION_IMPUTED_INPUT_PATH)
        test_df_raw = pd.read_parquet(TEST_IMPUTED_INPUT_PATH)
    except Exception as e:
        print(f"Error loading data: {e}"); return

    predictor_cols = get_predictor_columns(train_df_raw)

    print("Preprocessing training data...")
    x_num_train, x_cat_train, y_train, num_scaler, cat_encoders, numerical_cols, categorical_cols, cat_cardinalities = preprocess_data(
        train_df_raw, predictor_cols, fit_scalers=True
    )
    print(f"Numerical features: {len(numerical_cols)}, Categorical features: {len(categorical_cols)}")
    if cat_cardinalities: print(f"Categorical cardinalities: {cat_cardinalities}")


    print("Preprocessing validation data...")
    x_num_val, x_cat_val, y_val, _, _, _, _, _ = preprocess_data(
        val_df_raw, predictor_cols, fit_scalers=False, num_scaler=num_scaler, cat_encoders=cat_encoders
    )
    print("Preprocessing test data...")
    x_num_test, x_cat_test, y_test, _, _, _, _, _ = preprocess_data(
        test_df_raw, predictor_cols, fit_scalers=False, num_scaler=num_scaler, cat_encoders=cat_encoders
    )
    
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(torch.tensor(x_num_train), 
                                  torch.tensor(x_cat_train) if x_cat_train is not None else torch.empty(len(x_num_train), 0, dtype=torch.int64), # Handle case with no cat features
                                  torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(x_num_val), 
                                torch.tensor(x_cat_val) if x_cat_val is not None else torch.empty(len(x_num_val), 0, dtype=torch.int64),
                                torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(x_num_test), 
                                 torch.tensor(x_cat_test) if x_cat_test is not None else torch.empty(len(x_num_test), 0, dtype=torch.int64),
                                 torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=x_num_train.shape[1] if x_num_train is not None else 0,
        cat_cardinalities=cat_cardinalities if cat_cardinalities else [], # Must be empty list if no cat features
        d_token=D_TOKEN,
        n_blocks=N_BLOCKS,
        attention_n_heads=ATTENTION_N_HEADS,
        attention_dropout=ATTENTION_DROPOUT,
        ffn_d_hidden_factor=FFN_D_HIDDEN_FACTOR,
        ffn_dropout=FFN_DROPOUT,
        residual_dropout=RESIDUAL_DROPOUT,
        d_out=1, # For regression
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    print("Starting training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5 # For early stopping

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_x_num, batch_x_cat, batch_y in train_loader:
            batch_x_num, batch_x_cat, batch_y = batch_x_num.to(device), batch_x_cat.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            # rtdl FTTransformer expects x_num and x_cat (can be None if not present)
            outputs = model(batch_x_num if batch_x_num.nelement() > 0 else None, 
                            batch_x_cat if batch_x_cat.nelement() > 0 else None)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x_num, batch_x_cat, batch_y in val_loader:
                batch_x_num, batch_x_cat, batch_y = batch_x_num.to(device), batch_x_cat.to(device), batch_y.to(device)
                outputs = model(batch_x_num if batch_x_num.nelement() > 0 else None,
                                batch_x_cat if batch_x_cat.nelement() > 0 else None)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
            model_save_path = os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}_best.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
    
    # Load best model for evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}_best.pt")))
    model.eval()
    
    all_predictions_test = []
    all_y_test = []
    with torch.no_grad():
        for batch_x_num, batch_x_cat, batch_y in test_loader:
            batch_x_num, batch_x_cat, batch_y = batch_x_num.to(device), batch_x_cat.to(device), batch_y.to(device)
            outputs = model(batch_x_num if batch_x_num.nelement() > 0 else None,
                                batch_x_cat if batch_x_cat.nelement() > 0 else None)
            all_predictions_test.extend(outputs.cpu().numpy())
            all_y_test.extend(batch_y.cpu().numpy())

    all_predictions_test = np.array(all_predictions_test).flatten()
    all_y_test = np.array(all_y_test).flatten()
    
    # Handle potential NaNs in y_test for metrics calculation
    valid_indices = ~np.isnan(all_y_test)
    if np.sum(valid_indices) == 0:
        print("Error: All y_test values are NaN after filtering. Cannot calculate metrics.")
        oos_r2, mse = np.nan, np.nan
    else:
        all_predictions_test_valid = all_predictions_test[valid_indices]
        all_y_test_valid = all_y_test[valid_indices]
        if len(all_y_test_valid) > 0 :
            oos_r2 = r2_score(all_y_test_valid, all_predictions_test_valid)
            mse = mean_squared_error(all_y_test_valid, all_predictions_test_valid)
        else:
            print("Warning: No valid (non-NaN) y_test values remaining to calculate metrics.")
            oos_r2, mse = np.nan, np.nan


    print(f"Out-of-Sample (OOS) R-squared on Test Set: {oos_r2:.6f}")
    print(f"Mean Squared Error (MSE) on Test Set: {mse:.6f}")

    # Save Predictions
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    # Re-attach permno and date for predictions file - need original test_df_raw order
    # Ensure the test_df_raw corresponds to the order of y_test / all_predictions_test
    # This assumes test_loader processes test_df_raw in order without shuffle
    
    # Create a DataFrame for predictions, ensuring IDs are aligned
    # Note: If test_df_raw was filtered for NaNs in target before preprocessing,
    # this alignment needs careful handling. Here, we assume test_df_raw is full.
    # For robust alignment, it's better to carry IDs through preprocessing or re-index.
    # For simplicity, assuming order is maintained:
    
    # Ensure test_df_raw has 'permno' and 'date'
    # And that its length matches y_test before NaN filtering for metric calculation
    # If y_test had NaNs, all_predictions_test will be shorter.
    # We should save predictions for ALL test samples, even if target was NaN.
    # The `all_predictions_test` at this point matches the rows processed by `test_loader`.
    # If `test_df_raw` was not filtered, the length should match.

    if len(all_predictions_test) == len(test_df_raw):
        predictions_output_df = pd.DataFrame({
            ID_COLUMN: test_df_raw[ID_COLUMN].values,
            DATE_COLUMN: test_df_raw[DATE_COLUMN].values,
            'prediction': all_predictions_test
        })
        predictions_save_path = os.path.join(PREDICTIONS_DIR, f"predictions_{MODEL_NAME.lower()}.parquet")
        predictions_output_df.to_parquet(predictions_save_path, index=False)
        print(f"Test predictions saved to {predictions_save_path}")
    else:
        print(f"Warning: Length of predictions ({len(all_predictions_test)}) does not match test_df_raw ({len(test_df_raw)}). Skipping saving predictions to avoid misalignment.")
        print("This might happen if original y_test contained NaNs that were dropped for metrics but not for prediction output length.")


    # Log Metrics
    metrics_to_log = {
        'out_of_sample_r2': oos_r2 if not np.isnan(oos_r2) else 'NaN', # CSV friendly
        'mse': mse if not np.isnan(mse) else 'NaN',
    }
    log_metrics_to_csv(MODEL_NAME, metrics_to_log)
    print(f"Metrics logged to {CSV_FILE}")

    print(f"\n--- {MODEL_NAME} Model Script Complete ---")

if __name__ == "__main__":
    main() 