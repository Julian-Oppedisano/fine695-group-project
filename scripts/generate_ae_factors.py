import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs') # For TensorFlow logs if needed

TRAIN_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VAL_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

TRAIN_AE_FACTORS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'train_ae_factors.parquet')
VAL_AE_FACTORS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_ae_factors.parquet')
TEST_AE_FACTORS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'test_ae_factors.parquet')

ENCODER_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'autoencoder_encoder.keras')
FEATURE_SCALER_PATH = os.path.join(SAVED_MODEL_DIR, 'autoencoder_feature_scaler.joblib')

# Parameters from your model_autoencoder.py
ENCODING_DIM = 32
AE_EPOCHS = 50 # Max epochs for AE training
AE_BATCH_SIZE = 256
AE_ACTIVATIONS = 'relu'
AE_OPTIMIZER = 'adam'

# Columns to identify specific data points, and the original target (not used for AE training directly)
TARGET_COLUMN = 'stock_exret' # Present in input files, but not used for AE input
DATE_COLUMN = 'date'
ID_COLUMN = 'permno'

# Helper to get predictor columns (adapted from train_catboost.py)
def get_original_predictor_columns(df):
    excluded_cols = [TARGET_COLUMN, DATE_COLUMN, ID_COLUMN, 'year', 'month', 'day', 
                     'eom_date', 'size_class', 'comb_code', 'month_num',
                     'stock_exret_t_plus_1', 'stock_exret_t_plus_2', 
                     'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12',
                     'SHRCD', 
                     # Add the newly identified problematic object columns for AE input
                     'size_port', 'stock_ticker', 'CUSIP', 'comp_name']
    # Also exclude 'target_quintile' if it was added in other scripts (though should not be in imputed features)
    excluded_cols.append('target_quintile') 
    
    original_predictors = [col for col in df.columns if col not in excluded_cols and 
                           not col.startswith('month_') and not col.startswith('eps_')]
    return original_predictors

# Autoencoder Definition (from your model_autoencoder.py)
def create_autoencoder_model(input_dim, encoding_dim, ae_activations, ae_optimizer):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation=ae_activations)(input_layer)
    encoded = Dense(64, activation=ae_activations)(encoded)
    encoder_output = Dense(encoding_dim, activation=ae_activations)(encoded)
    encoder_model = Model(input_layer, encoder_output, name="encoder")

    decoder_input = Input(shape=(encoding_dim,)) # This should be 'encoding_dim'
    decoded = Dense(64, activation=ae_activations)(decoder_input)
    decoded = Dense(128, activation=ae_activations)(decoded)
    # Using 'linear' for reconstruction if data is StandardScaler normalized (can be negative)
    # If features are always positive and scaled 0-1 (e.g. MinMaxScaler), 'sigmoid' might be better.
    decoder_output = Dense(input_dim, activation='linear')(decoded) 
    decoder_model = Model(decoder_input, decoder_output, name="decoder")

    autoencoder_input = Input(shape=(input_dim,))
    encoded_img = encoder_model(autoencoder_input)
    reconstructed_img = decoder_model(encoded_img)
    autoencoder_model = Model(autoencoder_input, reconstructed_img, name="autoencoder")
    
    autoencoder_model.compile(optimizer=ae_optimizer, loss='mse')
    return autoencoder_model, encoder_model

def main():
    print("--- Starting Autoencoder Factor Generation ---")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True) # Ensure processed dir exists for output

    # Load data
    print("Loading data...")
    train_df_full = pd.read_parquet(TRAIN_INPUT_PATH)
    val_df_full = pd.read_parquet(VAL_INPUT_PATH)
    test_df_full = pd.read_parquet(TEST_INPUT_PATH)

    print(f"Train full shape: {train_df_full.shape}")
    print(f"Validation full shape: {val_df_full.shape}")
    print(f"Test full shape: {test_df_full.shape}")

    # Identify original predictor columns
    original_predictors = get_original_predictor_columns(train_df_full)
    print(f"Identified {len(original_predictors)} original predictor columns.")
    if not original_predictors:
        raise ValueError("No original predictor columns identified. Check get_original_predictor_columns logic.")

    # Prepare feature sets for AE
    X_train_orig = train_df_full[original_predictors].copy()
    X_val_orig = val_df_full[original_predictors].copy()
    X_test_orig = test_df_full[original_predictors].copy()

    # --- Handle Infinities before scaling ---
    print("Handling potential infinities in original predictor columns...")
    for df_part in [X_train_orig, X_val_orig, X_test_orig]:
        df_part.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaNs that might have been created from infinities, or already existed
        # Using 0 as a neutral fill before scaling. Could also use median if preferred.
        df_part.fillna(0, inplace=True) 
    print("Infinity handling complete.")
    # --- End Handle Infinities ---

    # --- Start Diagnostics for X_train_orig dtypes ---
    print("\n--- X_train_orig Column Dtype Diagnostics ---")
    object_cols = X_train_orig.select_dtypes(include=['object']).columns
    if not object_cols.empty:
        print(f"Found {len(object_cols)} columns with object dtype:")
        for col in object_cols:
            print(f"  Column: {col}")
            print(f"    Unique values (sample): {X_train_orig[col].unique()[:20]}") # Show up to 20 unique values
            try:
                # Attempt to convert to numeric to see if it's mostly numeric with some odd strings
                pd.to_numeric(X_train_orig[col])
                print(f"    Column {col} CAN be converted to numeric (potentially with NaNs for errors).")
            except ValueError:
                print(f"    Column {col} CANNOT be fully converted to numeric due to non-numeric strings.")
    else:
        print("No columns with object dtype found in X_train_orig.")
    print("--- End X_train_orig Column Dtype Diagnostics ---\n")
    # --- End Diagnostics ---

    # Imputation (already done in our Parquet files, but ensure no NaNs remain from other ops)
    # If NaNs are impossible here, this can be skipped. Parquet files are from imputation.
    # X_train_orig.fillna(X_train_orig.median(), inplace=True) # Or 0 if median is not appropriate post-imputation
    # X_val_orig.fillna(X_train_orig.median(), inplace=True) # Use train median for val and test
    # X_test_orig.fillna(X_train_orig.median(), inplace=True)

    # Scaling features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_val_scaled = scaler.transform(X_val_orig)
    X_test_scaled = scaler.transform(X_test_orig)
    
    joblib.dump(scaler, FEATURE_SCALER_PATH)
    print(f"Feature scaler saved to {FEATURE_SCALER_PATH}")

    input_dim = X_train_scaled.shape[1]

    # Create and Train Autoencoder
    print("Creating and training Autoencoder...")
    autoencoder, encoder = create_autoencoder_model(input_dim, ENCODING_DIM, AE_ACTIVATIONS, AE_OPTIMIZER)
    
    print(autoencoder.summary())
    print(encoder.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1) # Increased patience slightly

    if X_train_scaled.shape[0] < AE_BATCH_SIZE or X_val_scaled.shape[0] == 0:
        raise ValueError("Insufficient training or validation data for Autoencoder.")

    history = autoencoder.fit(X_train_scaled, X_train_scaled,
                              epochs=AE_EPOCHS,
                              batch_size=AE_BATCH_SIZE,
                              validation_data=(X_val_scaled, X_val_scaled),
                              callbacks=[early_stopping],
                              verbose=1) # Print training progress
    
    print(f"Autoencoder training complete. Final val_loss: {history.history['val_loss'][-1]:.4f}")
    encoder.save(ENCODER_MODEL_PATH)
    print(f"Encoder model saved to {ENCODER_MODEL_PATH}")

    # Generate latent features
    print("Generating latent features using the trained encoder...")
    train_latent_features = encoder.predict(X_train_scaled, batch_size=AE_BATCH_SIZE, verbose=1)
    val_latent_features = encoder.predict(X_val_scaled, batch_size=AE_BATCH_SIZE, verbose=1)
    test_latent_features = encoder.predict(X_test_scaled, batch_size=AE_BATCH_SIZE, verbose=1)

    # Create column names for latent features
    ae_factor_cols = [f'ae_factor_{i}' for i in range(ENCODING_DIM)]

    # Save latent features with identifiers
    print("Saving latent features...")
    train_ae_factors_df = pd.DataFrame(train_latent_features, columns=ae_factor_cols)
    train_ae_factors_df[ID_COLUMN] = train_df_full[ID_COLUMN].values
    train_ae_factors_df[DATE_COLUMN] = train_df_full[DATE_COLUMN].values
    train_ae_factors_df = train_ae_factors_df[[ID_COLUMN, DATE_COLUMN] + ae_factor_cols] # Reorder
    train_ae_factors_df.to_parquet(TRAIN_AE_FACTORS_OUTPUT_PATH, index=False)
    print(f"Train AE factors saved to {TRAIN_AE_FACTORS_OUTPUT_PATH}")

    val_ae_factors_df = pd.DataFrame(val_latent_features, columns=ae_factor_cols)
    val_ae_factors_df[ID_COLUMN] = val_df_full[ID_COLUMN].values
    val_ae_factors_df[DATE_COLUMN] = val_df_full[DATE_COLUMN].values
    val_ae_factors_df = val_ae_factors_df[[ID_COLUMN, DATE_COLUMN] + ae_factor_cols]
    val_ae_factors_df.to_parquet(VAL_AE_FACTORS_OUTPUT_PATH, index=False)
    print(f"Validation AE factors saved to {VAL_AE_FACTORS_OUTPUT_PATH}")

    test_ae_factors_df = pd.DataFrame(test_latent_features, columns=ae_factor_cols)
    test_ae_factors_df[ID_COLUMN] = test_df_full[ID_COLUMN].values
    test_ae_factors_df[DATE_COLUMN] = test_df_full[DATE_COLUMN].values
    test_ae_factors_df = test_ae_factors_df[[ID_COLUMN, DATE_COLUMN] + ae_factor_cols]
    test_ae_factors_df.to_parquet(TEST_AE_FACTORS_OUTPUT_PATH, index=False)
    print(f"Test AE factors saved to {TEST_AE_FACTORS_OUTPUT_PATH}")

    print("--- Autoencoder Factor Generation Complete ---")

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    # Consider adding other seeds for reproducibility if other libraries are used for randomness
    main() 