import pandas as pd
import numpy as np
import os

# Define file paths
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_WINSORIZED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_winsorized.parquet')
VALIDATION_WINSORIZED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_winsorized.parquet')
TEST_WINSORIZED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_winsorized.parquet')

# Output paths for normalized data
TRAIN_NORMALIZED_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_normalized.parquet')
VALIDATION_NORMALIZED_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_normalized.parquet')
TEST_NORMALIZED_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_normalized.parquet')

def get_predictor_columns(df):
    """Helper function to identify numeric predictor columns."""
    identifier_cols = ['permno', 'stock_ticker', 'CUSIP', 'comp_name'] 
    target_cols = ['stock_exret', 'ret_eom']
    date_related_cols = ['date', 'year', 'month'] 
    seasonality_dummies = [f'month_{i}' for i in range(2, 13)]
    cols_to_exclude = identifier_cols + target_cols + date_related_cols + seasonality_dummies
    potential_predictor_cols = [col for col in df.columns if col not in cols_to_exclude]
    numeric_predictor_cols = df[potential_predictor_cols].select_dtypes(include=np.number).columns.tolist()
    print(f"Identified {len(numeric_predictor_cols)} numeric predictor columns for normalization.")
    return numeric_predictor_cols

def calculate_monthly_stats(df, predictor_cols, group_cols=['year', 'month']):
    """Calculates mean and std for each predictor, by month, on the training set."""
    print("Calculating monthly mean and std on training data...")
    # Calculate mean and std deviation in a single pass if possible, or separately
    grouped = df.groupby(group_cols)[predictor_cols]
    monthly_means = grouped.mean()
    monthly_stds = grouped.std()
    
    # Rename columns for clarity when merging, e.g., age_mean, age_std
    monthly_means = monthly_means.rename(columns=lambda c: f"{c}_mean")
    monthly_stds = monthly_stds.rename(columns=lambda c: f"{c}_std")
    
    monthly_stats = pd.concat([monthly_means, monthly_stds], axis=1)
    print("Monthly mean and std calculation complete.")
    return monthly_stats

def normalize_dataframe(df, predictor_cols, monthly_stats, group_cols=['year', 'month']):
    """Applies normalization to a DataFrame using pre-calculated monthly stats."""
    print(f"Applying normalization... Original shape: {df.shape}")
    df_normalized = df.copy()

    if 'year' not in df_normalized.columns: df_normalized['year'] = df_normalized['date'].dt.year
    if 'month' not in df_normalized.columns: df_normalized['month'] = df_normalized['date'].dt.month

    # Merge the stats for all predictor columns at once
    df_normalized = pd.merge(df_normalized, monthly_stats, on=group_cols, how='left')

    for col in predictor_cols:
        if col not in df_normalized.columns:
            print(f"Warning: Predictor column '{col}' not found. Skipping normalization.")
            continue
        
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"

        if mean_col not in df_normalized.columns or std_col not in df_normalized.columns:
            print(f"Warning: Stats columns for '{col}' ('{mean_col}', '{std_col}') not found after merge. Skipping normalization for this column.")
            continue
        
        # Normalize: (value - mean) / std
        # Handle std == 0 (or very close to 0) to avoid division by zero or large numbers.
        # If std is 0, it means all values in that group were the same. (value - mean) will be 0.
        # So, the normalized value should be 0.
        df_normalized[col] = (df_normalized[col] - df_normalized[mean_col]) / df_normalized[std_col]
        df_normalized[col] = df_normalized[col].fillna(0) # Fill NaNs from division by zero std with 0
        # Also, if a std was NaN (e.g. single observation in a group), result will be NaN, fill with 0.

    # Drop the temporary mean and std columns for all predictors
    stat_cols_to_drop = [f"{c}_mean" for c in predictor_cols] + [f"{c}_std" for c in predictor_cols]
    # Only drop those that actually exist to avoid KeyErrors if some were skipped
    existing_stat_cols_to_drop = [sc for sc in stat_cols_to_drop if sc in df_normalized.columns]
    df_normalized.drop(columns=existing_stat_cols_to_drop, inplace=True)

    print(f"Normalization application complete. Shape after: {df_normalized.shape}")
    return df_normalized

def main():
    print("--- Starting Normalization Process ---")
    print("Loading winsorized datasets...")
    try:
        train_df = pd.read_parquet(TRAIN_WINSORIZED_INPUT_PATH)
        validation_df = pd.read_parquet(VALIDATION_WINSORIZED_INPUT_PATH)
        test_df = pd.read_parquet(TEST_WINSORIZED_INPUT_PATH)
        print("Winsorized datasets loaded successfully.")
        print(f"Train shape: {train_df.shape}, Val shape: {validation_df.shape}, Test shape: {test_df.shape}")
    except Exception as e: print(f"Error loading data: {e}"); return

    for df_name, df in zip(["Train", "Val", "Test"], [train_df, validation_df, test_df]):
        if 'date' not in df.columns: print(f"Error: 'date' missing in {df_name}."); return
        df['date'] = pd.to_datetime(df['date'])
        if 'year' not in df.columns: df['year'] = df['date'].dt.year
        if 'month' not in df.columns: df['month'] = df['date'].dt.month

    predictor_cols = get_predictor_columns(train_df)
    if not predictor_cols: print("Error: No predictors identified. Aborting."); return

    monthly_train_stats = calculate_monthly_stats(train_df, predictor_cols)

    print("\nNormalizing Train set...")
    train_normalized_df = normalize_dataframe(train_df, predictor_cols, monthly_train_stats)
    print("\nNormalizing Validation set...")
    validation_normalized_df = normalize_dataframe(validation_df, predictor_cols, monthly_train_stats)
    print("\nNormalizing Test set...")
    test_normalized_df = normalize_dataframe(test_df, predictor_cols, monthly_train_stats)

    print("\nSaving normalized datasets...")
    try:
        train_normalized_df.to_parquet(TRAIN_NORMALIZED_OUTPUT_PATH, index=False)
        print(f"Normalized train saved: {TRAIN_NORMALIZED_OUTPUT_PATH}. Shape: {train_normalized_df.shape}")
        validation_normalized_df.to_parquet(VALIDATION_NORMALIZED_OUTPUT_PATH, index=False)
        print(f"Normalized val saved: {VALIDATION_NORMALIZED_OUTPUT_PATH}. Shape: {validation_normalized_df.shape}")
        test_normalized_df.to_parquet(TEST_NORMALIZED_OUTPUT_PATH, index=False)
        print(f"Normalized test saved: {TEST_NORMALIZED_OUTPUT_PATH}. Shape: {test_normalized_df.shape}")
        print("\nNormalization and saving complete.")
    except Exception as e: print(f"Error saving normalized data: {e}")

if __name__ == "__main__":
    main() 