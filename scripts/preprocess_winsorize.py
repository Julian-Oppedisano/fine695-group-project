import pandas as pd
import numpy as np
import os

# Define file paths
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features.parquet')
VALIDATION_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features.parquet')
TEST_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features.parquet')

# Output paths for winsorized data
TRAIN_WINSORIZED_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_winsorized.parquet')
VALIDATION_WINSORIZED_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_winsorized.parquet')
TEST_WINSORIZED_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_winsorized.parquet')

# Winsorization thresholds
LOWER_PERCENTILE = 0.01
UPPER_PERCENTILE = 0.99

def get_predictor_columns(df):
    """Helper function to identify predictor columns."""
    # Columns to exclude (identifiers, targets, date-related that aren't features like month dummies)
    # Assuming 'stock_exret' and/or 'ret_eom' are targets.
    # 'year', 'month' are used for grouping but not features themselves to be winsorized.
    # 'date' is crucial for time series, also not a feature to winsorize here.
    # Seasonality dummies (month_2...month_12) are binary, not typically winsorized.
    identifier_cols = ['permno', 'stock_ticker', 'CUSIP', 'comp_name'] 
    target_cols = ['stock_exret', 'ret_eom']
    date_related_cols = ['date', 'year', 'month'] # 'year', 'month' might be created if not present
    seasonality_dummies = [f'month_{i}' for i in range(2, 13)]
    
    cols_to_exclude = identifier_cols + target_cols + date_related_cols + seasonality_dummies
    # Initial selection of potential predictors
    potential_predictor_cols = [col for col in df.columns if col not in cols_to_exclude]
    
    # Further refine to include only numeric types suitable for quantile calculation
    numeric_predictor_cols = df[potential_predictor_cols].select_dtypes(include=np.number).columns.tolist()
    
    print(f"Identified {len(numeric_predictor_cols)} numeric predictor columns for winsorization (out of {len(potential_predictor_cols)} potential). ")
    # print(f"First 5 numeric predictors: {numeric_predictor_cols[:5]}")
    return numeric_predictor_cols

def calculate_monthly_percentiles(df, predictor_cols, group_cols=['year', 'month']):
    """Calculates 1st and 99th percentiles for each predictor, by month, on the training set."""
    print(f"Calculating monthly percentiles on training data ({LOWER_PERCENTILE*100}th/{UPPER_PERCENTILE*100}th)... ")
    monthly_bounds = df.groupby(group_cols)[predictor_cols].quantile([LOWER_PERCENTILE, UPPER_PERCENTILE]).unstack()
    # Resulting MultiIndex columns: (predictor, percentile_level_1=0.01 or 0.99)
    # e.g. monthly_bounds[ ('age', 0.01) ] will give series indexed by (year, month)
    print("Monthly percentile calculation complete.")
    return monthly_bounds

def winsorize_dataframe(df, predictor_cols, monthly_bounds, group_cols=['year', 'month']):
    """Applies winsorization to a DataFrame using pre-calculated monthly bounds."""
    print(f"Applying winsorization... Original shape: {df.shape}")
    df_winsorized = df.copy()

    # Ensure year and month are present for merging/joining bounds
    if 'year' not in df_winsorized.columns:
        df_winsorized['year'] = df_winsorized['date'].dt.year
    if 'month' not in df_winsorized.columns:
        df_winsorized['month'] = df_winsorized['date'].dt.month

    # Iterate over each predictor column to apply winsorization
    for col in predictor_cols:
        if col not in df_winsorized.columns:
            print(f"Warning: Predictor column '{col}' not found in DataFrame during winsorization. Skipping.")
            continue
            
        # print(f"Winsorizing column: {col}") # Verbose
        # Prepare bounds for this specific column
        # The monthly_bounds df has MultiIndex columns like (predictor, percentile_value)
        # e.g., ('age', 0.01) and ('age', 0.99)
        lower_bound_col = (col, LOWER_PERCENTILE)
        upper_bound_col = (col, UPPER_PERCENTILE)

        if lower_bound_col not in monthly_bounds.columns or upper_bound_col not in monthly_bounds.columns:
            print(f"Warning: Percentile bounds for column '{col}' not found in monthly_bounds. Skipping winsorization for this column.")
            continue

        current_col_bounds = monthly_bounds[[lower_bound_col, upper_bound_col]].copy()
        current_col_bounds.columns = ['lower', 'upper'] # Simplify column names
        
        # Merge these bounds back to the df_winsorized for this column
        # Ensure df_winsorized has 'year' and 'month' columns available for merging
        df_winsorized = pd.merge(df_winsorized, current_col_bounds, on=group_cols, how='left', suffixes=('', '_bounds'))
        # df_winsorized = df_winsorized.set_index(group_cols, append=True)
        # df_winsorized = df_winsorized.join(current_col_bounds, how='left')
        # df_winsorized = df_winsorized.reset_index(level=group_cols)

        # Apply clipping using the merged columns
        # If current_col_bounds had columns 'lower' and 'upper', they are now in df_winsorized
        df_winsorized[col] = df_winsorized[col].clip(lower=df_winsorized['lower'], upper=df_winsorized['upper'])
        
        # Drop the temporary lower/upper bound columns
        df_winsorized.drop(columns=['lower', 'upper'], inplace=True)

    print(f"Winsorization application complete. Shape after: {df_winsorized.shape}")
    return df_winsorized

def main():
    print("--- Starting Winsorization Process ---")
    # Load data
    print("Loading datasets...")
    try:
        train_df = pd.read_parquet(TRAIN_INPUT_PATH)
        validation_df = pd.read_parquet(VALIDATION_INPUT_PATH)
        test_df = pd.read_parquet(TEST_INPUT_PATH)
        print("Datasets loaded successfully.")
        print(f"Train shape: {train_df.shape}, Validation shape: {validation_df.shape}, Test shape: {test_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure date columns are datetime and year/month are available
    for df_name, df in zip(["Train", "Validation", "Test"], [train_df, validation_df, test_df]):
        if 'date' not in df.columns:
            print(f"Error: 'date' column missing in {df_name} df.")
            return
        df['date'] = pd.to_datetime(df['date'])
        if 'year' not in df.columns: df['year'] = df['date'].dt.year
        if 'month' not in df.columns: df['month'] = df['date'].dt.month

    # Identify predictor columns (from train_df, should be same for others)
    predictor_cols = get_predictor_columns(train_df)
    if not predictor_cols:
        print("Error: No predictor columns identified. Aborting winsorization.")
        return

    # 1. Calculate monthly percentiles on the TRAIN set
    monthly_perc_bounds = calculate_monthly_percentiles(train_df, predictor_cols)

    # 2. Apply these bounds to winsorize TRAIN, VALIDATION, and TEST sets
    print("\nWinsorizing Train set...")
    train_winsorized_df = winsorize_dataframe(train_df, predictor_cols, monthly_perc_bounds)
    
    print("\nWinsorizing Validation set...")
    validation_winsorized_df = winsorize_dataframe(validation_df, predictor_cols, monthly_perc_bounds)
    
    print("\nWinsorizing Test set...")
    test_winsorized_df = winsorize_dataframe(test_df, predictor_cols, monthly_perc_bounds)

    # Save winsorized data
    print("\nSaving winsorized datasets...")
    try:
        train_winsorized_df.to_parquet(TRAIN_WINSORIZED_OUTPUT_PATH, index=False)
        print(f"Winsorized train set saved to {TRAIN_WINSORIZED_OUTPUT_PATH}. Shape: {train_winsorized_df.shape}")
        
        validation_winsorized_df.to_parquet(VALIDATION_WINSORIZED_OUTPUT_PATH, index=False)
        print(f"Winsorized validation set saved to {VALIDATION_WINSORIZED_OUTPUT_PATH}. Shape: {validation_winsorized_df.shape}")
        
        test_winsorized_df.to_parquet(TEST_WINSORIZED_OUTPUT_PATH, index=False)
        print(f"Winsorized test set saved to {TEST_WINSORIZED_OUTPUT_PATH}. Shape: {test_winsorized_df.shape}")
        print("\nWinsorization and saving complete.")
    except Exception as e:
        print(f"Error saving one or more winsorized data splits: {e}")

if __name__ == "__main__":
    main() 