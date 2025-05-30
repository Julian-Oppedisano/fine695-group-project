import pandas as pd
import numpy as np
import os

# Define file paths
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_NORMALIZED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_normalized.parquet')
VALIDATION_NORMALIZED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_normalized.parquet')
TEST_NORMALIZED_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_normalized.parquet')

# Output paths for imputed data
TRAIN_IMPUTED_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VALIDATION_IMPUTED_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_IMPUTED_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

def get_predictor_columns(df):
    """Helper function to identify numeric predictor columns."""
    identifier_cols = ['permno', 'stock_ticker', 'CUSIP', 'comp_name'] 
    target_cols = ['stock_exret', 'ret_eom']
    date_related_cols = ['date', 'year', 'month'] 
    seasonality_dummies = [f'month_{i}' for i in range(2, 13)]
    cols_to_exclude = identifier_cols + target_cols + date_related_cols + seasonality_dummies
    potential_predictor_cols = [col for col in df.columns if col not in cols_to_exclude]
    numeric_predictor_cols = df[potential_predictor_cols].select_dtypes(include=np.number).columns.tolist()
    print(f"Identified {len(numeric_predictor_cols)} numeric predictor columns for imputation.")
    return numeric_predictor_cols

def calculate_monthly_medians(df, predictor_cols, group_cols=['year', 'month']):
    """Calculates median for each predictor, by month, on the training set."""
    print("Calculating monthly medians on training data...")
    grouped = df.groupby(group_cols)[predictor_cols]
    monthly_medians = grouped.median()
    # Rename columns for clarity when merging, e.g., age_median
    monthly_medians = monthly_medians.rename(columns=lambda c: f"{c}_median_train_monthly")
    print("Monthly median calculation complete.")
    return monthly_medians

def calculate_overall_train_medians(df, predictor_cols):
    """Calculates overall median for each predictor on the training set."""
    print("Calculating overall medians on training data (for fallback)..." )
    overall_medians = df[predictor_cols].median()
    # Convert Series to DataFrame with _median_train_overall suffix
    overall_medians_df = overall_medians.to_frame().T.rename(columns=lambda c: f"{c}_median_train_overall")
    print("Overall median calculation complete.")
    return overall_medians_df # Return as a one-row DataFrame for easier broadcasting or merging

def impute_dataframe(df, predictor_cols, monthly_medians, overall_medians_df, group_cols=['year', 'month']):
    """Applies imputation to a DataFrame using pre-calculated monthly and overall medians."""
    print(f"Applying imputation... Original shape: {df.shape}, NaNs before: {df[predictor_cols].isnull().sum().sum()}")
    df_imputed = df.copy()

    if 'year' not in df_imputed.columns: df_imputed['year'] = df_imputed['date'].dt.year
    if 'month' not in df_imputed.columns: df_imputed['month'] = df_imputed['date'].dt.month

    # Merge monthly medians
    df_imputed = pd.merge(df_imputed, monthly_medians, on=group_cols, how='left')

    for col in predictor_cols:
        if col not in df_imputed.columns:
            print(f"Warning: Predictor column '{col}' not found. Skipping imputation.")
            continue
        
        median_col_monthly = f"{col}_median_train_monthly"
        overall_median_val = overall_medians_df.get(f"{col}_median_train_overall", pd.Series(0.0))[0] # Default to 0.0 if not found
        
        if median_col_monthly not in df_imputed.columns:
            print(f"Warning: Monthly median column for '{col}' ('{median_col_monthly}') not found. Using overall median.")
            df_imputed[col] = df_imputed[col].fillna(overall_median_val)
            continue

        # Impute using monthly median first
        df_imputed[col] = df_imputed[col].fillna(df_imputed[median_col_monthly])
        
        # If any NaNs remain (e.g., new month/year group not in train), use overall train median for that column
        if df_imputed[col].isnull().any():
            # print(f"Column '{col}' still has NaNs after monthly median imputation. Applying overall train median.")
            df_imputed[col] = df_imputed[col].fillna(overall_median_val)
            
    # Drop the temporary median columns
    monthly_median_cols_to_drop = [f"{c}_median_train_monthly" for c in predictor_cols]
    existing_monthly_median_cols_to_drop = [sc for sc in monthly_median_cols_to_drop if sc in df_imputed.columns]
    df_imputed.drop(columns=existing_monthly_median_cols_to_drop, inplace=True)

    print(f"Imputation application complete. Shape after: {df_imputed.shape}, NaNs after: {df_imputed[predictor_cols].isnull().sum().sum()}")
    return df_imputed

def main():
    print("--- Starting Imputation Process ---")
    print("Loading normalized datasets...")
    try:
        train_df = pd.read_parquet(TRAIN_NORMALIZED_INPUT_PATH)
        validation_df = pd.read_parquet(VALIDATION_NORMALIZED_INPUT_PATH)
        test_df = pd.read_parquet(TEST_NORMALIZED_INPUT_PATH)
        print("Normalized datasets loaded successfully.")
        print(f"Train shape: {train_df.shape}, Val shape: {validation_df.shape}, Test shape: {test_df.shape}")
    except Exception as e: print(f"Error loading data: {e}"); return

    for df_name, df in zip(["Train", "Val", "Test"], [train_df, validation_df, test_df]):
        if 'date' not in df.columns: print(f"Error: 'date' missing in {df_name}."); return
        df['date'] = pd.to_datetime(df['date'])
        if 'year' not in df.columns: df['year'] = df['date'].dt.year
        if 'month' not in df.columns: df['month'] = df['date'].dt.month

    predictor_cols = get_predictor_columns(train_df)
    if not predictor_cols: print("Error: No predictors identified. Aborting."); return

    monthly_train_medians = calculate_monthly_medians(train_df, predictor_cols)
    overall_train_medians = calculate_overall_train_medians(train_df, predictor_cols)

    print("\nImputing Train set...")
    train_imputed_df = impute_dataframe(train_df, predictor_cols, monthly_train_medians, overall_train_medians)
    print("\nImputing Validation set...")
    validation_imputed_df = impute_dataframe(validation_df, predictor_cols, monthly_train_medians, overall_train_medians)
    print("\nImputing Test set...")
    test_imputed_df = impute_dataframe(test_df, predictor_cols, monthly_train_medians, overall_train_medians)

    # Final check for NaNs in predictor columns
    for df_name, df_final in zip(["Train_Imputed", "Validation_Imputed", "Test_Imputed"], [train_imputed_df, validation_imputed_df, test_imputed_df]):
        final_nan_count = df_final[predictor_cols].isnull().sum().sum()
        if final_nan_count == 0:
            print(f"{df_name}: Confirmed no NaNs in predictor columns.")
        else:
            print(f"WARNING - {df_name}: {final_nan_count} NaNs still present in predictor columns.")
            # print(df_final[predictor_cols].isnull().sum()[df_final[predictor_cols].isnull().sum() > 0])

    print("\nSaving imputed datasets...")
    try:
        train_imputed_df.to_parquet(TRAIN_IMPUTED_OUTPUT_PATH, index=False)
        print(f"Imputed train saved: {TRAIN_IMPUTED_OUTPUT_PATH}. Shape: {train_imputed_df.shape}")
        validation_imputed_df.to_parquet(VALIDATION_IMPUTED_OUTPUT_PATH, index=False)
        print(f"Imputed val saved: {VALIDATION_IMPUTED_OUTPUT_PATH}. Shape: {validation_imputed_df.shape}")
        test_imputed_df.to_parquet(TEST_IMPUTED_OUTPUT_PATH, index=False)
        print(f"Imputed test saved: {TEST_IMPUTED_OUTPUT_PATH}. Shape: {test_imputed_df.shape}")
        print("\nImputation and saving complete.")
    except Exception as e: print(f"Error saving imputed data: {e}")

if __name__ == "__main__":
    main() 