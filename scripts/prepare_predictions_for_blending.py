import pandas as pd
import os
import glob

# --- Configuration ---
SOURCE_PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
TARGET_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# Files to process from the source directory and their intended output names
# We are focusing on files that seem to be test predictions based on naming
FILES_TO_PROCESS = {
    "predictions_catboost.parquet": "pred_catboost.csv",
    "predictions_xgboost.parquet": "pred_xgboost.csv",
    # Add other files here if they are test predictions and need conversion
    # e.g., "predictions_tabtransformer.parquet": "pred_tabtransformer.csv" if it exists and is wanted
}

ID_COLUMNS = ['permno', 'date']
# List of possible names for the prediction column in the source parquet files
POSSIBLE_PREDICTION_COL_NAMES = ['predicted_stock_exret', 'prediction', 'pred']
OUTPUT_PREDICTION_COL_NAME = 'prediction' # Standardized name in the output CSV

def find_prediction_column(df, possible_names):
    """Finds the first matching prediction column from a list of possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def convert_parquet_to_csv(source_file_path, target_file_path, id_cols, pred_col_options, output_pred_col):
    """Reads a parquet prediction file, extracts key columns, and saves as CSV."""
    try:
        df = pd.read_parquet(source_file_path)
        print(f"Successfully read: {source_file_path}")
    except Exception as e:
        print(f"Error reading parquet file {source_file_path}: {e}")
        return False

    # Check for ID columns
    missing_id_cols = [col for col in id_cols if col not in df.columns]
    if missing_id_cols:
        print(f"Warning: File {source_file_path} is missing ID columns: {missing_id_cols}. Skipping.")
        return False

    # Find the actual prediction column name
    actual_pred_col = find_prediction_column(df, pred_col_options)
    if not actual_pred_col:
        print(f"Warning: Could not find a prediction column in {source_file_path} from options: {pred_col_options}. Skipping.")
        return False
    
    print(f"Found prediction column '{actual_pred_col}' in {source_file_path}")
    # --- Add unique value check ---
    num_unique_preds = df[actual_pred_col].nunique()
    print(f"Number of unique values in prediction column '{actual_pred_col}': {num_unique_preds}")
    if num_unique_preds <= 1:
        print(f"Warning: Prediction column '{actual_pred_col}' in {source_file_path} has {num_unique_preds} unique value(s). This will lead to NaN/undefined correlations.")
    # --- End unique value check ---

    # Create the output DataFrame
    try:
        output_df = df[id_cols + [actual_pred_col]].copy()
        output_df.rename(columns={actual_pred_col: output_pred_col}, inplace=True)
    except KeyError as e:
        print(f"Error selecting or renaming columns for {source_file_path}: {e}. Skipping.")
        return False

    # Save to CSV
    try:
        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
        output_df.to_csv(target_file_path, index=False)
        print(f"Successfully converted and saved to: {target_file_path}")
        return True
    except Exception as e:
        print(f"Error saving CSV file {target_file_path}: {e}")
        return False

def main():
    print("--- Starting Prediction File Preparation Script ---")
    os.makedirs(TARGET_RESULTS_DIR, exist_ok=True)
    
    success_count = 0
    failure_count = 0

    if not FILES_TO_PROCESS:
        print("No files configured in FILES_TO_PROCESS. Exiting.")
        return

    for source_filename, target_filename in FILES_TO_PROCESS.items():
        source_path = os.path.join(SOURCE_PREDICTIONS_DIR, source_filename)
        target_path = os.path.join(TARGET_RESULTS_DIR, target_filename)

        if not os.path.exists(source_path):
            print(f"Source file not found: {source_path}. Skipping.")
            failure_count += 1
            continue
        
        if convert_parquet_to_csv(source_path, target_path, ID_COLUMNS, POSSIBLE_PREDICTION_COL_NAMES, OUTPUT_PREDICTION_COL_NAME):
            success_count += 1
        else:
            failure_count += 1
            
    print(f"--- Prediction File Preparation Complete ---")
    print(f"Successfully converted files: {success_count}")
    print(f"Failed conversions: {failure_count}")

if __name__ == "__main__":
    main() 