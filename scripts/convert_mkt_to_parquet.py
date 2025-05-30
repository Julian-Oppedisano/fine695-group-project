import pandas as pd
import os

# Define file paths relative to the project root
# Assuming the script is in fine695-group-project/scripts/
# and the data is in fine695-group-project/data/raw/
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'mkt_ind1 (1).csv')
PARQUET_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'mkt.parquet')
EXPECTED_COLUMNS = ['RF', 'year', 'month', 'sp_ret']
MIN_ROWS = 300

def convert_mkt_csv_to_parquet(csv_path, parquet_path, columns_to_keep):
    """
    Converts the market data CSV to Parquet, keeping specified columns.

    Args:
        csv_path (str): Path to the input CSV file.
        parquet_path (str): Path to the output Parquet file.
        columns_to_keep (list): List of column names to keep.
    """
    print(f"Starting conversion of {csv_path} to {parquet_path}...")

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return 0, False

    try:
        df = pd.read_csv(csv_path)
        
        # Check if all expected columns are present
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing expected columns in CSV: {missing_cols}. Available columns: {df.columns.tolist()}")
            return 0, False
            
        df_selected = df[columns_to_keep]
        
        # Ensure the directory for the parquet file exists
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        
        df_selected.to_parquet(parquet_path, engine='pyarrow', index=False)
        num_rows = len(df_selected)
        print(f"Conversion complete. {num_rows} rows processed.")
        print(f"Parquet file saved to: {parquet_path}")
        
        return num_rows, num_rows >= MIN_ROWS

    except Exception as e:
        print(f"Error during conversion: {e}")
        return 0, False

if __name__ == "__main__":
    row_count, condition_met = convert_mkt_csv_to_parquet(CSV_FILE_PATH, PARQUET_FILE_PATH, EXPECTED_COLUMNS)
    
    if row_count > 0:
        print(f"Total rows in mkt.parquet: {row_count}")
        if condition_met:
            print(f"Parquet file has at least {MIN_ROWS} rows. Task 2.3 condition met.")
        else:
            print(f"Warning: Parquet file has {row_count} rows, which is less than the required {MIN_ROWS}.")
    else:
        print("Failed to create mkt.parquet.") 