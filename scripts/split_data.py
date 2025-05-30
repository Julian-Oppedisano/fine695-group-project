import pandas as pd
import os

# Define file paths
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
LAGGED_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'ml_features_lagged.parquet')

TRAIN_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features.parquet')
VALIDATION_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features.parquet')
TEST_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features.parquet')

# Define date ranges for splits
TRAIN_START_DATE = '2000-01-01'
TRAIN_END_DATE = '2016-12-31'
VALIDATION_START_DATE = '2017-01-01'
VALIDATION_END_DATE = '2019-12-31'
TEST_START_DATE = '2020-01-01'
TEST_END_DATE = '2023-12-31' # As per outline, adjusted to end of 2023

def main():
    print(f"Loading lagged features from {LAGGED_FEATURES_PATH}...")
    if not os.path.exists(LAGGED_FEATURES_PATH):
        print(f"Error: Lagged features file not found at {LAGGED_FEATURES_PATH}")
        return
    
    try:
        features_df = pd.read_parquet(LAGGED_FEATURES_PATH)
        print("Lagged features loaded successfully.")
        print(f"Full dataset shape: {features_df.shape}")
        if 'date' in features_df.columns:
            features_df['date'] = pd.to_datetime(features_df['date'])
            print(f"Date range in loaded data: {features_df['date'].min()} to {features_df['date'].max()}")
        else:
            print("Error: 'date' column not found after loading.")
            return
    except Exception as e:
        print(f"Error reading lagged features file or processing date: {e}")
        return

    # Ensure 'date' column is in datetime format
    if 'date' not in features_df.columns:
        print("Error: 'date' column not found in the features DataFrame.")
        return

    # Perform time-based splits
    print(f"\nSplitting data...")
    train_df = features_df[(features_df['date'] >= TRAIN_START_DATE) & (features_df['date'] <= TRAIN_END_DATE)]
    validation_df = features_df[(features_df['date'] >= VALIDATION_START_DATE) & (features_df['date'] <= VALIDATION_END_DATE)]
    test_df = features_df[(features_df['date'] >= TEST_START_DATE) & (features_df['date'] <= TEST_END_DATE)]

    # Save the splits
    print(f"\nSaving data splits to {PROCESSED_DIR}...")
    try:
        train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)
        print(f"Train set saved to {TRAIN_OUTPUT_PATH}. Shape: {train_df.shape}")
        
        validation_df.to_parquet(VALIDATION_OUTPUT_PATH, index=False)
        print(f"Validation set saved to {VALIDATION_OUTPUT_PATH}. Shape: {validation_df.shape}")
        
        test_df.to_parquet(TEST_OUTPUT_PATH, index=False)
        print(f"Test set saved to {TEST_OUTPUT_PATH}. Shape: {test_df.shape}")
        print("\nData splitting and saving complete.")
    except Exception as e:
        print(f"Error saving one or more data splits: {e}")

if __name__ == "__main__":
    main() 