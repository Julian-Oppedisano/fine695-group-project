import pytest
import pandas as pd
import os

# Define file paths relative to the project root
# Assuming tests are run from the project root directory
ORIGINAL_PANEL_PATH = os.path.join("data", "raw", "course_panel.parquet")
LAGGED_FEATURES_PATH = os.path.join("data", "processed", "ml_features_lagged.parquet")

# Choose a few columns that are known to be part of the 'potential_predictors' list in make_features.py
# and thus should have been lagged. We pick some that were likely in the original panel.
# Avoid 'extra features' for this specific test if their original unlagged version isn't in ORIGINAL_PANEL_PATH easily.
# Example: 'age' or 'at_gr1' were identified as existing in course_panel.parquet
TEST_PREDICTOR_COLUMNS = ['age', 'at_gr1', 'be_me', 'market_equity'] 

@pytest.fixture(scope="module") # Load data once per test module
def loaded_data():
    """Pytest fixture to load original and lagged dataframes."""
    if not os.path.exists(ORIGINAL_PANEL_PATH):
        pytest.fail(f"Original panel file not found at: {ORIGINAL_PANEL_PATH}")
    if not os.path.exists(LAGGED_FEATURES_PATH):
        pytest.fail(f"Lagged features file not found at: {LAGGED_FEATURES_PATH}")
    
    try:
        original_df = pd.read_parquet(ORIGINAL_PANEL_PATH)
        lagged_df = pd.read_parquet(LAGGED_FEATURES_PATH)
        
        # Ensure date columns are datetime objects for proper comparison and manipulation
        original_df['date'] = pd.to_datetime(original_df['date'])
        lagged_df['date'] = pd.to_datetime(lagged_df['date'])
        
        return original_df, lagged_df
    except Exception as e:
        pytest.fail(f"Error reading Parquet files: {e}")

def test_predictor_lagging_no_lookahead(loaded_data):
    """
    Test that predictor values in the lagged DataFrame correspond to the
    original predictor values from the previous month, ensuring no look-ahead bias.
    """
    original_df, lagged_df = loaded_data

    # Take a small sample of permnos to test to keep test runtime reasonable
    sample_permnos = original_df['permno'].drop_duplicates().sample(n=min(5, original_df['permno'].nunique()), random_state=42)

    for permno_to_test in sample_permnos:
        original_stock_data = original_df[original_df['permno'] == permno_to_test].set_index('date').sort_index()
        lagged_stock_data = lagged_df[lagged_df['permno'] == permno_to_test].set_index('date').sort_index()

        if lagged_stock_data.empty:
            continue # Should not happen if permno is in original and processing was correct

        # Test for a few dates for each stock (e.g., 2nd and a few random later available dates)
        # Avoid the very first date for original_stock_data as it won't have a t-1
        test_dates_for_lagged_df = lagged_stock_data.index.drop_duplicates()
        
        # We need dates in lagged_df that have a corresponding t-1 in original_df
        # Consider dates from the 2nd observation onwards in lagged_df
        if len(test_dates_for_lagged_df) < 2:
            continue # Not enough data points for this permno to test lag properly

        # Select a few dates to test more thoroughly from the lagged data
        # Ensure these dates also allow for a t-1 lookup in original data
        dates_to_check = pd.Series(test_dates_for_lagged_df[1:]).sample(n=min(3, len(test_dates_for_lagged_df)-1), random_state=42).tolist()
        if not dates_to_check and len(test_dates_for_lagged_df) >=2:
             dates_to_check = [test_dates_for_lagged_df[1]] # at least one if possible
        
        for current_date_in_lagged in dates_to_check:
            # We expect features at current_date_in_lagged to be from (current_date_in_lagged - 1 month)
            # For month-end data, a simple date offset might not be perfect.
            # A more robust way is to find the row in original_data whose date is the one immediately preceding current_date_in_lagged
            
            # Find the index (date) of the row in original_stock_data that immediately precedes current_date_in_lagged
            # This assumes dates are sorted, which they are by set_index().sort_index()
            potential_previous_dates = original_stock_data.index[original_stock_data.index < current_date_in_lagged]
            if potential_previous_dates.empty:
                # This means current_date_in_lagged is the first or second date for this stock in lagged_df, 
                # and there's no t-1 for it in original_df that we can map to for checking the lagged value.
                # Lagged values for the first observation of a stock will be NaN, which is correct.
                for predictor_col in TEST_PREDICTOR_COLUMNS:
                    if predictor_col in lagged_stock_data.columns:
                        lagged_value = lagged_stock_data.loc[current_date_in_lagged, predictor_col]
                        assert pd.isna(lagged_value), (
                            f"For permno {permno_to_test}, date {current_date_in_lagged}, predictor {predictor_col}: "
                            f"Expected NaN for first effective observation after lagging, but got {lagged_value}. "
                            f"(No preceding date found in original data for comparison)"
                        )
                continue # Move to next date or permno
            
            previous_date_in_original = potential_previous_dates[-1] # The closest preceding date

            for predictor_col in TEST_PREDICTOR_COLUMNS:
                if predictor_col not in original_stock_data.columns or predictor_col not in lagged_stock_data.columns:
                    print(f"Warning: Test column {predictor_col} not in both DFs for permno {permno_to_test}. Skipping.")
                    continue

                lagged_value = lagged_stock_data.loc[current_date_in_lagged, predictor_col]
                original_value_t_minus_1 = original_stock_data.loc[previous_date_in_original, predictor_col]
                
                # Compare, being mindful of NaNs (e.g. if original value was NaN, lagged should be NaN)
                if pd.isna(original_value_t_minus_1):
                    assert pd.isna(lagged_value), (
                        f"For permno {permno_to_test}, date {current_date_in_lagged}, predictor {predictor_col}: "
                        f"Original value at t-1 ({previous_date_in_original}) was NaN, expected lagged value to be NaN, but got {lagged_value}."
                    )
                else:
                    assert original_value_t_minus_1 == lagged_value, (
                        f"For permno {permno_to_test}, date {current_date_in_lagged}, predictor {predictor_col}: "
                        f"Lagged value {lagged_value} does not match original value at t-1 ({previous_date_in_original}), which was {original_value_t_minus_1}."
                    ) 