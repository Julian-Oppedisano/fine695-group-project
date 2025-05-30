import pandas as pd
import numpy as np
import os

# Define file paths
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
COURSE_PANEL_PATH = os.path.join(RAW_DIR, 'course_panel.parquet')
FACTOR_DESCRIPTIONS_PATH = os.path.join(RAW_DIR, 'factor_variable_descriptions_complete.csv')
OUTPUT_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'features_2000-2024.parquet')
LAGGED_OUTPUT_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'ml_features_lagged.parquet')

# --- Placeholder functions for 147 Factor Calculations ---
# These will need to be implemented based on academic definitions
# and available columns in course_panel.parquet

def calculate_age(df):
    # Example: df['age'] = (df['current_date_col'] - df['ipo_date_col']).dt.days / 365.25
    # This is a placeholder - actual logic needed
    if 'age' not in df.columns and 'placeholder_date_col' in df.columns and 'placeholder_ipo_date_col' in df.columns:
        # df['age'] = (pd.to_datetime(df['placeholder_date_col']) - pd.to_datetime(df['placeholder_ipo_date_col'])).dt.days / 365.25
        df['age'] = np.nan # Actual calculation needed
        print("Calculated placeholder for age")
    elif 'age' in df.columns:
        print("Column 'age' already exists or calculation previously attempted.")
    else:
        print("Skipping 'age': Missing necessary source columns (placeholder_date_col, placeholder_ipo_date_col) or 'age' itself.")
    return df

# ... Add placeholder functions for all 147 factors ...
# Example for another factor:
# def calculate_at_be(df):
#     if 'total_assets' in df.columns and 'book_equity' in df.columns:
#         df['at_be'] = df['total_assets'] / df['book_equity']
#         print("Calculated at_be")
#     else:
#         df['at_be'] = np.nan
#         print("Skipping 'at_be': Missing source columns (total_assets, book_equity)")
#     return df

# --- Functions for 'Extra' Feature Engineering ---
def calculate_earnings_surprise(df):
    # Needs 'eps_actual_col' and 'eps_meanest_col'
    if 'eps_actual_col' in df.columns and 'eps_meanest_col' in df.columns:
        df['earnings_surprise'] = df['eps_actual_col'] - df['eps_meanest_col']
        print("Calculated earnings_surprise")
    else:
        df['earnings_surprise'] = np.nan
        print("Skipping 'earnings_surprise': Missing source columns (eps_actual_col, eps_meanest_col)")
    return df

def calculate_accruals(df):
    # Needs specific definition and source columns
    # Example: (Net Income - CFO - Investing Cash Flow) / Avg Total Assets
    # df['accruals'] = ...
    df['accruals'] = np.nan # Placeholder
    print("Calculated placeholder for accruals")
    return df

def calculate_asset_growth(df):
    # Needs 'total_assets_col' and a date column for year-over-year growth
    # df['asset_growth'] = df.groupby('permno_col')['total_assets_col'].pct_change(periods=1) # Assuming annual data or needs adjustment
    df['asset_growth'] = np.nan # Placeholder
    print("Calculated placeholder for asset_growth")
    return df

def calculate_momentum(df, months):
    # Needs a 'returns_col' (e.g., 'stock_exret') and assumes 'date_col' and 'permno_col' (or other stock_id)
    # This is a simplified example for cumulative returns. True momentum often involves lags.
    if 'stock_exret' in df.columns and 'date_col' in df.columns and 'permno_col' in df.columns:
        # Ensure date is datetime and sorted
        # df['date_col'] = pd.to_datetime(df['date_col'])
        # df = df.sort_values(by=['permno_col', 'date_col'])
        # df[f'momentum_{months}m'] = df.groupby('permno_col')['stock_exret'].rolling(window=months, min_periods=max(1,months//2)).sum().reset_index(level=0, drop=True)
        df[f'momentum_{months}m'] = np.nan # Placeholder - complex logic, needs date and permno columns identified
        print(f"Calculated placeholder for momentum_{months}m")
    else:
        df[f'momentum_{months}m'] = np.nan
        print(f"Skipping momentum_{months}m: Missing source columns (stock_exret, date_col, permno_col)")
    return df

def calculate_quality_roe(df):
    # Needs 'net_income_col' and 'book_equity_col'
    if 'net_income_col' in df.columns and 'book_equity_col' in df.columns:
        # df['roe'] = df['net_income_col'] / df['book_equity_col']
        df['roe'] = np.nan # Placeholder
        print("Calculated placeholder for roe")
    else:
        df['roe'] = np.nan
        print("Skipping 'roe': Missing source columns (net_income_col, book_equity_col)")
    return df

def calculate_quality_cfo_assets(df):
    # Needs 'cfo_col' and 'total_assets_col'
    if 'cfo_col' in df.columns and 'total_assets_col' in df.columns:
        # df['cfo_assets'] = df['cfo_col'] / df['total_assets_col']
        df['cfo_assets'] = np.nan # Placeholder
        print("Calculated placeholder for cfo_assets")
    else:
        df['cfo_assets'] = np.nan
        print("Skipping 'cfo_assets': Missing source columns (cfo_col, total_assets_col)")
    return df

def calculate_seasonality_dummies(df):
    # Needs a 'date_col'
    if 'date_col' in df.columns:
        # df['month'] = pd.to_datetime(df['date_col']).dt.month
        # month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True) # drop_first to avoid multicollinearity
        # df = pd.concat([df, month_dummies], axis=1)
        # df = df.drop(columns=['month'])
        print("Calculated placeholder for seasonality_dummies")
        # Add placeholder columns for now, as actual date_col is not confirmed
        for m in range(2, 13): # month_2 to month_12
             df[f'month_{m}'] = np.nan
    else:
        print("Skipping 'seasonality_dummies': Missing source column (date_col)")
    return df

def engineer_extra_features(df):
    print("\n--- Engineering 'Extra' Features ---")
    df_processed = df.copy()
    if 'date' in df_processed.columns:
        # Attempt to convert date column, assuming it might be YYYYMMDD integers or already datetime
        try:
            # If it's already datetime-like (e.g., from a previous load), this won't hurt
            # If it's integer YYYYMMDD, specify format
            if pd.api.types.is_integer_dtype(df_processed['date']):
                df_processed['date'] = pd.to_datetime(df_processed['date'], format='%Y%m%d')
            else: # Otherwise, assume it can be directly converted or is already datetime
                df_processed['date'] = pd.to_datetime(df_processed['date'])
            print(f"Date column converted. Min: {df_processed['date'].min()}, Max: {df_processed['date'].max()}")
        except Exception as e:
            print(f"Warning: 'date' column found but could not be converted to datetime: {e}. Critical features might be skipped or incorrect.")
            # Add placeholders for all extra features if date is missing or problematic
            for m in [1, 3, 6, 12]: df_processed[f'momentum_{m}m'] = np.nan
            for m_idx in range(2,13): df_processed[f'month_{m_idx}'] = np.nan
            df_processed['earnings_surprise'] = np.nan
            df_processed['accruals'] = np.nan
            df_processed['asset_growth'] = np.nan
            df_processed['quality_roe'] = np.nan
            df_processed['quality_cfo_assets'] = np.nan
            return df_processed # Return early if no date column
    else:
        print("Warning: 'date' column not found. Critical features will be skipped.")
        # Add placeholders for all extra features if date is missing
        # (Copying placeholder logic for consistency)
        for m in [1, 3, 6, 12]: df_processed[f'momentum_{m}m'] = np.nan
        for m_idx in range(2,13): df_processed[f'month_{m_idx}'] = np.nan
        df_processed['earnings_surprise'] = np.nan
        df_processed['accruals'] = np.nan
        df_processed['asset_growth'] = np.nan
        df_processed['quality_roe'] = np.nan
        df_processed['quality_cfo_assets'] = np.nan
        return df_processed # Return early if no date column

    if 'eps_actual' in df_processed.columns and 'eps_meanest' in df_processed.columns:
        df_processed['earnings_surprise'] = df_processed['eps_actual'] - df_processed['eps_meanest']
        print("Calculated: earnings_surprise")
    else: df_processed['earnings_surprise'] = np.nan; print("Placeholder: earnings_surprise")
    if 'taccruals_at' in df_processed.columns: 
        df_processed['accruals'] = df_processed['taccruals_at']; print("Assigned: accruals")
    else: df_processed['accruals'] = np.nan; print("Placeholder: accruals")
    if 'at_gr1' in df_processed.columns: 
        df_processed['asset_growth'] = df_processed['at_gr1']; print("Assigned: asset_growth")
    else: df_processed['asset_growth'] = np.nan; print("Placeholder: asset_growth")
    
    # Momentum calculation requires date index, so ensure df_processed.date is okay.
    if 'stock_exret' in df_processed.columns and 'permno' in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed['date']):
        # Create a temporary DataFrame with 'date' as index for groupby().shift()
        # This avoids modifying df_processed's index directly within the loop if not careful
        df_temp_momentum = df_processed.set_index('date') 
        for m_period in [1, 3, 6, 12]:
            if m_period == 1:
                 df_processed[f'momentum_{m_period}m'] = df_temp_momentum.groupby('permno')['stock_exret'].shift(1).values
            else:
                 df_processed[f'momentum_{m_period}m'] = df_temp_momentum.groupby('permno')['stock_exret'].shift(1).rolling(window=m_period, min_periods=int(m_period*0.8)).sum().values
            print(f"Calculated: momentum_{m_period}m")
        # No need to reset index on df_processed as df_temp_momentum was a separate object with date index.
    else: 
        print(f"Skipping momentum calculation: stock_exret, permno not found, or date column not datetime ({df_processed['date'].dtype if 'date' in df_processed else 'date col missing'}).")
        for m_period in [1,3,6,12]: df_processed[f'momentum_{m_period}m'] = np.nan; print(f"Placeholder: momentum_{m_period}m")
    
    if 'ni_be' in df_processed.columns: 
        df_processed['quality_roe'] = df_processed['ni_be']; print("Assigned: quality_roe")
    else: df_processed['quality_roe'] = np.nan; print("Placeholder: quality_roe")
    if 'ocf_at' in df_processed.columns: 
        df_processed['quality_cfo_assets'] = df_processed['ocf_at']; print("Assigned: quality_cfo_assets")
    else: df_processed['quality_cfo_assets'] = np.nan; print("Placeholder: quality_cfo_assets")
    
    if pd.api.types.is_datetime64_any_dtype(df_processed['date']):
        df_processed['month_num'] = df_processed['date'].dt.month
        for m_idx in range(2, 13): df_processed[f'month_{m_idx}'] = np.where(df_processed['month_num'] == m_idx, 1, 0)
        df_processed.drop(columns=['month_num'], inplace=True); print("Calculated: seasonality_dummies")
    else:
        print(f"Skipping seasonality_dummies: date column not datetime ({df_processed['date'].dtype if 'date' in df_processed else 'date col missing'}).")
        for m_idx in range(2,13): df_processed[f'month_{m_idx}'] = np.nan # Add placeholders

    return df_processed

# --- Function to Apply Lags ---
def apply_predictor_lags(df, id_col, date_col, predictor_cols, lag_periods=1):
    print(f"\n--- Applying {lag_periods}-month lag to predictors ---")
    if not id_col in df.columns or not date_col in df.columns:
        print(f"Error: ID column '{id_col}' or date column '{date_col}' not found. Skipping lagging.")
        return df
    
    # Crucially, ensure the date_col is datetime for proper sorting and operations
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        print(f"Warning: Date column '{date_col}' for lagging is not datetime. Attempting conversion (assuming YYYYMMDD int or parsable string).")
        try:
            if pd.api.types.is_integer_dtype(df[date_col]):
                 df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
            else:
                 df[date_col] = pd.to_datetime(df[date_col])
            print(f"Lagging date column '{date_col}' converted. Min: {df[date_col].min()}, Max: {df[date_col].max()}")
        except Exception as e:
            print(f"Error converting date column '{date_col}' for lagging: {e}. Lagging may fail or be incorrect.")
            # Potentially return df here if date conversion is critical and fails

    df_lagged = df.copy() # Work on a copy
    df_lagged = df_lagged.sort_values(by=[id_col, date_col]) # Sort before groupby.shift

    actual_predictors = [col for col in predictor_cols if col in df_lagged.columns]
    if not actual_predictors:
        print("Warning: No specified predictor columns found in the DataFrame. Nothing to lag.")
        return df_lagged
        
    print(f"Lagging {len(actual_predictors)} columns: {actual_predictors[:5]}... (and others)") 
    df_lagged[actual_predictors] = df_lagged.groupby(id_col)[actual_predictors].shift(lag_periods)
    print(f"Lagging complete. DataFrame shape after lagging: {df_lagged.shape}")
    return df_lagged

def main():
    print(f"Loading course panel data from {COURSE_PANEL_PATH}...")
    if not os.path.exists(COURSE_PANEL_PATH):
        print(f"Error: Course panel parquet file not found at {COURSE_PANEL_PATH}")
        return
    panel_df = pd.read_parquet(COURSE_PANEL_PATH)
    print("Course panel data loaded successfully.")
    print(f"Initial shape: {panel_df.shape}")
    
    # --- Initial Date Column Conversion (Crucial Fix) ---
    if 'date' in panel_df.columns:
        print(f"Original 'date' column dtype: {panel_df['date'].dtype}")
        try:
            if pd.api.types.is_integer_dtype(panel_df['date']):
                print("Attempting to convert 'date' column from integer (assuming YYYYMMDD).")
                panel_df['date'] = pd.to_datetime(panel_df['date'], format='%Y%m%d')
            elif not pd.api.types.is_datetime64_any_dtype(panel_df['date']):
                print("Attempting to convert 'date' column from object/string.")
                panel_df['date'] = pd.to_datetime(panel_df['date'])
            # If already datetime, no action needed here, it will be confirmed by next print
            print(f"Panel 'date' column after initial conversion: Min: {panel_df['date'].min()}, Max: {panel_df['date'].max()}")
        except Exception as e:
            print(f"CRITICAL Error: Could not convert 'date' column in initial panel_df: {e}. Aborting.")
            return
    else:
        print("CRITICAL Error: 'date' column not found in initial panel_df. Aborting.")
        return
    # --- End Initial Date Column Conversion ---

    print(f"\n--- Processing Official Factor List from {FACTOR_DESCRIPTIONS_PATH} ---")
    if not os.path.exists(FACTOR_DESCRIPTIONS_PATH):
        print(f"Error: Factor descriptions CSV file not found at {FACTOR_DESCRIPTIONS_PATH}. Cannot process official factors.")
        # Initialize an empty list if file not found, so extra features can still run
        official_factor_names = [] 
    else:
        factor_descriptions_df = pd.read_csv(FACTOR_DESCRIPTIONS_PATH)
        official_factor_names = factor_descriptions_df['variable'].tolist()
        print(f"Loaded {len(official_factor_names)} official factor names to check/create.")

        for factor_name in official_factor_names:
            if factor_name in panel_df.columns:
                print(f"Exists: Official factor '{factor_name}' already in DataFrame.")
            else:
                panel_df[factor_name] = np.nan
                print(f"Placeholder: Official factor '{factor_name}' created as NaN. Needs calculation.")
    
    # --- Engineer 'Extra' Features ---
    # This function will add new columns or use existing ones for the extra features.
    # It operates on a copy, so we assign the result back.
    panel_df = engineer_extra_features(panel_df)
    
    # --- Define Predictor Columns for Lagging ---
    # Exclude identifiers, target variables, and other non-predictor metadata
    # Assuming 'stock_exret' and/or 'ret_eom' are targets. 'RF' is also a market variable.
    # 'date' is already converted to datetime in engineer_extra_features
    identifier_cols = ['permno', 'stock_ticker', 'CUSIP', 'comp_name', 'year', 'month'] 
    target_cols = ['stock_exret', 'ret_eom'] # Add others if they are direct future returns
    # Other non-feature cols from original data: SHRCD, EXCHCD, mspread, size_port, RF, date(will be index for lagging)
    
    # All columns that are not identifiers or targets are potential predictors
    potential_predictors = [col for col in panel_df.columns if col not in identifier_cols + target_cols + ['date']]
    # We need to be careful if 'date' column was dropped or if some factors are actually future info.
    # For now, assume all other calculated/existing factors are predictors from period t.

    # --- Apply 1-month lag to predictors ---
    # The `apply_predictor_lags` function expects `date` column to exist for sorting.
    # It also assumes `permno` is the stock identifier.
    # The output of this is the final feature set for ML model training.
    final_ml_features_df = apply_predictor_lags(panel_df, id_col='permno', date_col='date', predictor_cols=potential_predictors, lag_periods=1)

    # Ensure PROCESSED_DIR exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"\nSaving ML-ready features (with lags) to {LAGGED_OUTPUT_FEATURES_PATH}...")
    final_ml_features_df.to_parquet(LAGGED_OUTPUT_FEATURES_PATH, index=False)
    print("ML-ready features saved successfully.")
    print(f"Final ML features shape: {final_ml_features_df.shape}")

    # Check initial Task 4.1 condition (number of columns before lagging)
    # The column count check for Task 4.1 should be on `panel_df` (before lagging removes/adds nothing to total cols but shifts data)
    num_cols_before_lagging = panel_df.shape[1]
    if num_cols_before_lagging >= 160:
        print(f"Pre-lagged feature set has {num_cols_before_lagging} columns (Target: ≥ 160). Task 4.1 condition met.")
    else:
        print(f"Warning: Pre-lagged feature set has {num_cols_before_lagging} columns. Task 4.1 condition NOT YET MET.")

    # Count placeholders for official factors (as before)
    nan_placeholder_factors = 0
    if official_factor_names:
        # Check in the original panel_df, before any lagging might introduce NaNs
        for factor_name in official_factor_names:
            if factor_name in panel_df.columns and panel_df[factor_name].isnull().all():
                # This check is if the factor was a *newly created* placeholder (all NaN)
                # It doesn't distinguish if an *existing* column was all NaN.
                is_placeholder = not (factor_name in panel_df.columns and not panel_df[factor_name].isnull().all())
                if is_placeholder:
                     nan_placeholder_factors +=1
        print(f"Number of official factors that are placeholders (all NaN): {nan_placeholder_factors} out of {len(official_factor_names)}")
    print("Review make_features.py to implement calculations for any remaining placeholder factors.")

if __name__ == "__main__":
    main() 