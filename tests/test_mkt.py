import pytest
import pandas as pd
import os

# Define the path to the parquet file relative to the project root
PARQUET_FILE_PATH = os.path.join("data", "raw", "mkt.parquet")

@pytest.fixture
def mkt_dataframe():
    """Pytest fixture to read the market data Parquet file."""
    if not os.path.exists(PARQUET_FILE_PATH):
        pytest.fail(f"Market data Parquet file not found at: {PARQUET_FILE_PATH}")
    
    try:
        df = pd.read_parquet(PARQUET_FILE_PATH)
        return df
    except Exception as e:
        pytest.fail(f"Error reading market data Parquet file: {e}")

def test_rf_not_null(mkt_dataframe):
    """Test that the RF (risk-free rate) column has no null values."""
    assert mkt_dataframe['RF'].isnull().sum() == 0, \
        f"RF column contains null values. Null count: {mkt_dataframe['RF'].isnull().sum()}"

def test_sp_ret_is_float(mkt_dataframe):
    """Test that the sp_ret (S&P 500 return) column is of float type."""
    # Need to check an actual pandas/numpy float dtype, not just 'float' string
    # Common float dtypes are numpy.float64, numpy.float32
    assert pd.api.types.is_float_dtype(mkt_dataframe['sp_ret']), \
        f"sp_ret column is not a float type. Actual type: {mkt_dataframe['sp_ret'].dtype}"

# Example of how to run this test:
# Ensure pytest is installed: conda install pytest -c conda-forge
# Navigate to the project root directory.
# Run: pytest tests/test_mkt.py 