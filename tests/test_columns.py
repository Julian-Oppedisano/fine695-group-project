import pytest
import pandas as pd
import pyarrow.parquet as pq
import os

# Define the path to the parquet file relative to the project root
# Assuming tests are run from the project root directory
PARQUET_FILE_PATH = os.path.join("data", "raw", "course_panel.parquet")
EXPECTED_TARGET_COLUMN = "stock_exret"
MIN_EXPECTED_PREDICTORS = 147
MIN_TOTAL_COLUMNS = MIN_EXPECTED_PREDICTORS + 1 # Target + Predictors

@pytest.fixture
def parquet_columns():
    """Pytest fixture to read Parquet file schema and return column names."""
    if not os.path.exists(PARQUET_FILE_PATH):
        pytest.fail(f"Parquet file not found at: {PARQUET_FILE_PATH}")
    
    try:
        # Read only the schema to avoid loading the whole file
        schema = pq.read_schema(PARQUET_FILE_PATH)
        columns = schema.names
        return columns
    except Exception as e:
        pytest.fail(f"Error reading Parquet file schema: {e}")

def test_target_column_exists(parquet_columns):
    """Test that the target column 'stock_exret' exists."""
    assert EXPECTED_TARGET_COLUMN in parquet_columns, \
        f"Target column '{EXPECTED_TARGET_COLUMN}' not found in Parquet file."

def test_minimum_number_of_columns(parquet_columns):
    """Test that there is at least the target column and 147 predictor columns."""
    actual_num_columns = len(parquet_columns)
    assert actual_num_columns >= MIN_TOTAL_COLUMNS, \
        f"Expected at least {MIN_TOTAL_COLUMNS} columns (target + {MIN_EXPECTED_PREDICTORS} predictors), \
but found {actual_num_columns}."

# Example of how to run this test:
# Ensure pytest is installed in your conda environment: conda install pytest -c conda-forge
# Navigate to the project root directory in your terminal.
# Run: pytest tests/test_columns.py 