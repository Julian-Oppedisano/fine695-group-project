import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq

def convert_csv_to_parquet(csv_file_path, parquet_file_path, chunk_size=100000):
    """
    Converts a large CSV file to Parquet format using chunked reading.

    Args:
        csv_file_path (str): Path to the input CSV file.
        parquet_file_path (str): Path to the output Parquet file.
        chunk_size (int): Number of rows per chunk.
    """
    print(f"Starting conversion of {csv_file_path} to {parquet_file_path}...")
    total_rows = 0
    writer = None

    # Ensure the directory for the parquet file exists
    os.makedirs(os.path.dirname(parquet_file_path), exist_ok=True)

    for i, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunk_size)):
        total_rows += len(chunk)
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            # For the first chunk, create the writer and write with schema
            writer = pq.ParquetWriter(parquet_file_path, table.schema)
            print(f"Processed first chunk of {len(chunk)} rows. Schema written.")
        
        writer.write_table(table)
        print(f"Processed chunk {i+1} of {len(chunk)} rows. Total rows: {total_rows}")

    if writer:
        writer.close()
        print("Parquet writer closed.")

    print(f"Conversion complete. Total rows processed: {total_rows}")
    print(f"Parquet file saved to: {parquet_file_path}")
    return total_rows

if __name__ == "__main__":
    # Assuming the script is in fine695-group-project/scripts/
    # and the data is in fine695-group-project/data/raw/
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'goup_project_sample_v3.csv')
    parquet_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'course_panel.parquet')

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
    else:
        # Delete existing parquet file if it exists, to prevent schema conflicts on re-run
        if os.path.exists(parquet_path):
            print(f"Deleting existing parquet file: {parquet_path}")
            os.remove(parquet_path)

        row_count = convert_csv_to_parquet(csv_path, parquet_path)
        # Log the row count to a file or print it as needed
        with open("data/raw/course_panel_row_count.txt", "w") as f:
            f.write(str(row_count))
        print(f"Total row count ({row_count}) saved to data/raw/course_panel_row_count.txt")

        # Verify parquet file size
        if os.path.exists(parquet_path):
            parquet_size_bytes = os.path.getsize(parquet_path)
            parquet_size_mb = parquet_size_bytes / (1024 * 1024)
            print(f"Parquet file size: {parquet_size_mb:.2f} MB")
            if parquet_size_mb < 300:
                print("Parquet file size is less than 300 MB. Task 2.1 condition met.")
            else:
                print("Warning: Parquet file size is NOT less than 300 MB.")
        else:
            print(f"Error: Parquet file not created at {parquet_path}") 