import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
PREDICTION_FILE_PATTERN = "pred_*.csv" # Pattern to find prediction files
EXCLUDE_FILES = ["pred_blend.csv", "pred_auto.csv"] # Files to exclude from blending (e.g., previous blends or auto if handled separately)
# If pred_auto.csv should be included in blending, remove it from EXCLUDE_FILES

ID_COLUMNS = ['permno', 'date']
PREDICTION_COLUMN_NAME = 'prediction' # Standard name of the prediction column in each file

OUTPUT_BLEND_FILENAME = "pred_blend.csv"
OUTPUT_CORR_MATRIX_FILENAME = "prediction_correlation_matrix.png"

def load_and_merge_predictions(results_dir, file_pattern, id_cols, pred_col_name, exclude_files):
    """Loads all individual prediction CSVs and merges them."""
    all_pred_files = glob.glob(os.path.join(results_dir, file_pattern))
    
    valid_pred_files = [
        f for f in all_pred_files 
        if os.path.basename(f) not in exclude_files and os.path.basename(f) != OUTPUT_BLEND_FILENAME
    ]

    if not valid_pred_files:
        print(f"No prediction files found matching '{file_pattern}' in '{results_dir}' (excluding {exclude_files}).")
        return None

    print(f"Found prediction files for blending: {valid_pred_files}")
    
    merged_df = None
    for i, file_path in enumerate(valid_pred_files):
        try:
            model_name = os.path.basename(file_path).replace('pred_', '').replace('.csv', '')
            pred_df = pd.read_csv(file_path)
            
            if not all(col in pred_df.columns for col in id_cols + [pred_col_name]):
                print(f"Warning: File {file_path} is missing required columns ({id_cols + [pred_col_name]}). Skipping.")
                continue
            
            # Rename prediction column to be model-specific before merging
            pred_df.rename(columns={pred_col_name: f"pred_{model_name}"}, inplace=True)
            
            if merged_df is None:
                merged_df = pred_df[id_cols + [f"pred_{model_name}"]].copy()
            else:
                # Ensure IDs are of the same type for merging, especially if date is involved
                for col in id_cols:
                    if merged_df[col].dtype != pred_df[col].dtype:
                        try:
                            if 'date' in col.lower(): # A simple heuristic for date columns
                                merged_df[col] = pd.to_datetime(merged_df[col])
                                pred_df[col] = pd.to_datetime(pred_df[col])
                            else: # Try converting to a common numeric type like int64 for IDs
                                merged_df[col] = merged_df[col].astype(np.int64)
                                pred_df[col] = pred_df[col].astype(np.int64)
                        except Exception as e:
                            print(f"Warning: Could not align dtype for column '{col}' between merged data and {file_path}. Error: {e}")

                merged_df = pd.merge(merged_df, pred_df[id_cols + [f"pred_{model_name}"]], on=id_cols, how='outer')
            print(f"Successfully loaded and merged: {file_path}")
        except Exception as e:
            print(f"Error loading or processing file {file_path}: {e}. Skipping.")
            
    if merged_df is None or merged_df.empty:
        print("No data to blend after loading and merging attempts.")
        return None
        
    # Drop rows where all predictions are NaN (can happen with outer merge if IDs don't fully overlap)
    pred_cols = [col for col in merged_df.columns if col.startswith('pred_')]
    if not pred_cols:
        print("Warning: No prediction columns found in the merged DataFrame.")
        return merged_df # Or None, depending on desired handling
        
    merged_df.dropna(subset=pred_cols, how='all', inplace=True)
    return merged_df

def plot_correlation_matrix(df, pred_cols_prefix, output_path):
    """Plots and saves the correlation matrix of predictions."""
    pred_columns = [col for col in df.columns if col.startswith(pred_cols_prefix)]
    if len(pred_columns) < 2:
        print("Correlation matrix requires at least two prediction series. Skipping plot.")
        return

    correlation_matrix = df[pred_columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Model Predictions')
    
    try:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Correlation matrix saved to {output_path}")
    except Exception as e:
        print(f"Error saving correlation matrix plot: {e}")
    plt.close() # Close the plot to free memory

def rank_average_predictions(df, pred_cols_prefix, output_pred_col_name):
    """Performs rank averaging on the prediction columns."""
    pred_columns = [col for col in df.columns if col.startswith(pred_cols_prefix)]
    if not pred_columns:
        print("No prediction columns found for rank averaging.")
        return df

    # Rank each model's predictions. Higher prediction = higher rank (ascending=False)
    # Then average these ranks.
    # NaNs in predictions are handled by rank (they won't contribute to average rank for that row or get a low rank)
    ranks_df = df[pred_columns].rank(axis=1, method='average', na_option='keep', ascending=False)
    df[output_pred_col_name] = ranks_df.mean(axis=1) # Average of ranks
    
    # Optional: Scale the average rank back to a typical prediction range if desired,
    # or normalize. For now, raw average rank is used.
    # Example: (df[output_pred_col_name] - df[output_pred_col_name].min()) / (df[output_pred_col_name].max() - df[output_pred_col_name].min())

    print("Rank averaging complete.")
    return df

def main():
    print("--- Starting Prediction Blending Script ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load and merge predictions
    merged_predictions = load_and_merge_predictions(
        RESULTS_DIR, 
        PREDICTION_FILE_PATTERN, 
        ID_COLUMNS, 
        PREDICTION_COLUMN_NAME,
        EXCLUDE_FILES
    )

    if merged_predictions is None or merged_predictions.empty:
        print("Stopping script as no predictions were loaded or merged.")
        return

    # --- Start Diagnostics for merged_predictions ---
    print("\n--- Diagnostics for merged_predictions ---")
    print("merged_predictions.info():")
    merged_predictions.info()
    
    pred_cols_for_diag = [col for col in merged_predictions.columns if col.startswith('pred_')]
    if pred_cols_for_diag:
        print("\nNaN counts in prediction columns:")
        print(merged_predictions[pred_cols_for_diag].isnull().sum())
        
        print("\nShape of merged_predictions with rows having predictions from ALL models (dropna how='any'):")
        print(merged_predictions.dropna(subset=pred_cols_for_diag, how='any').shape)
        
        print("\nHead of prediction columns:")
        print(merged_predictions[pred_cols_for_diag].head())
    else:
        print("\nNo prediction columns (starting with 'pred_') found for diagnostics.")
    print("--- End Diagnostics ---\n")
    # --- End Diagnostics ---

    # 2. Plot and save correlation matrix
    corr_matrix_path = os.path.join(RESULTS_DIR, OUTPUT_CORR_MATRIX_FILENAME)
    plot_correlation_matrix(merged_predictions, 'pred_', corr_matrix_path)

    # 3. Perform rank averaging
    blended_df = rank_average_predictions(merged_predictions, 'pred_', PREDICTION_COLUMN_NAME)
    
    # 4. Save the blended predictions
    output_blend_path = os.path.join(RESULTS_DIR, OUTPUT_BLEND_FILENAME)
    
    # Ensure ID columns and the new blended prediction column are present
    final_columns_to_save = ID_COLUMNS + [PREDICTION_COLUMN_NAME]
    missing_final_cols = [col for col in final_columns_to_save if col not in blended_df.columns]
    if missing_final_cols:
        print(f"Error: Final blended DataFrame is missing essential columns: {missing_final_cols}. Cannot save.")
        return

    try:
        blended_df[final_columns_to_save].to_csv(output_blend_path, index=False)
        print(f"Blended predictions saved to {output_blend_path}")
    except Exception as e:
        print(f"Error saving blended predictions: {e}")

    print("--- Prediction Blending Script Complete ---")

if __name__ == "__main__":
    # Ensure matplotlib can run in a headless environment if needed
    import matplotlib
    matplotlib.use('Agg') 
    main() 