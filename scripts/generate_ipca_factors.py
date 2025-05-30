import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA # Explicitly import PCA
import joblib
import os

# --- InstrumentedPCA Class Definition (from user) ---
class InstrumentedPCA:
    def __init__(self, n_factors, n_components_pca=None, interactions_only=True):
        self.n_factors = n_factors
        self.n_components_pca = n_components_pca if n_components_pca is not None else n_factors
        self.interactions_only = interactions_only
        self.scaler_X = StandardScaler()
        self.scaler_Z = StandardScaler()
        self.beta_ = None  # Stores the (L, P) matrix of factor loadings on instruments
        self.gamma_ = None # Stores the (P, K) matrix of factor loadings on original char's
        # For Z=None or empty Z case, to store PCA components of X
        self.components_ = None 

    def _instrument_features(self, Z_input):
        if Z_input is None or Z_input.shape[1] == 0:
            return np.ones((Z_input.shape[0], 1))

        # Scaler should be fit on Z_train, then used to transform Z_train, Z_val, Z_test
        # In fit, we fit_transform. In transform, we just transform.
        if hasattr(self.scaler_Z, 'mean_') and self.scaler_Z.mean_ is not None: # Check if scaler is fitted
            Z_scaled = self.scaler_Z.transform(Z_input)
        else:
            Z_scaled = self.scaler_Z.fit_transform(Z_input) # Fit and transform if not fitted (e.g. during fit method)
        
        instrumented_features_list = []
        if not self.interactions_only:
            instrumented_features_list.append(Z_scaled)

        for i in range(Z_scaled.shape[1]):
            for j in range(i, Z_scaled.shape[1]):
                instrumented_features_list.append((Z_scaled[:, i] * Z_scaled[:, j]).reshape(-1, 1))
        
        if not instrumented_features_list:
             return np.ones((Z_input.shape[0], 1))

        F = np.concatenate(instrumented_features_list, axis=1)
        return F

    def fit(self, X, Z, y=None):
        N, P = X.shape
        if Z is None or Z.shape[1] == 0:
            print("Warning: No instruments (Z) provided. IPCA will behave like PCA on X.")
            X_scaled = self.scaler_X.fit_transform(X)
            pca_model = PCA(n_components=self.n_factors, svd_solver='full')
            # self.components_ = pca_model.fit_transform(X_scaled) # These are the factors themselves
            pca_model.fit(X_scaled)
            self.gamma_ = pca_model.components_.T # (P, K) principal components of X
            self.beta_ = np.zeros((1, self.n_factors)) # Placeholder
            # For transform in Z=None case, we essentially project X_scaled onto self.gamma_
            return self

        X_scaled = self.scaler_X.fit_transform(X)
        # Fit scaler_Z with Z from training data and transform Z.
        # The _instrument_features method handles Z scaling internally using self.scaler_Z
        # Call fit_transform on scaler_Z here to ensure it's fitted.
        self.scaler_Z.fit(Z) # Fit the Z scaler explicitly
        F = self._instrument_features(Z) # This will now use the fitted scaler_Z to transform Z then create F
        L = F.shape[1]

        beta_hat = np.zeros((L, P))
        for p_idx in range(P):
            reg = LinearRegression(fit_intercept=False) 
            reg.fit(F, X_scaled[:, p_idx])
            beta_hat[:, p_idx] = reg.coef_

        # --- Corrected Gamma Calculation ---
        # We need Gamma (P,K) from eigenvectors of B' Sigma_Z B (P,P matrix)
        # Assuming Sigma_Z is Identity (due to scaled Z), we need eigenvectors of beta_hat.T @ beta_hat.
        # If SVD of beta_hat (L,P) = U S V.T, then V are the eigenvectors of beta_hat.T @ beta_hat.
        # V has shape (P, P) or (P, rank_of_beta_hat).
        # To get V using sklearn.PCA:
        # Fit PCA on beta_hat (L samples, P features). pca.components_ are (K_pca, P).
        # These components are the K_pca principal axes in the P-dimensional space, which form V.T (rows are eigenvectors).
        # So, pca.components_.T gives (P, K_pca). This is our Gamma.

        # Determine effective number of components for PCA on beta_hat
        # n_features for this PCA is P. n_samples is L.
        effective_n_gamma_components = min(self.n_factors, P, L) # n_factors is K, we want K eigenvectors
        if effective_n_gamma_components != self.n_factors:
            print(f"Warning: n_factors for gamma PCA (effectively K) adjusted from {self.n_factors} to {effective_n_gamma_components} due to data dimensions P={P}, L={L}.")
        if effective_n_gamma_components <= 0:
            raise ValueError(f"Cannot perform PCA for gamma with P={P}, L={L} and effective_n_gamma_components={effective_n_gamma_components}.")

        # Perform PCA on beta_hat (L, P) to get K principal components/axes in P-dim space
        pca_for_gamma = PCA(n_components=effective_n_gamma_components, svd_solver='full')
        pca_for_gamma.fit(beta_hat) # Fit on (L, P) matrix
        # pca_for_gamma.components_ has shape (K, P)
        self.gamma_ = pca_for_gamma.components_.T # Gamma should be (P, K)
        # --- End Corrected Gamma Calculation ---
        
        # Original KPS method for n_components_pca was related to PCA on beta_hat.T for dimensionality reduction of instrument space.
        # The self.n_components_pca parameter seems to be about how many PCs of Beta.T are used to define Gamma.
        # This is a bit different from directly getting Gamma from B'B eigenvectors.
        # For now, the above direct method for Gamma (P,K) is used.
        # The self.n_components_pca from init might need to be re-interpreted or used differently if KPS specific PCA on Beta.T is needed for Gamma.
        # The provided code's PCA was on beta_hat.T which led to (L,K) gamma.

        self.beta_ = beta_hat # Store (L,P) beta_hat for transform.
        return self

    def transform(self, X, Z):
        if self.gamma_ is None:
            raise ValueError("IPCA model has not been fitted yet.")

        X_scaled = self.scaler_X.transform(X)

        if Z is None or Z.shape[1] == 0:
            # This implies it was fitted with Z=None, so self.gamma_ are PCs of X.
            return X_scaled @ self.gamma_

        # Z is present, use the full IPCA transform logic
        # _instrument_features will use the already fitted self.scaler_Z.transform
        F = self._instrument_features(Z) 
        X_pred_instrumented = F @ self.beta_
        factors = X_pred_instrumented @ self.gamma_
        return factors

    def fit_transform(self, X, Z, y=None):
        self.fit(X, Z, y)
        return self.transform(X, Z)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Ensure scaler_Z parameters are captured correctly, even if Z was empty during fit
        scaler_Z_mean = self.scaler_Z.mean_ if hasattr(self.scaler_Z, 'mean_') and self.scaler_Z.mean_ is not None else None
        scaler_Z_scale = self.scaler_Z.scale_ if hasattr(self.scaler_Z, 'scale_') and self.scaler_Z.scale_ is not None else None
        
        joblib.dump({
            'n_factors': self.n_factors,
            'n_components_pca': self.n_components_pca,
            'interactions_only': self.interactions_only,
            'scaler_X_params': (self.scaler_X.mean_, self.scaler_X.scale_),
            'scaler_Z_params': (scaler_Z_mean, scaler_Z_scale),
            'beta_': self.beta_,
            'gamma_': self.gamma_,
            'components_': self.components_ 
        }, path)
        print(f"IPCA model saved to {path}")

    @classmethod
    def load(cls, path):
        params = joblib.load(path)
        model = cls(n_factors=params['n_factors'], 
                    n_components_pca=params['n_components_pca'],
                    interactions_only=params['interactions_only'])
        
        model.scaler_X.mean_ = params['scaler_X_params'][0]
        model.scaler_X.scale_ = params['scaler_X_params'][1]
        
        scaler_Z_mean, scaler_Z_scale = params['scaler_Z_params']
        if scaler_Z_mean is not None and scaler_Z_scale is not None:
            model.scaler_Z.mean_ = scaler_Z_mean
            model.scaler_Z.scale_ = scaler_Z_scale
        else:
            # If Z was None or empty during fit, scaler_Z might not have mean_/scale_
            # Re-initialize a basic one; it won't be used if Z is None/empty at transform.
            model.scaler_Z = StandardScaler()
            
        model.beta_ = params['beta_']
        model.gamma_ = params['gamma_']
        model.components_ = params.get('components_') # Use .get for robustness
        print(f"IPCA model loaded from {path}")
        return model

# --- Configuration for Factor Generation Script ---
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

TRAIN_INPUT_PATH = os.path.join(PROCESSED_DIR, 'train_features_imputed.parquet')
VAL_INPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_features_imputed.parquet')
TEST_INPUT_PATH = os.path.join(PROCESSED_DIR, 'test_features_imputed.parquet')

TRAIN_IPCA_FACTORS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'train_ipca_factors.parquet')
VAL_IPCA_FACTORS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'validation_ipca_factors.parquet')
TEST_IPCA_FACTORS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'test_ipca_factors.parquet')

IPCA_MODEL_SAVE_PATH = os.path.join(SAVED_MODEL_DIR, 'ipca_model.joblib')

# IPCA Parameters
N_IPCA_FACTORS = 32  # Number of latent factors to generate
N_COMPONENTS_PCA_IPCA = 64 # Number of components for PCA step in IPCA
INTERACTIONS_ONLY_IPCA = False # Whether to include instruments themselves or only interactions

TARGET_COLUMN_EXCL = 'stock_exret' 
DATE_COLUMN_EXCL = 'date'
ID_COLUMN_EXCL = 'permno'

# Define a function to get predictor columns for X (characteristics)
def get_X_characteristics_cols(df):
    excluded_cols = [
        TARGET_COLUMN_EXCL, DATE_COLUMN_EXCL, ID_COLUMN_EXCL, 'year', 'month', 'day', 
        'eom_date', 'size_class', 'comb_code', 'month_num',
        'stock_exret_t_plus_1', 'stock_exret_t_plus_2', 
        'stock_exret_t_plus_3', 'stock_exret_t_plus_6', 'stock_exret_t_plus_12',
        'SHRCD', 
        'size_port', 'stock_ticker', 'CUSIP', 'comp_name', # From AE script exclusions
        'target_quintile' # if present from other scripts
    ]
    predictor_cols = [col for col in df.columns if col not in excluded_cols and 
                           not col.startswith('month_') and not col.startswith('eps_')]
    return predictor_cols

# Define the Z instrument columns based on selection
Z_INSTRUMENT_COLS = [
    'market_equity', 'be_me', 'momentum_12m', 'gp_at', 
    'asset_growth', 'debt_me', 'ivol_capm_252d', 
    'turnover_126d', 'accruals', 'age'
]

def main():
    print("--- Starting IPCA Factor Generation ---")
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading data...")
    train_df_full = pd.read_parquet(TRAIN_INPUT_PATH)
    val_df_full = pd.read_parquet(VAL_INPUT_PATH)
    test_df_full = pd.read_parquet(TEST_INPUT_PATH)

    all_X_cols = get_X_characteristics_cols(train_df_full)
    print(f"Identified {len(all_X_cols)} characteristic columns (X) for IPCA.")
    print(f"Using {len(Z_INSTRUMENT_COLS)} instrument columns (Z) for IPCA: {Z_INSTRUMENT_COLS}")

    # Ensure Z_INSTRUMENT_COLS are present in all_X_cols (or at least in train_df_full.columns)
    missing_Z_cols = [col for col in Z_INSTRUMENT_COLS if col not in train_df_full.columns]
    if missing_Z_cols:
        raise ValueError(f"Missing instrument columns in loaded data: {missing_Z_cols}")
    
    # Prepare X and Z matrices
    X_train = train_df_full[all_X_cols].copy()
    Z_train = train_df_full[Z_INSTRUMENT_COLS].copy()
    X_val = val_df_full[all_X_cols].copy()
    Z_val = val_df_full[Z_INSTRUMENT_COLS].copy()
    X_test = test_df_full[all_X_cols].copy()
    Z_test = test_df_full[Z_INSTRUMENT_COLS].copy()

    # Handle infinities and NaNs (important before scaling and modeling)
    print("Handling potential infinities and NaNs in X and Z matrices...")
    for df_part in [X_train, X_val, X_test, Z_train, Z_val, Z_test]:
        # Replace inf with NaN first, then fill all NaNs (original or from inf)
        df_part.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Check if all columns are numeric before filling. If not, this might be an issue.
        for col in df_part.columns:
            if pd.api.types.is_numeric_dtype(df_part[col]):
                df_part[col].fillna(0, inplace=True) # Or use median/mean if appropriate
            else:
                 # This case should ideally not happen if columns are selected correctly
                print(f"Warning: Column {col} is not numeric. Attempting to fill NaNs with 0. Check column selection.")
                try:
                    df_part[col].fillna(0, inplace=True) 
                except TypeError:
                    print(f"Error: Could not fill NaNs in non-numeric column {col} with 0. Please check data.")
                    # Consider filling with a string placeholder like 'MISSING' or raising error
                    df_part[col].fillna('MISSING', inplace=True)

    print("Fitting IPCA model...")
    ipca_model = InstrumentedPCA(
        n_factors=N_IPCA_FACTORS, 
        n_components_pca=N_COMPONENTS_PCA_IPCA, 
        interactions_only=INTERACTIONS_ONLY_IPCA
    )
    ipca_model.fit(X_train.values, Z_train.values) # Pass numpy arrays
    print("IPCA model fitted.")

    ipca_model.save(IPCA_MODEL_SAVE_PATH)

    print("Generating IPCA factors for train, validation, and test sets...")
    train_ipca_factors = ipca_model.transform(X_train.values, Z_train.values)
    val_ipca_factors = ipca_model.transform(X_val.values, Z_val.values)
    test_ipca_factors = ipca_model.transform(X_test.values, Z_test.values)

    # Create column names for IPCA factors
    ipca_factor_cols = [f'ipca_factor_{i}' for i in range(N_IPCA_FACTORS)]

    # Save factors with original identifiers
    print("Saving IPCA factors...")
    train_ipca_factors_df = pd.DataFrame(train_ipca_factors, columns=ipca_factor_cols)
    train_ipca_factors_df[ID_COLUMN_EXCL] = train_df_full[ID_COLUMN_EXCL].values
    train_ipca_factors_df[DATE_COLUMN_EXCL] = train_df_full[DATE_COLUMN_EXCL].values
    train_ipca_factors_df = train_ipca_factors_df[[ID_COLUMN_EXCL, DATE_COLUMN_EXCL] + ipca_factor_cols]
    train_ipca_factors_df.to_parquet(TRAIN_IPCA_FACTORS_OUTPUT_PATH, index=False)
    print(f"Train IPCA factors saved to {TRAIN_IPCA_FACTORS_OUTPUT_PATH}")

    val_ipca_factors_df = pd.DataFrame(val_ipca_factors, columns=ipca_factor_cols)
    val_ipca_factors_df[ID_COLUMN_EXCL] = val_df_full[ID_COLUMN_EXCL].values
    val_ipca_factors_df[DATE_COLUMN_EXCL] = val_df_full[DATE_COLUMN_EXCL].values
    val_ipca_factors_df = val_ipca_factors_df[[ID_COLUMN_EXCL, DATE_COLUMN_EXCL] + ipca_factor_cols]
    val_ipca_factors_df.to_parquet(VAL_IPCA_FACTORS_OUTPUT_PATH, index=False)
    print(f"Validation IPCA factors saved to {VAL_IPCA_FACTORS_OUTPUT_PATH}")

    test_ipca_factors_df = pd.DataFrame(test_ipca_factors, columns=ipca_factor_cols)
    test_ipca_factors_df[ID_COLUMN_EXCL] = test_df_full[ID_COLUMN_EXCL].values
    test_ipca_factors_df[DATE_COLUMN_EXCL] = test_df_full[DATE_COLUMN_EXCL].values
    test_ipca_factors_df = test_ipca_factors_df[[ID_COLUMN_EXCL, DATE_COLUMN_EXCL] + ipca_factor_cols]
    test_ipca_factors_df.to_parquet(TEST_IPCA_FACTORS_OUTPUT_PATH, index=False)
    print(f"Test IPCA factors saved to {TEST_IPCA_FACTORS_OUTPUT_PATH}")

    print("--- IPCA Factor Generation Complete ---")

if __name__ == '__main__':
    np.random.seed(42)
    main() 