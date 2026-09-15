# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"
# Use CEARun
from models.gp import MTGPR

import numpy as np
import pandas as pd
import warnings

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import Descriptors, Fragments, rdFingerprintGenerator

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Suppress convergence warnings for the sake of clean output during small-sample testing
warnings.filterwarnings("ignore")

RDKIT_AVAILABLE = True

def featurize_smiles(smiles_list, fp_size=512, radius=2):
    """
    Converts a list or array of SMILES strings into a Morgan fingerprint bit vector matrix 
    and returns corresponding feature names.
    
    Parameters:
    - smiles_list: List or array of SMILES strings.
    - fp_size: Number of bits for the fingerprint (default: 512).
    - radius: Radius of the Morgan fingerprint (default: 2 -> ECFP4 equivalent).
    
    Returns:
    - X: np.ndarray of shape (N_samples, fp_size) containing 0/1 bits as floats.
    - feature_names: List of strings ['Morgan_Bit_0', 'Morgan_Bit_1', ...]
    """
        
    features_list = []

    # Initialize fingerprint generator
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fp_size
    )

    # Setup feature names once
    desc_names = [f"Desc_{name}" for name, _ in Descriptors.descList]
    frag_names = [name for name, func in Fragments.__dict__.items() if name.startswith("fr_") and callable(func)]
    fp_names = [f"Morgan_Bit_{i}" for i in range(fp_size)]
    feature_names = desc_names + frag_names

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi).strip())

        if mol is None:
            warnings.warn(
                f"Invalid SMILES string encountered: '{smi}'. Outputting zero"
                " vector."
            )
            features_list.append(None)
            continue

        # Extract values for descriptors, fragments, and fingerprints
        descs = [float(func(mol)) for _, func in Descriptors.descList]
        frags = [float(func(mol)) for name, func in Fragments.__dict__.items()if name.startswith("fr_") and callable(func)]
        fp_bits = list(fpgen.GetFingerprint(mol))

        # Append all numeric values for this molecule
        features_list.append(descs + frags)

    # Handle missing/invalid entries
    n_features = len(feature_names)
    clean_features = [
        f if f is not None else [0.0] * n_features for f in features_list
    ]

    return np.array(clean_features, dtype=float), feature_names

def extract_smiles_and_targets(filepath, target_columns=None, smiles_column='SMILES'):
    """
    Extracts raw SMILES strings and property target values (Y) from an Excel file.
    
    Parameters:
    - filepath: Path to the Excel (.xlsx) file.
    - target_columns: List of target column names in the Excel file.
    - smiles_column: Name of the column containing SMILES strings.
    
    Returns:
    - smiles_data: 1D numpy array of SMILES strings.
    - Y: 2D numpy array of target property values.
    """
    if target_columns is None:
        target_columns = ['T_m', 'dH_fus', 'dH_f']
        
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the Excel file at '{filepath}'.")
    
    # Ensure target columns exist in the DataFrame
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in Excel file: {missing_targets}")
    
    # Find SMILES column (case-insensitive check if exact match missing)
    if smiles_column not in df.columns:
        matched_cols = [c for c in df.columns if str(c).strip().upper() == smiles_column.upper()]
        if matched_cols:
            smiles_column = matched_cols[0]
        else:
            raise ValueError(f"Could not find SMILES column '{smiles_column}' in Excel file.")
            
    # Drop rows missing target values or SMILES strings
    df = df.dropna(subset=target_columns + [smiles_column])
    
    smiles_data = df[smiles_column].astype(str).values
    Y = df[target_columns].values

    return smiles_data, Y

def apply_tier1_unsupervised_filter(
    X_matrix, 
    feature_names, 
    variance_thresh=0.01, 
    corr_thresh=0.90, 
    nan_strategy='drop_col',
    show_plots=True
):
    """
    Tier 1 Unsupervised Feature Filter with Visual Diagnostics.
    
    Parameters:
    -----------
    X_matrix : np.ndarray or pd.DataFrame
        The raw feature matrix (N_samples x P_descriptors)
    feature_names : list of str
        List of descriptor names corresponding to columns of X_matrix
    variance_thresh : float
        Drop features with variance below this value (default 0.01)
    corr_thresh : float
        Drop one feature from any pair with absolute Pearson correlation |r| > corr_thresh
    show_plots : bool
        If True, displays diagnostic bar chart and correlation heatmap
        
    Returns:
    --------
    X_filtered_df : pd.DataFrame
        The filtered feature matrix as a DataFrame with preserved column names
    """
    # 1. Convert input to pandas DataFrame
    if not isinstance(X_matrix, pd.DataFrame):
        df_raw = pd.DataFrame(X_matrix, columns=feature_names)
    else:
        df_raw = X_matrix.copy()
        
    n_initial = df_raw.shape[1]
    print(f"\n================ TIER 1 FEATURE FILTER ================")
    print(f"Initial Feature Count: {n_initial}")

    # -------------------------------------------------------------
    # Step 0: Handle NaN Values
    # -------------------------------------------------------------
    nan_cols = df_raw.columns[df_raw.isna().any() | np.isinf(df_raw).any()].tolist()
    if len(nan_cols) > 0:
        print(f"\n[Step 0] NaN Detection:")
        print(f"    - Found {len(nan_cols)} descriptors containing NaN values.")
        print(f"      Examples with NaNs: {nan_cols[:5]}")
        
        if nan_strategy == 'drop_col':
            # Drop any column that has even a single NaN value
            df_clean = df_raw.drop(columns=nan_cols)
            print(f"    - Action: Dropped all {len(nan_cols)} NaN-containing columns.")
        elif nan_strategy == 'impute_median':
            # Fill NaNs with column median (or 0.0 if entire column is NaN)
            df_clean = df_raw.fillna(df_raw.median()).fillna(0.0)
            print(f"    - Action: Imputed NaNs using column median values.")
    else:
        df_clean = df_raw.copy()
        print(f"\n[Step 0] NaN Detection: Clean! No NaN values found.")

    df_raw = df_clean
    # 2. Variance Thresholding (Remove near-constant features)
    vt = VarianceThreshold(threshold=variance_thresh)
    vt.fit(df_raw)
    
    retained_var_cols = df_raw.columns[vt.get_support()]
    dropped_var_cols = df_raw.columns[~vt.get_support()]
    
    df_var = df_raw[retained_var_cols].copy()
    n_after_var = df_var.shape[1]
    
    print(f"\n[Step 1] Low Variance Filter (Threshold <= {variance_thresh}):")
    print(f"    - Removed {len(dropped_var_cols)} / {n_initial} features")
    if len(dropped_var_cols) > 0:
        sample_dropped = list(dropped_var_cols[:5])
        print(f"    - Examples dropped: {sample_dropped}" + ("..." if len(dropped_var_cols) > 5 else ""))
        
    # 3. Collinearity Pruning (Remove high Pearson correlation |r| > corr_thresh)
    corr_matrix = df_var.corr().abs()
    
    # Upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify columns to drop due to high correlation with an earlier column
    cols_to_drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] > corr_thresh)]
    
    print(f"\n[Step 2] Collinearity Filter (|r| > {corr_thresh}):")
    for col in cols_to_drop_corr:
        correlated_with = upper_tri.index[upper_tri[col] > corr_thresh].tolist()
        print(f"    - Dropping '{col}' (Correlated with: {correlated_with})")
        
    df_filtered = df_var.drop(columns=cols_to_drop_corr).copy()
    n_final = df_filtered.shape[1]
    
    print(f"\n---> Retained Candidate Pool: {n_final} features (Reduced from {n_initial})")
    print(f"=======================================================\n")
    
    # 4. Visualization Plots
    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot A: Feature Retention Breakdown
        counts = {
            'Raw Input': n_initial,
            'Post Variance': n_after_var,
            'Post Collinearity': n_final
        }
        palette = sns.color_palette("Blues_r", n_colors=3)
        bars = axes[0].bar(counts.keys(), counts.values(), color=palette, edgecolor='black')
        axes[0].set_title("Tier 1 Feature Reduction", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Number of Descriptors")
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels above bars
        for bar in bars:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + (0.02 * n_initial), 
                         int(yval), ha='center', va='bottom', fontweight='bold')
            
        # Plot B: Correlation Heatmap of Filtered Features
        if n_final > 1:
            sns.heatmap(df_filtered.corr(), cmap="coolwarm", vmin=-1, vmax=1, 
                        ax=axes[1], cbar=True, annot=False, square=True)
            axes[1].set_title(f"Correlation Matrix of {n_final} Retained Features", 
                              fontsize=12, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, "Too few features for heatmap", ha='center', va='center')
            
        plt.tight_layout()
        plt.show()
        
    return df_filtered

def apply_tier2_supervised_filter(
    X_train_filtered_df, 
    Y_train, 
    target_names=['T_m', 'dH_fus', 'dH_f'],
    max_features=6, 
    show_plots=True
):
    """
    Tier 2 Supervised Feature Selection using LOOCV-LASSO Stability Scoring.
    
    Parameters:
    -----------
    X_train_filtered_df : pd.DataFrame
        DataFrame of features retained after Tier 1 (N_samples x P_tier1)
    Y_train : np.ndarray
        Target array of shape (N_samples, 3)
    target_names : list of str
        Labels for the target properties
    max_features : int
        Target number of physical features for GP (default: 6, recommended: 5-8)
    show_plots : bool
        If True, displays selection frequency bar charts across target tasks
        
    Returns:
    --------
    X_tier2_df : pd.DataFrame
        The final DataFrame sliced down to the top max_features physical drivers
    selected_feature_names : list of str
        Names of the retained physical descriptors
    """
    N, P = X_train_filtered_df.shape
    feature_names = X_train_filtered_df.columns.tolist()
    
    print(f"\n================ TIER 2 SUPERVISED FILTER ================")
    print(f"Input Candidate Pool: {P} features | Target Matrix: {Y_train.shape}")
    print(f"Selection Strategy: LOOCV-LASSO Stability (Target Core Size: {max_features})")
    
    # 1. Standardize Features and Targets for LASSO regularization equality
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train_filtered_df.values)
    
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_train)
    
    # Track feature selection counts across targets and LOOCV folds
    feature_scores = {feat: 0.0 for feat in feature_names}
    task_feature_scores = {task: {feat: 0 for feat in feature_names} for task in target_names}
    
    # 2. Iterate over each target task (MTGP multivariable driver selection)
    for task_idx, task_name in enumerate(target_names):
        y_task = Y_scaled[:, task_idx]
        
        # Leave-One-Out Cross-Validation Loop
        for i in range(N):
            # Split LOOCV
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            
            X_tr, y_tr = X_scaled[mask], y_task[mask]
            
            # Fit LassoCV with automatic alpha search
            lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
            lasso.fit(X_tr, y_tr)
            
            # Identify non-zero coefficients
            non_zero_indices = np.where(np.abs(lasso.coef_) > 1e-5)[0]
            
            for idx in non_zero_indices:
                feat = feature_names[idx]
                feature_scores[feat] += np.abs(lasso.coef_[idx])
                task_feature_scores[task_name][feat] += 1

    # 3. Sort features by aggregate stability & weight score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    selected_feature_names = [feat for feat, score in sorted_features[:max_features]]
    
    print(f"\nTop {max_features} Selected Physical Descriptors:")
    for rank, (feat, score) in enumerate(sorted_features[:max_features], 1):
        print(f"  {rank}. {feat:<30} (Cumulative Score: {score:.3f})")
    print(f"==========================================================\n")
    
    # 4. Slice Tier 1 DataFrame down to Tier 2
    X_tier2_df = X_train_filtered_df[selected_feature_names].copy()
    
    # 5. Diagnostic Visualization
    if show_plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Prepare data for top 12 candidate features comparison
        top_candidates = [feat for feat, _ in sorted_features[:min(12, P)]]
        df_plot = pd.DataFrame([
            {
                'Feature': feat,
                'Task': task,
                'Selection_Count': task_feature_scores[task][feat]
            }
            for feat in top_candidates
            for task in target_names
        ])
        
        sns.barplot(
            data=df_plot, 
            x='Selection_Count', 
            y='Feature', 
            hue='Task', 
            ax=ax, 
            palette='viridis'
        )
        
        # Draw threshold line for selected features
        ax.axhline(y=max_features - 0.5, color='red', linestyle='--', label=f'Cutoff (Top {max_features})')
        ax.set_title(f"Tier 2 Feature Selection Stability Across Tasks (LOOCV-LASSO)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Selection Frequency Across LOOCV Folds")
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        
    return X_tier2_df, selected_feature_names

if __name__ == "__main__":
    print("=== Precursor GP Pipeline ===")

    TRAIN_FILE_PATH = "MTGPR_train.xlsx"
    TEST_FILE_PATH = "MTGPR_test.xlsx"
    
    # X_train, Y_train = generate_mock_data(n_samples=40, n_features=1500)
    smiles_train, Y_train = extract_smiles_and_targets(TRAIN_FILE_PATH)
    smiles_test, Y_test = extract_smiles_and_targets(TEST_FILE_PATH)
    Y_train[:,0] = Y_train[:,1] / Y_train[:,0]
    Y_test[:,0] = Y_test[:,1] / Y_test[:,0]
    
    print(f"Featurizing {len(smiles_train)} SMILES strings via RDKit...")
    X_train, X_train_names = featurize_smiles(smiles_train)
    X_test, X_test_names = featurize_smiles(smiles_test)

    X_train_tier1 = apply_tier1_unsupervised_filter(X_train, X_train_names, show_plots=False)

    X_train_tier2, final_features = apply_tier2_supervised_filter(
        X_train_tier1, 
        Y_train, 
        max_features=6, 
        show_plots=True
    )

    # Convert test data to DataFrame and slice using the SAME retained feature list
    X_test_df = pd.DataFrame(X_test, columns=X_test_names)
    X_test_filtered = X_test_df[final_features]

    print(
        f"Train Matrix Shape: {X_train_tier2.shape}"
    )  # (N_train, 49)
    print(f"Test Matrix Shape:  {X_test_filtered.shape}")  # (N_test, 49)

    gp_framework = MTGPR()
    gp_framework.fit(X_train_tier2.values, Y_train)
    
    print("\n=== Evaluating Candidate Precursors ===")
    predictions = gp_framework.predict(X_test_filtered.values)
    
    # Print results formatted nicely
    for i in range(len(X_test_filtered.values)):
        print(f"\nCandidate {i+1}: {smiles_test[i]}")
        for prop in ['T_m', 'dH_fus', 'dH_f']:
            if prop == 'T_m':
                pred = predictions['dH_fus']['prediction'][i]/predictions[prop]['prediction'][i]
                uncert = predictions[prop]['uncertainty'][i]
                flag = predictions[prop]['high_risk_flag'][i]
            else:
                pred = predictions[prop]['prediction'][i]
                uncert = predictions[prop]['uncertainty'][i]
                flag = predictions[prop]['high_risk_flag'][i]
            
            flag_str = "[WARNING: EXTRAPOLATION]" if flag else "[RELIABLE]"
            print(f"  {prop:6s}: {pred:7.2f} ± {uncert:6.2f} {flag_str}")

  