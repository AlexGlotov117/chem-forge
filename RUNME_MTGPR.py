# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"
# Use CEARun
from models.gp import MTGPPipeline

import numpy as np
import pandas as pd
import warnings

from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdFingerprintGenerator, Descriptors3D, AllChem

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Suppress convergence warnings for the sake of clean output during small-sample testing
warnings.filterwarnings("ignore")

MONOATOMIC_ION_PROPERTIES = {
    # Cations
    "[Li+]": {
        "MolWt": 6.94,
        "IonRadius": 0.76,
        "Electronegativity": 0.98,
        "ValenceElectrons": 2.0,
    },
    "[Na+]": {
        "MolWt": 22.99,
        "IonRadius": 1.02,
        "Electronegativity": 0.93,
        "ValenceElectrons": 8.0,
    },
    "[K+]": {
        "MolWt": 39.10,
        "IonRadius": 1.38,
        "Electronegativity": 0.82,
        "ValenceElectrons": 8.0,
    },
    # Anions
    "[Cl-]": {
        "MolWt": 35.45,
        "IonRadius": 1.81,
        "Electronegativity": 3.16,
        "ValenceElectrons": 8.0,
    },
    "[Br-]": {
        "MolWt": 79.90,
        "IonRadius": 1.96,
        "Electronegativity": 2.96,
        "ValenceElectrons": 8.0,
    },
    "[I-]": {
        "MolWt": 126.90,
        "IonRadius": 2.20,
        "Electronegativity": 2.66,
        "ValenceElectrons": 8.0,
    },
}

def get_rdkit_descriptors_with_names(mol, prefix=""):
    """Calculates RDKit 2D descriptors and returns a dictionary with prefixed

    names.
    """
    desc_dict = {}
    is_none = mol is None

    for name, func in Descriptors._descList:
        col_name = f"{prefix}_{name}" if prefix else name
        if is_none:
            desc_dict[col_name] = 0.0
        else:
            try:
                val = func(mol)
                desc_dict[col_name] = val if np.isfinite(val) else 0.0
            except Exception:
                desc_dict[col_name] = 0.0
    return desc_dict

def get_morgan_fingerprint_dict(mol, radius=2, n_bits=32, prefix=""):
    """Generates a low-dimensional Morgan Fingerprint bit vector dictionary."""
    fp_dict = {}
    if mol is None:
        for i in range(n_bits):
            fp_dict[f"{prefix}_MorganBit_{i}"] = 0.0
        return fp_dict

    # Compute ECFP4-like bit vector
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    fp = fpgen.GetFingerprint(mol)
    for i in range(n_bits):
        fp_dict[f"{prefix}_MorganBit_{i}"] = float(fp[i])

    return fp_dict

def generate_3d_mol(mol):
    """Generates a fast 3D ETKDG conformer and optimizes it with MMFF94."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Skip single atomic ions (e.g., Li+, Na+, Cl-)—they have no 3D shape
    if mol.GetNumAtoms() == 1:
        return None

    m3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    # Embed conformer
    res = AllChem.EmbedMolecule(m3d, params)
    if res != 0:
        # Fallback to random coordinates if ETKDG fails
        AllChem.EmbedMolecule(m3d, useRandomCoords=True)

    try:
        AllChem.MMFFOptimizeMolecule(m3d, maxIters=200)
    except Exception:
        pass  # If forcefield fails, proceed with embedded coordinates

    return m3d

def get_3d_descriptors_with_names(mol, prefix=""):
    """Extracts fast 3D shape, volume, radius of gyration, and principal moments."""
    keys = [
        "Asphericity",
        "Eccentricity",
        "InertialShapeFactor",
        "NPR1",
        "NPR2",
        "PMI1",
        "PMI2",
        "PMI3",
        "RadiusOfGyration",
        "SpherocityIndex",
    ]

    d_3d = {}
    m3d = generate_3d_mol(mol) if mol else None

    if m3d is None or m3d.GetNumConformers() == 0:
        for k in keys:
            d_3d[f"{prefix}_3D_{k}"] = 0.0
        d_3d[f"{prefix}_3D_VanDerWaalsVolume"] = 0.0
        return d_3d

    # Calculate RDKit 3D descriptors
    d_3d[f"{prefix}_3D_Asphericity"] = Descriptors3D.Asphericity(m3d)
    d_3d[f"{prefix}_3D_Eccentricity"] = Descriptors3D.Eccentricity(m3d)
    d_3d[f"{prefix}_3D_InertialShapeFactor"] = Descriptors3D.InertialShapeFactor(
        m3d
    )
    d_3d[f"{prefix}_3D_NPR1"] = Descriptors3D.NPR1(m3d)
    d_3d[f"{prefix}_3D_NPR2"] = Descriptors3D.NPR2(m3d)
    d_3d[f"{prefix}_3D_PMI1"] = Descriptors3D.PMI1(m3d)
    d_3d[f"{prefix}_3D_PMI2"] = Descriptors3D.PMI2(m3d)
    d_3d[f"{prefix}_3D_PMI3"] = Descriptors3D.PMI3(m3d)
    d_3d[f"{prefix}_3D_RadiusOfGyration"] = Descriptors3D.RadiusOfGyration(m3d)
    d_3d[f"{prefix}_3D_SpherocityIndex"] = Descriptors3D.SpherocityIndex(m3d)

    # Fast van der Waals volume approximation
    d_3d[f"{prefix}_3D_VanDerWaalsVolume"] = (
        AllChem.ComputeMolVolume(m3d) if m3d else 0.0
    )

    return d_3d

def featurize_single_compound_to_dict(smiles, fp_bits=32):
    """Featurizes SMILES into a composite dictionary containing 2D, 3D,

    fingerprints, and bulk thermodynamic descriptors.
    """
    frags = smiles.split(".")
    mols = [
        Chem.MolFromSmiles(f) for f in frags if Chem.MolFromSmiles(f) is not None
    ]

    cation_mol, anion_mol = None, None
    for m in mols:
        charge = Chem.GetFormalCharge(m)
        if charge > 0:
            cation_mol = m
        elif charge < 0:
            anion_mol = m

    is_ionic = 1.0 if (cation_mol is not None or anion_mol is not None) else 0.0

    # =========================================================================
    # 1. Fragment Descriptors (2D + 3D Shape & Volume)
    # =========================================================================
    if is_ionic == 1.0:
        d_cat_2d = get_rdkit_descriptors_with_names(cation_mol, prefix="Cation")
        d_an_2d = get_rdkit_descriptors_with_names(anion_mol, prefix="Anion")
        d_neu_2d = get_rdkit_descriptors_with_names(None, prefix="Neutral")

        d_cat_3d = get_3d_descriptors_with_names(cation_mol, prefix="Cation")
        d_an_3d = get_3d_descriptors_with_names(anion_mol, prefix="Anion")
        d_neu_3d = get_3d_descriptors_with_names(None, prefix="Neutral")

        mw_cat = Descriptors.MolWt(cation_mol) if cation_mol else 0.0
        mw_an = Descriptors.MolWt(anion_mol) if anion_mol else 0.0
        tpsa_cat = Descriptors.TPSA(cation_mol) if cation_mol else 0.0
        tpsa_an = Descriptors.TPSA(anion_mol) if anion_mol else 0.0
        rot_cat = (
            Descriptors.NumRotatableBonds(cation_mol) if cation_mol else 0
        )
        rot_an = Descriptors.NumRotatableBonds(anion_mol) if anion_mol else 0

        # Substructure Fingerprints
        fp_cat = get_morgan_fingerprint_dict(
            cation_mol, n_bits=fp_bits, prefix="Cation"
        )
        fp_an = get_morgan_fingerprint_dict(
            anion_mol, n_bits=fp_bits, prefix="Anion"
        )
        fp_neu = get_morgan_fingerprint_dict(
            None, n_bits=fp_bits, prefix="Neutral"
        )

    else:
        parent_mol = mols[0] if len(mols) > 0 else None
        d_cat_2d = get_rdkit_descriptors_with_names(None, prefix="Cation")
        d_an_2d = get_rdkit_descriptors_with_names(None, prefix="Anion")
        d_neu_2d = get_rdkit_descriptors_with_names(
            parent_mol, prefix="Neutral"
        )

        d_cat_3d = get_3d_descriptors_with_names(None, prefix="Cation")
        d_an_3d = get_3d_descriptors_with_names(None, prefix="Anion")
        d_neu_3d = get_3d_descriptors_with_names(parent_mol, prefix="Neutral")

        mw_cat, mw_an, tpsa_cat, tpsa_an = 0.0, 0.0, 0.0, 0.0
        rot_cat, rot_an = 0, 0

        fp_cat = get_morgan_fingerprint_dict(
            None, n_bits=fp_bits, prefix="Cation"
        )
        fp_an = get_morgan_fingerprint_dict(
            None, n_bits=fp_bits, prefix="Anion"
        )
        fp_neu = get_morgan_fingerprint_dict(
            parent_mol, n_bits=fp_bits, prefix="Neutral"
        )

    # =========================================================================
    # 2. Bulk & Physical Lattice Assembly Descriptors
    # =========================================================================
    vol_cat = d_cat_3d.get("Cation_3D_VanDerWaalsVolume", 0.0)
    vol_an = d_an_3d.get("Anion_3D_VanDerWaalsVolume", 0.0)

    # Bulk density proxies & Kapustinskii-like ionic packing terms
    effective_vol = (
        (vol_cat + vol_an)
        if is_ionic
        else d_neu_3d.get("Neutral_3D_VanDerWaalsVolume", 0.0)
    )
    total_mw = (
        (mw_cat + mw_an)
        if is_ionic
        else (Descriptors.MolWt(mols[0]) if mols else 0.0)
    )

    d_assembly = {
        "Assembly_is_ionic": is_ionic,
        "Assembly_total_mw": total_mw,
        "Assembly_mw_ratio": mw_cat / (mw_an + 1e-5),
        "Assembly_tpsa_ratio": tpsa_cat / (tpsa_an + 1e-5),
        "Assembly_volume_ratio": vol_cat / (vol_an + 1e-5),
        "Assembly_total_rotatable_bonds": float(rot_cat + rot_an),
        "Assembly_packing_density_proxy": total_mw / (effective_vol + 1e-5),
        "Assembly_electrostatic_charge_density": (
            (1.0 / (effective_vol + 1e-5)) if is_ionic else 0.0
        ),
    }

    # Combine everything
    return {
        **d_cat_2d,
        **d_an_2d,
        **d_neu_2d,
        **d_cat_3d,
        **d_an_3d,
        **d_neu_3d,
        **d_assembly,
        **fp_cat,
        **fp_an,
        **fp_neu,
    }


def build_composite_dataframe(smiles_list):
    """Builds a fully labeled pandas DataFrame directly from SMILES."""
    rows = [featurize_single_compound_to_dict(s) for s in smiles_list]
    return pd.DataFrame(rows)

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
    feature_names = desc_names + frag_names + fp_names

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
        features_list.append(descs + frags + fp_names)

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
        df = pd.read_excel(filepath, na_values=["—", "-", "N/A"])
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
            
    # Drop rows missing  SMILES strings
    df = df.dropna(subset=[smiles_column])
    
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
    
    Y_scaled = np.full_like(Y_train, fill_value=np.nan)
    for t_idx in range(Y_train.shape[1]):
        col_data = Y_train[:, t_idx]
        valid_mask = ~np.isnan(col_data)
        if np.sum(valid_mask) > 1:
            mean = np.mean(col_data[valid_mask])
            std = np.std(col_data[valid_mask])
            std = 1.0 if std == 0 else std
            Y_scaled[valid_mask, t_idx] = (col_data[valid_mask] - mean) / std
    
    # Track feature selection counts across targets and LOOCV folds
    feature_scores = {feat: 0.0 for feat in feature_names}
    task_feature_scores = {task: {feat: 0 for feat in feature_names} for task in target_names}
    
    # 2. Iterate over each target task (MTGP multivariable driver selection)
    for task_idx, task_name in enumerate(target_names):
        y_task = Y_scaled[:, task_idx]

        valid_indices = np.where(~np.isnan(y_task))[0]
        N_valid = len(valid_indices)

        if N_valid < 5:
            print(
                f"Skipping Task '{task_name}': Insufficient observed samples ({N_valid})"
            )
            continue

        X_task_valid = X_scaled[valid_indices]
        y_task_valid = y_task[valid_indices]
        
        # Leave-One-Out Cross-Validation Loop
        for i in range(N_valid):
            # Split LOOCV
            mask = np.ones(N_valid, dtype=bool)
            mask[i] = False
            
            X_tr, y_tr = X_task_valid[mask], y_task_valid[mask]
            
            cv_folds = min(5, N_valid - 2)
            if cv_folds < 2:
                continue

            lasso = LassoCV(
                cv=cv_folds, max_iter=10000, random_state=42, tol=1e-3
            )
            try:
                lasso.fit(X_tr, y_tr)
            except Exception:
                continue
            
            # Identify non-zero coefficients
            non_zero_indices = np.where(np.abs(lasso.coef_) > 1e-5)[0]
            
            for idx in non_zero_indices:
                feat = feature_names[idx]
                feature_scores[feat] += np.abs(lasso.coef_[idx])
                task_feature_scores[task_name][feat] += 1

    # 3. Sort features by aggregate stability & weight score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    selected_feature_names = []

    # Sort each task's candidate features by its OWN score for that specific task
    task_rankings = {
        task: [
            f[0]
            for f in sorted(
                task_feature_scores[task].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        for task in target_names
    }

    # Interleave selection across tasks to hit exact max_features count
    rank_idx = 0
    while len(selected_feature_names) < max_features and rank_idx < P:
        for task in target_names:
            feat = task_rankings[task][rank_idx]
            if (
                feat not in selected_feature_names
                and len(selected_feature_names) < max_features
            ):
                selected_feature_names.append(feat)
        rank_idx += 1

    # Print out selected features alongside their global cumulative score
    print(
        f"\nTop {len(selected_feature_names)} Selected Physical Descriptors"
        f" (Target: {max_features}):"
    )
    for rank, feat in enumerate(selected_feature_names, 1):
        score = feature_scores[feat]
        print(f"  {rank}. {feat:<35} (Cumulative Score: {score:.3f})")
    print("==========================================================\n")
    
    # 4. Slice Tier 1 DataFrame down to Tier 2
    X_tier2_df = X_train_filtered_df[selected_feature_names].copy()
    
    # 5. Diagnostic Visualization
    if show_plots:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data for top 12 candidate features comparison
        top_candidates = [feat for feat, _ in sorted_features[:min(P,40)]]
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
    smiles_train, Y_train_raw = extract_smiles_and_targets(TRAIN_FILE_PATH)
    smiles_test, Y_test_raw = extract_smiles_and_targets(TEST_FILE_PATH)

    print(Y_train_raw)

    # ==============================================================================
    # 1. FEATURIZATION & COMPOSITE ASSEMBLY (X_sample)
    # ==============================================================================

    # Build raw composite feature matrices
    df_X_train = build_composite_dataframe(smiles_train)
    df_X_test = build_composite_dataframe(smiles_test)

    # Extract feature names (List of strings for plots/tables)
    feature_names = df_X_train.columns.tolist()

    # Extract numerical values (2D NumPy array for MTGPR)
    X_train_raw = df_X_train.values  # or df_X_train_selected.to_numpy()

    # Do the exact same for your test set using the selected feature names
    X_test_raw = df_X_test[feature_names].values

    # ==============================================================================
    # 2. TARGET TRANSFORMATION (T_m -> dS_fus)
    # Assumes Y target columns are ordered as: [T_m (K), dH_fus (kJ/mol), dH_f (kJ/mol)]
    # ==============================================================================


    def transform_targets_to_dS(Y_raw):
        Y_trans = Y_raw.copy()
        # dS_fus = dH_fus / T_m
        Y_trans[:, 0] = Y_raw[:, 1] / Y_raw[:, 0]
        return Y_trans


    Y_train_dS = transform_targets_to_dS(Y_train_raw)

    X_train_tier1 = apply_tier1_unsupervised_filter(X_train_raw, feature_names, show_plots=False)
    
    X_train_tier2, final_features = apply_tier2_supervised_filter(
        X_train_tier1, 
        Y_train_dS, 
        max_features=np.floor(X_train_tier1.shape[0] / 4.0), 
        show_plots=False
    )

    # Convert test data to DataFrame and slice using the SAME retained feature list
    X_test_filtered = df_X_test[final_features]

    print(
        f"Train Matrix Shape: {X_train_tier2.shape}"
    )  # (N_train, 49)
    print(f"Test Matrix Shape:  {X_test_filtered.shape}")  # (N_test, 49)

    # ==============================================================================
    # 5. MULTI-TASK GAUSSIAN PROCESS FIT & PREDICTION
    # ==============================================================================

    pipeline = MTGPPipeline(num_tasks=3, lr=0.01, num_epochs=1000)
    pipeline.fit(X_train_tier2.values, Y_train_dS)

    # Train MTGPR on the selected 8 features and dS_fus targets
    # gp_model = MTGPR_gpy()
    # gp_model.fit(X_train_tier2, Y_train_dS)

    
    print("\n=== Evaluating Candidate Precursors ===")
    means, stds = pipeline.predict(X_test_filtered.values)
    
    dS_pred, dS_std = means[:, 0], stds[:, 0]
    dH_pred, dH_std = means[:, 1], stds[:, 1]
    dH_f_pred, dH_f_std = means[:, 2], stds[:, 2]

    # Calculate derived T_m
    T_m_pred = dH_pred / dS_pred

    # Propagate uncertainty to T_m using Root-Sum-Square (RSS)
    T_m_std = np.sqrt(
        (dH_std / dS_pred)**2 + 
        ((dH_pred * dS_std) / (dS_pred**2))**2
    )

    T_m_true = Y_test_raw[:, 0] 
    dH_fus_true = Y_test_raw[:, 1] 
    dH_f_true = Y_test_raw[:, 2]

    # Output Results
    for i in range(len(X_test_filtered.values)):
        print(f"Candidate {i+1}: {smiles_test[i]}")
        print(f"  T_m    : {T_m_pred[i]:.2f} ± {T_m_std[i]:.2f} K (Actual: {T_m_true[i]} K)")
        print(f"  dH_fus : {dH_pred[i]:.2f} ± {dH_std[i]:.2f} kJ/mol (Actual: {dH_fus_true[i]} kJ/mol)")
        print(f"  dH_f   : {dH_f_pred[i]:.2f} ± {dH_f_std[i]:.2f} kJ/mol (Actual: {dH_f_true[i]} kJ/mol)\n")

    # Y_train[:,0] = Y_train[:,1] / Y_train[:,0]
    # Y_test[:,0] = Y_test[:,1] / Y_test[:,0]
    
    # print(f"Featurizing {len(smiles_train)} SMILES strings via RDKit...")
    # X_train, X_train_names = featurize_smiles(smiles_train)
    # X_test, X_test_names = featurize_smiles(smiles_test)

    # X_train_tier1 = apply_tier1_unsupervised_filter(X_train, X_train_names, show_plots=False)

    # X_train_tier2, final_features = apply_tier2_supervised_filter(
    #     X_train_tier1, 
    #     Y_train, 
    #     max_features=10, 
    #     show_plots=True
    # )

    # # Convert test data to DataFrame and slice using the SAME retained feature list
    # X_test_df = pd.DataFrame(X_test, columns=X_test_names)
    # X_test_filtered = X_test_df[final_features]

    # print(
    #     f"Train Matrix Shape: {X_train_tier2.shape}"
    # )  # (N_train, 49)
    # print(f"Test Matrix Shape:  {X_test_filtered.shape}")  # (N_test, 49)

    # gp_framework = MTGPR()
    # gp_framework.fit(X_train_tier2.values, Y_train)
    
    # print("\n=== Evaluating Candidate Precursors ===")
    # predictions = gp_framework.predict(X_test_filtered.values)
    
    # # Print results formatted nicely
    # for i in range(len(X_test_filtered.values)):
    #     print(f"\nCandidate {i+1}: {smiles_test[i]}")
    #     for prop in ['T_m', 'dH_fus', 'dH_f']:
    #         if prop == 'T_m':
    #             pred = predictions['dH_fus']['prediction'][i]/predictions[prop]['prediction'][i]
    #             uncert = predictions[prop]['uncertainty'][i]
    #             flag = predictions[prop]['high_risk_flag'][i]
    #         else:
    #             pred = predictions[prop]['prediction'][i]
    #             uncert = predictions[prop]['uncertainty'][i]
    #             flag = predictions[prop]['high_risk_flag'][i]
            
    #         flag_str = "[WARNING: EXTRAPOLATION]" if flag else "[RELIABLE]"
    #         print(f"  {prop:6s}: {pred:7.2f} ± {uncert:6.2f} {flag_str}")

  