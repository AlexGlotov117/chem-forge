# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"
# Use CEARun
from models.gp import MTGPR

import numpy as np
import pandas as pd
import warnings

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# Suppress convergence warnings for the sake of clean output during small-sample testing
warnings.filterwarnings("ignore")

RDKIT_AVAILABLE = True

def generate_mock_data(n_samples=40, n_features=1500):
    """Generates synthetic high-dimensional descriptors and physical targets."""
    np.random.seed(42)
    
    # Simulate standardized molecular descriptors (e.g., Morgan fingerprints, topological indices)
    X = np.random.randn(n_samples, n_features)
    
    # Simulate target properties (with some underlying hidden linear/non-linear relationships)
    # T_m (K): Typically 250 - 450 K
    T_m = 300 + 50 * np.sin(X[:, 0]) + 20 * X[:, 10] + np.random.randn(n_samples) * 5
    T_m = np.clip(T_m, 200, 600) # Ensure physical bounds
    
    # dH_fus (kJ/mol): Typically 5 - 40 kJ/mol
    dH_fus = 15 + 5 * X[:, 5] + 2 * X[:, 100]**2 + np.random.randn(n_samples) * 2
    dH_fus = np.clip(dH_fus, 1, 60)
    
    # dH_f (kJ/mol): Can be negative or positive (e.g., -500 to +500)
    dH_f = -200 + 150 * X[:, 1] - 50 * X[:, 2] + np.random.randn(n_samples) * 20
    
    Y = np.column_stack((T_m, dH_fus, dH_f))
    return X, Y

def featurize_smiles(smiles_list, feature_type='descriptors'):
    """
    Converts a list or array of SMILES strings into a numerical feature matrix X using RDKit.
    
    Parameters:
    - smiles_list: List or array of SMILES strings.
    - feature_type: 
        * 'descriptors': Physical & 2D chemical descriptors (MW, LogP, TPSA, HBD/HBA, rings, etc.)
        * 'fingerprints': Morgan Fingerprints / ECFP4 bit vectors (1024-bit).
        * 'combined': Concatenates physical descriptors and Morgan fingerprints.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required to featurize SMILES strings directly. "
            "Please install it using 'pip install rdkit'."
        )
        
    features_list = []
    calc_descriptors = Descriptors.descList
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            warnings.warn(f"Invalid SMILES string encountered: '{smi}'. Outputting zero vector.")
            features_list.append(None)
            continue
            
        mol_features = []
        
        # 1. 2D RDKit Descriptors (Physical/Chemical properties)
        if feature_type in ['descriptors', 'combined']:
            desc_vals = [func(mol) for _, func in calc_descriptors]
            # Clean up NaN/Inf entries if any descriptor calculation fails
            desc_vals = [0.0 if (np.isnan(v) or np.isinf(v)) else float(v) for v in desc_vals]
            mol_features.extend(desc_vals)
            
        # 2. Morgan Fingerprints (ECFP4, radius 2)
        if feature_type in ['fingerprints', 'combined']:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fp_bits = [int(b) for b in fp.ToBitString()]
            mol_features.extend(fp_bits)
            
        features_list.append(mol_features)
        
    # Handle dimension sizing if invalid SMILES were skipped
    valid_len = next((len(f) for f in features_list if f is not None), None)
    if valid_len is None:
        raise ValueError("None of the provided SMILES strings could be parsed into valid RDKit molecules.")
        
    clean_features = [f if f is not None else [0.0] * valid_len for f in features_list]
    return np.array(clean_features, dtype=float)

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


if __name__ == "__main__":
    print("=== Precursor GP Pipeline ===")

    TRAIN_FILE_PATH = "MTGPR_train.xlsx"
    TEST_FILE_PATH = "MTGPR_test.xlsx"
    
    # X_train, Y_train = generate_mock_data(n_samples=40, n_features=1500)
    smiles_train, Y_train = extract_smiles_and_targets(TRAIN_FILE_PATH)
    smiles_test, Y_test = extract_smiles_and_targets(TEST_FILE_PATH)
        
    print(f"Featurizing {len(smiles_train)} SMILES strings via RDKit...")
    X_train = featurize_smiles(smiles_train, feature_type='descriptors')
    X_test = featurize_smiles(smiles_test, feature_type='descriptors')
    
    print(f"Successfully loaded and featurized data: {X_train.shape[0]} samples, {X_train.shape[1]} features extracted.")
    
    gp_framework = MTGPR(variance_retained=0.95, max_components=5)
    gp_framework.fit(X_train, Y_train)
    
    print("\n=== Evaluating Candidate Precursors ===")
    predictions = gp_framework.predict(X_test)
    
    # Print results formatted nicely
    for i in range(len(X_test)):
        print(f"\nCandidate {i+1}: {smiles_test[i]}")
        for prop in ['T_m', 'dH_fus', 'dH_f']:
            pred = predictions[prop]['prediction'][i]
            uncert = predictions[prop]['uncertainty'][i]
            flag = predictions[prop]['high_risk_flag'][i]
            
            flag_str = "[WARNING: EXTRAPOLATION]" if flag else "[RELIABLE]"
            print(f"  {prop:6s}: {pred:7.2f} ± {uncert:6.2f} {flag_str}")