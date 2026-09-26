import numpy as np

def basic_features(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    mol = Chem.MolFromSmiles(smiles)
    return [Descriptors.ExactMolWt(mol), Descriptors.NumRotatableBonds(mol)]


def standard_molecular_features(target_smiles):
    """
    Converts a SMILES string into a high-dimensional structural fingerprint.
    Replaces simple metrics with a 2048-bit structural topology vector.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        # Fallback to an empty vector if SMILES parsing fails
        return np.zeros(2048, dtype=float)
        
    # Generate a radius-2 Morgan Fingerprint (equivalent to ECFP4)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    
    # Convert the internal RDKit bit vector into a standard NumPy array for our distance matrix
    features = np.zeros((1,), dtype=float)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, features)
    
    return features

def hybrid_physicochemical_encoder(smiles_str):
    """
    Combines 1024-bit Morgan Fingerprints with scaled physical descriptors
    (MW, TPSA, HBD, HBA, Rotatable Bonds, Formal Charge).
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, Descriptors, rdMolDescriptors

    MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return np.zeros(1024 + 7)

    # 1. Structural Fingerprint (1024 bits for tighter feature space)
    fp = MORGAN_GEN.GetFingerprint(mol)
    fp_arr = np.zeros((1024,), dtype=float)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)

    # 2. Key Thermodynamic Physical Descriptors
    phys_descriptors = np.array([
        Descriptors.ExactMolWt(mol) / 500.0,            # Scaled Molecular Weight
        Descriptors.TPSA(mol) / 200.0,                   # Topological Polar Surface Area
        rdMolDescriptors.CalcNumHBD(mol) / 10.0,         # H-Bond Donors
        rdMolDescriptors.CalcNumHBA(mol) / 10.0,         # H-Bond Acceptors
        rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0, # Flexibility
        Chem.GetFormalCharge(mol) / 2.0,                 # Charge state
        Descriptors.HeavyAtomCount(mol) / 50.0           # Heavy atom count
    ], dtype=float)

    # Concatenate bit vector + continuous physical vector
    return np.concatenate([fp_arr, phys_descriptors])

## NEWER VERSION OF FEATURIZATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdFingerprintGenerator, Descriptors3D, AllChem
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from thermo.group_contribution.joback import Joback
from sklearn.feature_selection import mutual_info_regression

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

    # relevant_keywords = ['fr_']

    # enthalpy_descriptors = [
    #     (name, func) for name, func in Descriptors._descList 
    #     if any(key in name for key in relevant_keywords)
    # ]

    # print(f"Selected:\n {Descriptors._descList}")
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

def get_element_counts(mol, prefix):
    if mol is None:
        return {
            f"{prefix}_Count_C": 0, f"{prefix}_Count_N": 0, 
            f"{prefix}_Count_O": 0, f"{prefix}_Count_B": 0, 
            f"{prefix}_Count_F": 0, f"{prefix}_Count_Cl": 0
        }
    
    atoms = mol.GetAtoms()
    return {
        f"{prefix}_Count_C": sum(1 for a in atoms if a.GetAtomicNum() == 6),
        f"{prefix}_Count_N": sum(1 for a in atoms if a.GetAtomicNum() == 7),
        f"{prefix}_Count_O": sum(1 for a in atoms if a.GetAtomicNum() == 8),
        f"{prefix}_Count_B": sum(1 for a in atoms if a.GetAtomicNum() == 5),
        f"{prefix}_Count_F": sum(1 for a in atoms if a.GetAtomicNum() == 9),
        f"{prefix}_Count_Cl": sum(1 for a in atoms if a.GetAtomicNum() == 17),
    }

def get_bond_counts(mol, prefix):
    """Dynamically counts exact bond types (e.g., C-H, C=C) in a molecule."""
    if mol is None:
        return {}
    
    # CRITICAL: RDKit hides H's by default. We must add them to count C-H, N-H, etc.
    try:
        mol_h = Chem.AddHs(mol)
    except Exception:
        mol_h = mol # Fallback if AddHs fails due to sanitization issues
        
    bond_counts = {}
    
    for bond in mol_h.GetBonds():
        a1 = bond.GetBeginAtom().GetSymbol()
        a2 = bond.GetEndAtom().GetSymbol()
        btype = str(bond.GetBondType())  # Outputs 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'
        
        # Sort atomic symbols alphabetically so C-H and H-C are grouped together as H-C
        elements = sorted([a1, a2])
        bond_key = f"{prefix}_Bond_{elements[0]}_{btype}_{elements[1]}"
        
        bond_counts[bond_key] = bond_counts.get(bond_key, 0) + 1
        
    return bond_counts

def featurize_single_compound_to_dict(smiles, fp_bits=16):
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
        if fp_bits > 0:
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
            fp_cat = None
            fp_an = None
            fp_neu = None

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

        if fp_bits > 0:
            fp_cat = get_morgan_fingerprint_dict(
                None, n_bits=fp_bits, prefix="Cation"
            )
            fp_an = get_morgan_fingerprint_dict(
                None, n_bits=fp_bits, prefix="Anion"
            )
            fp_neu = get_morgan_fingerprint_dict(
                parent_mol, n_bits=fp_bits, prefix="Neutral"
            )
        else:
            fp_cat = None
            fp_an = None
            fp_neu = None

    # =========================================================================
    # 2. Bulk & Physical Lattice Assembly Descriptors
    # =========================================================================
    vol_cat = d_cat_3d.get("Cation_3D_VanDerWaalsVolume", 0.0)
    vol_an = d_an_3d.get("Anion_3D_VanDerWaalsVolume", 0.0)

    effective_vol = (vol_cat + vol_an) if is_ionic else d_neu_3d.get("Neutral_3D_VanDerWaalsVolume", 0.0)
    total_mw = (mw_cat + mw_an) if is_ionic else (Descriptors.MolWt(mols[0]) if mols else 0.0)

    # NEW: Get actual formal charges for lattice energy calculation
    q_cat = abs(Chem.GetFormalCharge(cation_mol)) if cation_mol else 0.0
    q_an = abs(Chem.GetFormalCharge(anion_mol)) if anion_mol else 0.0

    # NEW: True Kapustinskii Lattice Energy Proxy ( U ~ |q+ * q-| / (V^(1/3)) )
    # Volume is proportional to r^3, so V^(1/3) gives the inter-ionic distance r
    lattice_energy_proxy = (q_cat * q_an) / (effective_vol**(1/3) + 1e-5) if is_ionic else 0.0

    d_assembly = {
        "Assembly_is_ionic": is_ionic,
        "Assembly_total_mw": total_mw,
        "Assembly_mw_ratio": mw_cat / (mw_an + 1e-5),
        "Assembly_tpsa_ratio": tpsa_cat / (tpsa_an + 1e-5),
        "Assembly_volume_ratio": vol_cat / (vol_an + 1e-5),
        "Assembly_total_rotatable_bonds": float(rot_cat + rot_an),
        "Assembly_packing_density_proxy": total_mw / (effective_vol + 1e-5),
        "Assembly_electrostatic_charge_density": lattice_energy_proxy,
    }

    # =========================================================================
    # NEW: Element Counts for Heat of Formation Additivity
    # =========================================================================
    if is_ionic == 1.0:
        elem_cat = get_element_counts(cation_mol, prefix="Cation")
        elem_an = get_element_counts(anion_mol, prefix="Anion")
        elem_neu = get_element_counts(None, prefix="Neutral")
    else:
        parent_mol = mols[0] if len(mols) > 0 else None
        elem_cat = get_element_counts(None, prefix="Cation")
        elem_an = get_element_counts(None, prefix="Anion")
        elem_neu = get_element_counts(parent_mol, prefix="Neutral")

    # =========================================================================
    # NEW: Exact Bond Type Counts for Heat of Formation
    # =========================================================================
    if is_ionic == 1.0:
        bonds_cat = get_bond_counts(cation_mol, prefix="Cation")
        bonds_an = get_bond_counts(anion_mol, prefix="Anion")
        bonds_neu = get_bond_counts(None, prefix="Neutral")
    else:
        parent_mol = mols[0] if len(mols) > 0 else None
        bonds_cat = get_bond_counts(None, prefix="Cation")
        bonds_an = get_bond_counts(None, prefix="Anion")
        bonds_neu = get_bond_counts(parent_mol, prefix="Neutral")

    # Combine everything
    return {
        **d_cat_2d, **d_an_2d, #**d_neu_2d,
        **d_cat_3d, **d_an_3d, #**d_neu_3d,
        **elem_cat, **elem_an, #**elem_neu,
        # **bonds_cat, **bonds_an, **bonds_neu,
        **d_assembly,
        # **fp_cat, **fp_an, **fp_neu,
    }


def build_composite_dataframe(smiles_list):
    """Builds a fully labeled pandas DataFrame directly from SMILES."""
    rows = [featurize_single_compound_to_dict(s) for s in smiles_list]
    return pd.DataFrame(rows)

class MolecularEncoder:
    """
    Handles feature generation, filtering, target transformation,
    and Joback physical prior calculations directly from SMILES.
    """
    def __init__(self, output_dir, variance_thresh=0.05, corr_thresh=0.98, override_features=None):
        self.variance_thresh = variance_thresh
        self.corr_thresh = corr_thresh
        self.override_features = override_features or [
            # "Cation_qed",
            # # "Neutral_FractionCSP3",
            # "Assembly_mw_ratio",
            # # "Neutral_Kappa1",
            # # "Anion_Chi0v"
            # # "Anion_HallKierAlpha",
            # # "Anion_MinPartialCharge",
            # # "Cation_MaxPartialCharge",
            # # "Anion_BCUT2D_MWLOW",
            # # "Neutral_BCUT2D_MWHI",
            # # "Neutral_Ipc",
            # # "Neutral_3D_NPR1",
            # # "Assembly_electrostatic_charge_density",
            # # "Neutral_HeavyAtomCount",
            # # "Neutral_ExactMolWt",
            # # "Assembly_packing_density_proxy"
            # # "Anion_LabuteASA",
            # "Cation_NumRotatableBonds",
            # "Cation_MolWt",
            # # "Anion_MolWt",
            # # "Cation_Kappa1",
            # # "Anion_Count_B",
            # "Cation_Count_C",
            # "Cation_Count_N",
            # "Assembly_electrostatic_charge_density",
            # "Cation_3D_Eccentricity",

            # "Neutral_ExactMolWt",
            # "Neutral_HeavyAtomCount",
            # "Assembly_electrostatic_charge_density",
            # "Assembly_packing_density_proxy",
            # # Neutral Shape (For non-salts)
            # "Cation_Kappa1", "Cation_Kappa2", "Cation_Kappa3",
            # # # Anion Shape (For salts)
            # "Anion_Kappa1", "Anion_Kappa2", "Anion_Kappa3",

            # "Cation_fr_NH0", "Neutral_fr_NH0", "Anion_fr_NH0"
            # "Cation_fr_halogen", "Neutral_fr_halogen", "Anion_fr_halogen"
            # "Cation_fr_quatN", "Neutral_fr_quatN", "Anion_fr_quatN"
            # "Cation_fr_unbrch_alkane", "Neutral_fr_unbrch_alkane", "Anion_fr_unbrch_alkane"
            # "Cation_fr_alkyl_halide", "Neutral_fr_alkyl_halide", "Anion_fr_alkyl_halide"
        ]
        self.selected_features = None

    def featurize(self, smiles_list):
        """Converts SMILES strings to raw feature DataFrame."""
        rows = [featurize_single_compound_to_dict(s) for s in smiles_list]
        return pd.DataFrame(rows)

    def compute_joback_priors(self, smiles_list, num_tasks=3):
        """
        Computes physical Joback priors [T_m, dH_fus, dH_f] directly from SMILES.
        Returns array of shape (N, 3).
        """
        n = len(smiles_list)
        preds = np.full((n, num_tasks), np.nan)
        
        for idx, smi in enumerate(smiles_list):
            try:
                obj = Joback(smi)
                t_m = obj.Tm(obj.counts) if hasattr(obj, 'Tm') else np.nan
                dh_fus = (obj.Hfus(obj.counts) / 1000.0) if hasattr(obj, 'Hfus') else np.nan
                dh_f = (obj.Hf(obj.counts) / 1000.0) if hasattr(obj, 'Hf') else np.nan
                preds[idx] = [t_m, dh_fus, dh_f]
            except Exception:
                pass # Unrecognized groups stay as NaN
                
        return preds

    def fit_transform_features(self, smiles_train, Y_train, target_names=['T_m', 'dH_fus', 'dH_f'], max_features=8, override_features=None, show_plots=False):
        df_raw = self.featurize(smiles_train)
        n_initial = df_raw.shape[1]

        # Merge instance overrides with call-time overrides
        overrides = list(set(self.override_features + (override_features or [])))
        
        # Verify override features actually exist in raw featurized data
        valid_overrides = [f for f in overrides if f in df_raw.columns]
        invalid_overrides = set(overrides) - set(valid_overrides)
        if invalid_overrides:
            print(f"[Warning] The following override features were not found in generated features: {invalid_overrides}")

        print(f"\n================ TIER 1 FEATURE FILTER ================")
        print(f"Initial Feature Count: {n_initial}")
        if valid_overrides:
            print(f"Override/Bypass Features Registered ({len(valid_overrides)}): {valid_overrides}")

        # Step 0: Clean NaNs
        nan_cols = df_raw.columns[df_raw.isna().any() | np.isinf(df_raw).any()].tolist()
        nan_cols_to_drop = [c for c in nan_cols if c not in valid_overrides]
        df_clean = df_raw.drop(columns=nan_cols_to_drop) if len(nan_cols_to_drop) > 0 else df_raw.copy()

        # Step 1: Variance Filter
        vt = VarianceThreshold(threshold=self.variance_thresh)
        vt.fit(df_clean)
        retained_var = set(df_clean.columns[vt.get_support()])
        retained_var.update(valid_overrides) # Explicitly keep overrides
        
        # FIX: Sort the list to lock the column order alphabetically and ensure determinism
        sorted_retained = sorted(list(retained_var))
        df_var = df_clean[sorted_retained].copy()
        n_after_var = df_var.shape[1]

        # Step 2: Collinearity Filter
        corr_matrix = df_var.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        cols_to_drop_corr = [
            col for col in upper_tri.columns 
            if any(upper_tri[col] > self.corr_thresh) and col not in valid_overrides
        ]

        df_t1 = df_var.drop(columns=cols_to_drop_corr).copy()
        n_t1 = df_t1.shape[1]
        print(f"---> Retained Tier 1 Pool: {n_t1} features")

        # Plot Tier 1
        if show_plots:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            counts = {'Raw Input': n_initial, 'Post Variance': n_after_var, 'Post Collinearity': n_t1}
            bars = axes[0].bar(counts.keys(), counts.values(), color=sns.color_palette("Blues_r", 3), edgecolor='black')
            axes[0].set_title("Tier 1 Feature Reduction", fontweight='bold')
            for bar in bars:
                axes[0].text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), int(bar.get_height()), ha='center', va='bottom')

            if n_t1 > 1:
                sns.heatmap(df_t1.corr(), cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1], square=True)
                axes[1].set_title(f"Correlation Matrix ({n_t1} Features)", fontweight='bold')
            plt.tight_layout()
            plt.show()

        # Tier 2 Supervised Filter
        print(f"\n================ TIER 2 SUPERVISED FILTER ================")
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(df_t1.values)
        feature_names = df_t1.columns.tolist()

        # Scale Targets Per Task 
        Y_scaled = np.full_like(Y_train, fill_value=np.nan)
        for t_idx in range(Y_train.shape[1]):
            col_data = Y_train[:, t_idx]
            valid_mask = ~np.isnan(col_data)
            if np.sum(valid_mask) > 1:
                mean = np.mean(col_data[valid_mask])
                std = np.std(col_data[valid_mask])
                std = 1.0 if std == 0 else std
                Y_scaled[valid_mask, t_idx] = (col_data[valid_mask] - mean) / std

        # Run Mutual Information Regression
        feature_scores = {f: 0.0 for f in feature_names}
        task_feature_scores = {
            task: {f: 0.0 for f in feature_names} for task in target_names
        }

        for task_idx, task_name in enumerate(target_names):
            y_task = Y_scaled[:, task_idx]
            valid_idx = np.where(~np.isnan(y_task))[0]
            if len(valid_idx) < 5:
                continue

            X_valid, y_valid = X_scaled[valid_idx], y_task[valid_idx]

            # Compute non-linear Mutual Information in one vectorized pass
            mi_scores = mutual_info_regression(
                X_valid, y_valid, n_neighbors=3, random_state=42
            )

            for idx, score in enumerate(mi_scores):
                if score > 1e-4:  # Threshold for non-zero mutual info
                    feat = feature_names[idx]
                    feature_scores[feat] += score
                    # Store exact MI score per task
                    task_feature_scores[task_name][feat] += score

        # Sort ONLY non-override features by MI score
        sorted_features = sorted(
            [f for f in feature_scores.items() if f[0] not in valid_overrides], 
            key=lambda x: x[1], 
            reverse=True
        )

        # Grab top-ranked features
        top_ranked = [f[0] for f in sorted_features[:(max_features - len(valid_overrides))]]
        
        # Combine overrides + top MI features dynamically
        final_selected = list(valid_overrides)
        for feat in top_ranked:
            if feat not in final_selected:
                final_selected.append(feat)

        self.selected_features = final_selected

        print(f"\nFinal Selected Features ({len(self.selected_features)} Total):")
        for rank, feat in enumerate(self.selected_features, 1):
            is_override = " [OVERRIDE]" if feat in valid_overrides else ""
            print(f"  {rank}. {feat:<35} (Score: {feature_scores.get(feat, 0.0):.3f}){is_override}")

        # Plot Tier 2 (Updated for Mutual Information)
        if show_plots:
            fig, ax = plt.subplots(figsize=(10, max(6, len(self.selected_features) * 0.35)))
            
            # Prepare DataFrame using the top 30 features (or available features)
            n_display = min(len(feature_names), 30)
            top_display_feats = [f[0] for f in sorted_features[:n_display]]
            
            # Include valid overrides in plot if present
            display_feats = list(dict.fromkeys(list(valid_overrides) + top_display_feats))

            df_plot = pd.DataFrame([
                {
                    'Feature': feat + (" *" if feat in self.selected_features else ""),
                    'Task': task, 
                    'MI_Score': task_feature_scores[task][feat]
                }
                for feat in display_feats
                for task in target_names
            ])
            
            sns.barplot(
                data=df_plot, 
                x='MI_Score', 
                y='Feature', 
                hue='Task', 
                ax=ax, 
                palette='viridis'
            )
            
            # Draw cutoff horizontal line if cutoff point falls within displayed features
            cutoff_index = max_features - len(valid_overrides)
            if cutoff_index < len(display_feats):
                ax.axhline(
                    y=cutoff_index - 0.5, 
                    color='red', 
                    linestyle='--', 
                    linewidth=1.5,
                    label=f'Selection Cutoff (Top {max_features})'
                )

            ax.set_title("Tier 2 Mutual Information Feature Importance", fontweight='bold', fontsize=12)
            ax.set_xlabel("Mutual Information Score", fontweight='bold')
            ax.set_ylabel("Feature (* = Selected)", fontweight='bold')
            ax.legend(loc='lower right')
            plt.tight_layout()
            plt.show()

        # RETURN DATAFRAME TO PRESERVE FEATURE NAMES
        return df_t1[self.selected_features]

    def transform_features(self, smiles_test):
        """Transforms test SMILES using retained feature names."""
        df_X = self.featurize(smiles_test)
        return df_X[self.selected_features]
