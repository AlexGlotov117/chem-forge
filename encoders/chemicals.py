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
