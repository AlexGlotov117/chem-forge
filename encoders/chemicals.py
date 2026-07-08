import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

def basic_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [Descriptors.ExactMolWt(mol), Descriptors.NumRotatableBonds(mol)]


def standard_molecular_features(target_smiles):
    """
    Converts a SMILES string into a high-dimensional structural fingerprint.
    Replaces simple metrics with a 2048-bit structural topology vector.
    """

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
