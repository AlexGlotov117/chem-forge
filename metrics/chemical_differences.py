import numpy as np

def tanimoto_distance(x1, x2):
    """
    Computes Tanimoto Distance between two 1D binary feature vectors.
    Distance = 1.0 - (Intersection / Union)
    """
    intersection = np.logical_and(x1, x2).sum()
    union = np.logical_or(x1, x2).sum()
    
    if union == 0:
        return 1.0
        
    similarity = intersection / union
    return 1.0 - similarity