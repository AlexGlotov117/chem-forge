# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"

from models.gp import MTGPR

import numpy as np
import warnings

# Suppress convergence warnings for the sake of clean output during small-sample testing
warnings.filterwarnings("ignore")

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

if __name__ == "__main__":
    print("=== Precursor GP Pipeline ===")
    
    X_train, Y_train = generate_mock_data(n_samples=40, n_features=1500)
    
    gp_framework = MTGPR(variance_retained=0.95, max_components=5)
    gp_framework.fit(X_train, Y_train)
    
    X_test, _ = generate_mock_data(n_samples=5, n_features=1500)
    X_test[-1, :] += 10.0 
    
    print("\n=== Evaluating Candidate Precursors ===")
    predictions = gp_framework.predict(X_test)
    
    # Print results formatted nicely
    for i in range(len(X_test)):
        print(f"\nCandidate {i+1}:")
        for prop in ['T_m', 'dH_fus', 'dH_f']:
            pred = predictions[prop]['prediction'][i]
            uncert = predictions[prop]['uncertainty'][i]
            flag = predictions[prop]['high_risk_flag'][i]
            
            flag_str = "[WARNING: EXTRAPOLATION]" if flag else "[RELIABLE]"
            print(f"  {prop:6s}: {pred:7.2f} ± {uncert:6.2f} {flag_str}")