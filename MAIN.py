import numpy as np
import matplotlib.pyplot as plt

from solvers import SLESolver
from chemicals import Compound, Mixture
from dataProcessing import exportSLE

# =============================================================================
# 1. DYNAMIC SYSTEM INITIALIZATION (inputs here)
# =============================================================================
# Define everything in one place
compounds_list = [
    Compound(
        name="MEA", 
        mw=61.084, 
        t_fus=283.45, 
        h_fus=15000, 
        h_f_298=-507.5,
        formula={"C": 2, "H": 7, "N": 1, "O": 1}
    ),
    Compound(
        name="Urea", 
        mw=60.05534, 
        t_fus=407.2, 
        h_fus=14600, 
        h_f_298=-333.33,
        formula={"C": 1, "N": 2, "H": 4, "O": 1}
    )
]


# Build the generic mixture object
system_mixture = Mixture(compounds=compounds_list)

# Query a specific state point directly (e.g., 60% MEA / 40% Urea)
state_point = system_mixture.evaluate_at(np.array([0.60, 0.40]))

print(f"Melting Point at this blend: {state_point['T_sle']:.1f} K")
print(f"Specific Impulse at this blend: {state_point['isp']:.1f} seconds")