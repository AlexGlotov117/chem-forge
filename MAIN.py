import numpy as np
import matplotlib.pyplot as plt

from solvers import SLESolver
from chemicals import Compound, Mixture
from dataProcessing import exportSLE

# =============================================================================
# 1. DYNAMIC SYSTEM INITIALIZATION (inputs here)
# =============================================================================
compounds_list = [
    Compound(name="Cyclohexane", mw=84.16, t_fus=279.84, h_fus=2628),
    Compound(name="Urea", mw=60.056, t_fus=407.2, h_fus=14600.0)
    # You can add compounds here
]

# Build the generic mixture object
system_mixture = Mixture(compounds=compounds_list)
N = system_mixture.num_components

# Unpack the generic matrix outputs
x_matrix, T_matrix, T_SLE = SLESolver().solve(system_mixture, steps=101)

# =============================================================================
# 2. CSV EXPORT
# =============================================================================
exportSLE("binary_phase_data.csv", system_mixture, x_matrix, T_matrix, T_SLE)