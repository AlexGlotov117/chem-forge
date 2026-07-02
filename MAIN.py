import numpy as np
# import matplotlib.pyplot as plt

from chemicals import Compound, Mixture

# =============================================================================
# 1. DYNAMIC SYSTEM INITIALIZATION (inputs here)
# =============================================================================
# Define everything in one place
compounds_list = [
    Compound(
        name="MEA", 
        mw=61.084, 
        T_fus=283.45, 
        h_fus=13800, 
        h_f_298=-507500,
        formula={"C": 2, "H": 7, "N": 1, "O": 1}
    ),
    Compound(
        name="Urea", 
        mw=60.05534, 
        T_fus=406.15, 
        h_fus=10340, 
        h_f_298=-333333,
        formula={"C": 1, "N": 2, "H": 4, "O": 1}
    )
]


# Build the generic mixture object
system_mixture = Mixture(compounds=compounds_list)

system_mixture.set_composition(x=np.array([0.4, 0.6]), gamma=np.array([1.15, 0.92]))

print(system_mixture.T_fus)           
print(system_mixture.isp)