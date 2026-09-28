import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar

# 1. Pure Component Melting Properties (Adjust T_fus and H_fus for your compounds)
R = 8.314  # Gas constant, J/(mol*K)

# Pure component 1 (tetramethylammonium chloride)
T_fus_1 = 639.63      # Melting temperature in Kelvin (example value)
H_fus_1 = 20490     # Enthalpy of fusion in J/mol (example value)

# Pure component 2 (tetrabutylammonium chloride)
T_fus_2 = 328.49       # Melting temperature in Kelvin (example value)
H_fus_2 = 19966     # Enthalpy of fusion in J/mol (example value)


# 2. NRTL Activity Coefficient Model
def nrtl_gamma(x1, x2, T, a12, a21, alpha=0.3):
    tau12 = a12 / T
    tau21 = a21 / T
    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)

    term1_g1 = tau21 * (G21 / (x1 + x2 * G21)) ** 2
    term2_g1 = (G12 * tau12) / ((x2 + x1 * G12) ** 2)
    ln_gamma1 = (x2 ** 2) * (term1_g1 + term2_g1)

    term1_g2 = tau12 * (G12 / (x2 + x1 * G12) ** 2)
    term2_g2 = (G21 * tau21) / ((x1 + x2 * G21) ** 2)
    ln_gamma2 = (x1 ** 2) * (term1_g2 + term2_g2)

    return np.exp(ln_gamma1), np.exp(ln_gamma2)


# 3. SLE Temperature Objective Functions (f(T) = 0)
def sle_objective_comp1(T, x1, a12, a21, alpha=0.3):
    x2 = 1.0 - x1
    g1, _ = nrtl_gamma(x1, x2, T, a12, a21, alpha)
    # SLE Equation: 1/T - 1/T_fus + R/H_fus * ln(x1 * gamma1) = 0
    return (1.0 / T) - (1.0 / T_fus_1) + (R / H_fus_1) * np.log(x1 * g1)

def sle_objective_comp2(T, x2, a12, a21, alpha=0.3):
    x1 = 1.0 - x2
    _, g2 = nrtl_gamma(x1, x2, T, a12, a21, alpha)
    return (1.0 / T) - (1.0 / T_fus_2) + (R / H_fus_2) * np.log(x2 * g2)


# Function to solve SLE temperature at a given x1
def get_sle_temperature(x1, a12, a21, alpha=0.3, component=1, T_guess=500.0):
    if component == 1:
        res = root_scalar(
            sle_objective_comp1,
            args=(x1, a12, a21, alpha),
            x0=T_guess,
            method="secant"
        )
    else:
        x2 = 1.0 - x1
        res = root_scalar(
            sle_objective_comp2,
            args=(x2, a12, a21, alpha),
            x0=T_guess,
            method="secant"
        )
    return res.root if res.converged else np.nan


# 4. Objective Function to Fit NRTL Parameters to Experimental Data
def objective(params, x1_data, x2_data, T_data, g1_true, g2_true, alpha=0.3):
    a12, a21 = params
    g1_pred, g2_pred = nrtl_gamma(x1_data, x2_data, T_data, a12, a21, alpha)
    err1 = np.mean(((g1_pred - g1_true) / g1_true) ** 2)
    err2 = np.mean(((g2_pred - g2_true) / g2_true) ** 2)
    return err1 + err2


# 5. Load Data & Fit Parameters
excel_file = "data/input/gammaInput.xlsx"
df = pd.read_excel(excel_file)
df.columns = df.columns.str.strip()

x1_exp = df["Composition (1)"].values
x2_exp = df["Composition (2)"].values
T_exp = df["Temperature"].values
g1_exp = df["Activity Coefficient (1)"].values
g2_exp = df["Activity Coefficient (2)"].values

# Fit a12 and a21
res = minimize(
    objective,
    [100.0, 100.0],
    args=(x1_exp, x2_exp, T_exp, g1_exp, g2_exp, 0.3),
    method="Nelder-Mead"
)
a12_opt, a21_opt = res.x

# 6. Compute SLE Temperature Curve Across Composition Domain
x1_grid = np.linspace(0.80, 0.98, 100)

# Calculating SLE Liquidus Line for Component 1 (the main component in your dataset)
T_sle_nrtl = [
    get_sle_temperature(x, a12_opt, a21_opt, alpha=0.3, component=1, T_guess=500.0)
    for x in x1_grid
]

# 7. Plotting Experimental T vs. NRTL Calculated SLE T
plt.figure(figsize=(8, 5))

# Experimental Points
plt.scatter(
    x1_exp, T_exp,
    color='crimson', s=90, zorder=5, label='Experimental SLE Data ($T_{\mathrm{exp}}$)'
)

# NRTL Calculated SLE Curve
plt.plot(
    x1_grid, T_sle_nrtl,
    'b-', linewidth=2, label='NRTL SLE Liquidus Curve ($T_{\mathrm{calc}}$)'
)

plt.xlabel('Composition $x_1$ (tetramethylammonium chloride)', fontsize=11)
plt.ylabel('Temperature (K)', fontsize=11)
plt.title('Solid-Liquid Equilibrium (SLE): Experimental vs. NRTL Fit', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')

plt.tight_layout()
plt.show()