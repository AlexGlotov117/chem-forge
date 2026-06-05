import numpy as np
import matplotlib.pyplot as plt

from solvers import SLESolver
from chemicals import Compound, Mixture

Cyclohexane = Compound(name="Cyclohexane", mw=84.16, t_fus=279.84, h_fus=2628)
Urea = Compound(name="Urea", mw=60.056, t_fus=407.2, h_fus=14600.0)

Cyclohexane_Urea = Mixture(compounds=[Cyclohexane, Urea])

results = SLESolver().solve(Cyclohexane_Urea, steps=101)
print(results)
## Plots
x1 = results[:, 1]  # Mole fraction of Cyclohexane
T_SLE = results[:, 2] # Calculated system liquidus temperature

# --- 3. Plotting the Phase Diagram ---
plt.figure(figsize=(6, 4.5), dpi=180)
plt.rcParams.update({'font.size': 10})

# Plot the calculated liquidus curve
plt.plot(x1 * 100.0, T_SLE, '--', linewidth=2, color='tab:blue', label="Ideal SLE")

# Identify the eutectic point mathematically from your results matrix
eutectic_idx = T_SLE.argmin()
x_eutectic = x1[eutectic_idx]
T_eutectic = T_SLE[eutectic_idx]

# Mark the eutectic point on the plot
plt.plot(x_eutectic * 100.0, T_eutectic, 'ro', label=f"Eutectic ({x_eutectic:.2f}, {T_eutectic:.1f} K)")
plt.axhline(y=T_eutectic, color='gray', linestyle=':', alpha=0.7)

# Format the plot properly
plt.title(f"Binary Phase Diagram: {Cyclohexane.name} - {Urea.name}")
plt.xlabel(f"Mole Percent of {Cyclohexane.name} (%)")
plt.ylabel("Temperature (K)")
plt.xlim(0, 100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="upper center")

plt.tight_layout()
plt.show()