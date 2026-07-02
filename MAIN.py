import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from chemicals import Compound, Mixture

pureComponents = pd.read_excel("pureComponents.xlsx", sheet_name="Input")
cominations = pd.read_excel("combinations.xlsx", sheet_name=None)

def create_compound_from_smiles(input_smiles: str) -> Compound:
    """
    Validates the input SMILES, converts it to a standard canonical format,
    looks up thermodynamic data from the database, and auto-generates 
    molecular properties using RDKit.
    """
    mol = Chem.MolFromSmiles(str(input_smiles).strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {input_smiles}")
    canonical_key = Chem.MolToSmiles(mol)
    
    row = pureComponents[pureComponents["SMILES"] == canonical_key].iloc[0]
        
    # Generate atomic formula dynamically for NASA-CEA
    mol_with_hs = Chem.AddHs(mol)
    formula_dict = {}
    for atom in mol_with_hs.GetAtoms():
        symbol = atom.GetSymbol()
        formula_dict[symbol] = formula_dict.get(symbol, 0) + 1
    
    print(row)
    return Compound(
        name=str(row["Full Name"]),  
        mw=ExactMolWt(mol),
        T_fus=float(row["Melting Temp\n[K]"]),
        h_fus=float(row["Heat of Fusion\n[J/mol]"]),
        h_f_298=float(row["Standard Heat of Formation\n[J/mol]"]),
        formula=formula_dict
    )

# 3. Loop through each sheet (Binary, Ternary, Quaternary, etc.)
for sheet_name, df_combos in cominations.items():
    print(f"\n" + "="*50)
    print(f"  STARTING SCREENING FOR SHEET: {sheet_name}")
    print("="*50)
    
    # Identify component columns dynamically for this specific sheet
    component_cols = [col for col in df_combos.columns if col.startswith("Component ")]
    
    # Loop through every row/combination in the current sheet
    for idx, row in df_combos.iterrows():
        row_smiles = [str(row[col]).strip() for col in component_cols if pd.notna(row[col])]
        
        if not row_smiles:
            continue
            
        try:
            # Generate the dynamic compound arrays using the SMILES strings directly
            compounds_list = [create_compound_from_smiles(s) for s in row_smiles]
            
            # Use the display symbols to format a clean filename string
            system_label = "_".join([c.name for c in compounds_list])
            print(f" -> Sweeping system [{idx+1}]: {system_label}")
            
            # Build your uniform mixture engine
            mixture = Mixture(compounds=compounds_list)
            
            # --- Execute your composition grid sweep and write out raw CSVs ---
            
        except Exception as e:
            print(f"Skipping index row {idx} due to calculation error: {e}")

# # =============================================================================
# # 1. DYNAMIC SYSTEM INITIALIZATION (inputs here)
# # =============================================================================
# # Define everything in one place
# compounds_list = [
#     Compound(
#         name="MEA", 
#         mw=61.084, 
#         T_fus=283.45, 
#         h_fus=13800, 
#         h_f_298=-507500,
#         formula={"C": 2, "H": 7, "N": 1, "O": 1}
#     ),
#     Compound(
#         name="Urea", 
#         mw=60.05534, 
#         T_fus=406.15, 
#         h_fus=10340, 
#         h_f_298=-333330,
#         formula={"C": 1, "N": 2, "H": 4, "O": 1}
#     )
# ]
# num_points = 101


# # Build the generic mixture object
# mixture = Mixture(compounds=compounds_list)

# if mixture.num_components != 2:
#     raise ValueError(f"This sweeping function is designed for binary (2-component) mixtures. "
#                         f"Provided mixture has {mixture.num_components} components: {mixture.names}")

# # Extract dynamic labels from the mixture instance
# comp_0_name, comp_1_name = mixture.names

# # Generate a smooth composition grid for the primary component (0.0 to 1.0)
# x_grid = np.linspace(0.0, 1.0, num_points)

# t_melting_points = []
# isp_values = []

# # Sweep across the composition axis
# for x0 in x_grid:
#     # Vectorized array allocation summing exactly to 1.0
#     x_vec = np.array([x0, 1.0 - x0])
    
#     # Pass context to the state machine
#     mixture.set_composition(x=x_vec)
    
#     # Safely extract properties on-the-fly
#     t_melting_points.append(mixture.T_fus)
    
#     # Optional safeguard: capture Isp if CEA supports the chemistry, otherwise append NaN
#     isp_values.append(mixture.isp[1])
        
# # Plot Generation
# import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots(figsize=(8, 5))

# # Left Y-Axis: Liquidus / Phase Behavior
# color_sle = 'tab:blue'
# ax1.set_xlabel(f'Mole Fraction of {comp_0_name} ($x_{{{comp_0_name}}}$)')
# ax1.set_ylabel('Liquidus Temperature $T_m$ (K)', color=color_sle)
# ax1.plot(x_grid, t_melting_points, color=color_sle, linewidth=2, label="Liquidus Envelope")
# ax1.tick_params(axis='y', labelcolor=color_sle)
# ax1.grid(True, linestyle=':', alpha=0.6)

# ax2 = ax1.twinx()  
# color_cea = 'tab:red'
# ax2.set_ylabel('CEA Performance', color=color_cea)
# ax2.plot(x_grid, isp_values, color=color_cea, linestyle='--', linewidth=2, label="$I_{sp}$")
# ax2.tick_params(axis='y', labelcolor=color_cea)

# fig.tight_layout()
# plt.title(f"{comp_0_name} + {comp_1_name}")
# plt.show()