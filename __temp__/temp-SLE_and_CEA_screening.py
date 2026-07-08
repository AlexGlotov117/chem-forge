import os
import numpy as np
import pandas as pd

from __temp__.alexChemicals import Compound, Mixture
from models.solvers import generate_x_grid
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

output_dir = "screening_results"

pureComponents = pd.read_excel("pureComponents.xlsx", sheet_name="Input")
cominations = pd.read_excel("combinations.xlsx", sheet_name=None)

num_points = 101

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
    
    # Force a fallback to a short name if Symbol is missing, and slice at 15 chars max
    cea_name = str(row["Full Name"]).strip()[:15] # Hard limit to 15 characters for CEA safety

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

            # Generate the composition matrix
            x_grid_matrix = generate_x_grid(num_components=mixture.num_components, steps=num_points)

            records = []

            # 3. Sweep across the N-dimensional composition matrix
            for x_vec in x_grid_matrix:
                # Pass the current composition row vector (e.g., [0.2, 0.5, 0.3]) to the state machine
                mixture.set_composition(x=x_vec)
                
                # 4. Safely extract properties on-the-fly with a fallback catch
                try:
                    # If mixture.isp returns a list/array index accordingly, adjust to match your class output
                    current_isp = mixture.isp[2]
                    current_t_adi = mixture.T_adi[0]
                    current_c_star = mixture.c_star[0]
                except Exception:
                    current_isp = np.nan
                    current_t_adi = np.nan
                    current_c_star = np.nan

                # 5. Build a dynamic row record mapping compositions back to column names cleanly
                row_record = {}
                for i, name in enumerate(mixture.names):
                    row_record[f"{name} Molar Composition \n[%]"] = x_vec[i]
                    # If your Mixture class tracks liquidus temperatures per component:
                    row_record[f"{name} Liquidus Temperature \n[K]"] = mixture.T_liq[i]
                    
                # Append the thermodynamic metrics
                row_record["Solid-Liquid Equilibrium Temperature \n[K]"] = mixture.T_fus
                row_record["Adiabatic Flame Temperature \n[K]"] = current_t_adi
                row_record["Characteristic Velocity \n[m/s]"] = current_c_star
                row_record["Specific Impulse \n[s]"] = current_isp
                
                records.append(row_record)

            # 6. Convert to DataFrame and save to CSV
            df_results = pd.DataFrame(records)
            
            system_label = "_".join(mixture.names)
            filename = f"{system_label}.csv"

            os.makedirs(output_dir, exist_ok=True)
            df_results.to_csv(os.path.join(output_dir, filename), index=False)
            
        except Exception as e:
            print(f"Skipping index row {idx} due to calculation error: {e}")