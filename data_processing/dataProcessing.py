import csv
import numpy as np
import pandas as pd

# from __temp__.alexChemicals import Mixture

# def exportSLE(filename: str, mixture: Mixture, x_matrix: np.ndarray, T_matrix: np.ndarray, T_sle: np.ndarray):
#     """
#     Dynamically generates and writes SLE phase data to a CSV file for N components.
#     """
#     # Build dynamic headers based on compound names
#     headers = []
#     for comp in mixture.compounds:
#         headers.append(f"x_{comp.name}")
        
#     for comp in mixture.compounds:
#         headers.append(f"T_{comp.name} (K)")
        
#     headers.append("T_SLE (K)")
    
#     # Combine all arrays side-by-side into one large data matrix
#     combined_data = np.column_stack((x_matrix, T_matrix, T_sle))
    
#     # Write data to disk
#     with open(filename, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(headers)      # Write the header row
#         writer.writerows(combined_data)  # Write all data rows
        
#     print(f"--> Phase diagram data successfully exported to: {filename}")

def createUniqueCombos(input_file: str, output_file: str = "combinations.xlsx", num_components: int = 2):
    from rdkit import Chem
    print(f"Loading compounds from {input_file}...")

    df_pure = pd.read_excel(input_file, sheet_name="Input")

    # Extract and canonicalize unique SMILES
    unique_smiles = set()

    for idx, raw_smiles in enumerate(df_pure["SMILES"]):
        if pd.isna(raw_smiles):
            continue
        
        # Clean string and convert to canonical format to ensure uniqueness
        clean_smiles = str(raw_smiles).strip()
        mol = Chem.MolFromSmiles(clean_smiles)
        
        if mol:
            canonical_key = Chem.MolToSmiles(mol)
            unique_smiles.add(canonical_key)
        else:
            print(f"Warning: Row {idx} has an invalid SMILES string: {clean_smiles}")

    smiles_list = sorted(list(unique_smiles))
    print(f"Found {len(smiles_list)} unique chemical species to combine.\n")

    # Generate combinations and write to sheets
    from itertools import combinations
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        
        # Sweep from Binary (2) up to your max specified component size
        for n in range(2, num_components + 1):
            sheet_name = f"{n}-Component"
                
            print(f"Generating all possible combinations for: {sheet_name}...")
            
            # itertools.combinations mathematically guarantees unique entries (order-independent)
            all_combos = list(combinations(smiles_list, n))
            
            if not all_combos:
                print(f" -> No combinations possible for {sheet_name}.")
                continue
                
            # Dynamically build standard tracking columns
            column_headers = [f"Component {i+1}" for i in range(n)]
            
            # Turn the list of combination tuples into a DataFrame layout
            df_combos = pd.DataFrame(all_combos, columns=column_headers)
            
            # Write this specific matrix tab into the workbook
            df_combos.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f" -> Successfully wrote {len(df_combos)} systems to sheet '{sheet_name}'.")

    print(f"\nProcessing complete! New matrix catalog compiled at: {output_file}")

def plotScreeningResults(results_dir="screening_results", 
                           y1_col="Solid-Liquid Equilibrium Temperature \n[K]", 
                           y2_col="Specific Impulse \n[s]",
                           y1_label="SLE Temperature (K)", 
                           y2_label="Isp (s)"):
    """
    Scans the results directory and generates a dual-axis line plot 
    for each mixture sweep CSV found.
    """
    import matplotlib.pyplot as plt
    import os

    # 1. Verify the folder exists
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist.")
        return

    # 2. Find all sweep CSV files
    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    
    if not csv_files:
        print(f"No sweep files found in '{results_dir}'.")
        return

    print(f"Found {len(csv_files)} data files to plot. Generating figures...")

    print(csv_files)
    for file_name in csv_files:
        file_path = os.path.join(results_dir, file_name)
        # 1. Load data and clean column headers
        df = pd.read_parquet(file_path) if file_name.endswith('.parquet') else pd.read_csv(file_path)

        if df.empty:
            continue

        # Remove embedded newlines and extra spaces from headers
        df.columns = df.columns.str.replace('\n', '').str.strip()

        # 2. Match variables cleanly
        x_cols = [col for col in df.columns if "Molar Composition" in col]
        if not x_cols:
            continue

        x_col = x_cols[0]
        y1_col = "Solid-Liquid Equilibrium Temperature [K]"
        y2_col = "Specific Impulse [s]"  # Or "Characteristic Velocity [m/s]" / "Adiabatic Flame Temperature [K]"

        # 3. Plotting routine
        fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

        # Primary Axis (SLE)
        color_y1 = '#1f77b4'
        ax1.set_xlabel(x_col, fontsize=11)
        ax1.set_ylabel(y1_col, color=color_y1, fontsize=11)
        line1 = ax1.plot(df[x_col], df[y1_col], color=color_y1, marker="o", markersize=3, linestyle="-", label=y1_col)
        ax1.tick_params(axis='y', labelcolor=color_y1)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Secondary Axis (CEA)
        ax2 = ax1.twinx()
        color_y2 = '#d62728'
        ax2.set_ylabel(y2_col, color=color_y2, fontsize=11)
        line2 = ax2.plot(df[x_col], df[y2_col], color=color_y2, marker="s", markersize=3, linestyle="--", label=y2_col)
        ax2.tick_params(axis='y', labelcolor=color_y2)

        # Legend & Layout
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

        plt.title(file_name.replace(".csv", "").replace("_", " vs "), fontsize=12, fontweight='bold', pad=12)
        plt.savefig(os.path.join(results_dir, file_name.replace(".csv", "--SLEvsISP.png")), bbox_inches='tight')
        plt.close()
        
        print(f" -> Compiled and saved plot: {file_path}")

from rdkit import Chem

def canonicalize_smiles(smiles_list):
    canonical_smiles = []

    for smiles in smiles_list:
        # Parse the SMILES string into an RDKit Molecule object
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            # Convert back to SMILES (RDKit generates canonical SMILES by default)
            can_smiles = Chem.MolToSmiles(mol, canonical=True)
            canonical_smiles.append(can_smiles)
        else:
            # Handle invalid SMILES strings
            canonical_smiles.append(None)

    return canonical_smiles

def extract_smiles_and_targets(filepath, target_columns=None, smiles_column='SMILES'):
    """
    Extracts raw SMILES strings and property target values (Y) from an Excel file.
    
    Parameters:
    - filepath: Path to the Excel (.xlsx) file.
    - target_columns: List of target column names in the Excel file.
    - smiles_column: Name of the column containing SMILES strings.
    
    Returns:
    - smiles_data: 1D numpy array of SMILES strings.
    - Y: 2D numpy array of target property values.
    """
    if target_columns is None:
        target_columns = ['T_m', 'dH_fus', 'dH_f']
        
    try:
        df = pd.read_excel(filepath, na_values=["—", "-", "N/A"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the Excel file at '{filepath}'.")
    
    # Ensure target columns exist in the DataFrame
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in Excel file: {missing_targets}")
    
    # Find SMILES column (case-insensitive check if exact match missing)
    if smiles_column not in df.columns:
        matched_cols = [c for c in df.columns if str(c).strip().upper() == smiles_column.upper()]
        if matched_cols:
            smiles_column = matched_cols[0]
        else:
            raise ValueError(f"Could not find SMILES column '{smiles_column}' in Excel file.")
            
    # Drop rows missing  SMILES strings
    df = df.dropna(subset=[smiles_column])
    
    smiles_data = df[smiles_column].astype(str).values
    Y = df[target_columns].values

    return smiles_data, Y

import os
import itertools

def generate_combinations(pure_df, arities=[2], output_file="data/output/combinations.xlsx"):
    """
    Generates N-ary unique component combinations (Binary, Ternary, etc.) from pure compounds.
    """
    valid_smiles = pure_df["SMILES"].dropna().unique().tolist()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for n in arities:
            sheet_name = {2: "Binary", 3: "Ternary", 4: "Quaternary"}.get(n, f"{n}-ary")
            combos = list(itertools.combinations(valid_smiles, n))
            
            col_names = [f"Component {i+1}" for i in range(n)]
            df_combos = pd.DataFrame(combos, columns=col_names)
            df_combos.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"[Workflow Prep] Combinations successfully saved to '{output_file}'.")

def prepare_MTGPR_Tm_Hfus_Hf(
    excel_path="data/input/pureComponents.xlsx", 
    predictor = None,
    combination_arities=[2],
    output_dir="data/output"
):
    """
    Checks for missing physical properties in pureComponents.xlsx, triggers GPR 
    imputation if needed, and writes populated inputs + system combinations.
    """
    df = pd.read_excel(excel_path, sheet_name="Input")
    
    # Target column identifiers
    tm_col = "Melting Temperature [K]"
    hfus_col = "Enthalpy of Fusion [kJ/mol]"
    hf_col = "Enthalpy of Formation [kJ/mol]"
    
    target_cols = [tm_col, hfus_col, hf_col]
    missing_mask = df[target_cols].isna().any(axis=1)
    
    # Determine output directory (use predictor's output dir if available)
    if predictor is not None and hasattr(predictor, 'output_dir'):
        out_filled_dir = predictor.output_dir
    else:
        out_filled_dir = output_dir

    os.makedirs(out_filled_dir, exist_ok=True)

    if missing_mask.any():
        print(f"[Workflow Prep] Found {missing_mask.sum()} components with missing properties.")
        
        if predictor is None:
            raise ValueError("[Workflow Prep] Missing properties detected, but no 'predictor' instance was passed.")

        # Ensure model weights/states are loaded
        if hasattr(predictor, 'is_trained') and not predictor.is_trained:
            predictor.load_model()

        missing_rows = df[missing_mask]
        preds = predictor.predict_smiles(missing_rows["SMILES"].tolist())

        # Impute missing entries selectively
        for idx in missing_rows.index:
            rel_i = missing_rows.index.get_loc(idx)
            if pd.isna(df.at[idx, tm_col]):
                df.at[idx, tm_col] = preds["T_m"][rel_i]
            if pd.isna(df.at[idx, hfus_col]):
                df.at[idx, hfus_col] = preds["dH_fus"][rel_i]
            if pd.isna(df.at[idx, hf_col]):
                df.at[idx, hf_col] = preds["dH_f"][rel_i]
                
        filled_excel_path = os.path.join(out_filled_dir, "pureComponents_filled.xlsx")
        with pd.ExcelWriter(filled_excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Input", index=False)
        print(f"[Workflow Prep] Updated filled component sheet saved to '{filled_excel_path}'.")
    else:
        print("[Workflow Prep] All pure component properties provided. Skipping GPR imputation.")
        filled_excel_path = os.path.join(out_filled_dir, "pureComponents_filled.xlsx")
        with pd.ExcelWriter(filled_excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Input", index=False)

    combos_path = os.path.join(out_filled_dir, "combinations.xlsx")
    generate_combinations(df, arities=combination_arities, output_file=combos_path)

    return filled_excel_path, combos_path