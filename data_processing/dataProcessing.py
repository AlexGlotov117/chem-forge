import csv
import numpy as np
import pandas as pd

from __temp__.alexChemicals import Mixture

def exportSLE(filename: str, mixture: Mixture, x_matrix: np.ndarray, T_matrix: np.ndarray, T_sle: np.ndarray):
    """
    Dynamically generates and writes SLE phase data to a CSV file for N components.
    """
    # Build dynamic headers based on compound names
    headers = []
    for comp in mixture.compounds:
        headers.append(f"x_{comp.name}")
        
    for comp in mixture.compounds:
        headers.append(f"T_{comp.name} (K)")
        
    headers.append("T_SLE (K)")
    
    # Combine all arrays side-by-side into one large data matrix
    combined_data = np.column_stack((x_matrix, T_matrix, T_sle))
    
    # Write data to disk
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)      # Write the header row
        writer.writerows(combined_data)  # Write all data rows
        
    print(f"--> Phase diagram data successfully exported to: {filename}")

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

    for file_name in csv_files:
        file_path = os.path.join(results_dir, file_name)
        df = pd.read_parquet(file_path) if file_name.endswith('.parquet') else pd.read_csv(file_path)
        
        if df.empty:
            continue

        # 3. Dynamically find the composition column (the x-axis variable)
        # Assumes columns are named like 'x_Urea', 'x_MEA', etc.
        x_cols = [col for col in df.columns if col.endswith("Molar Composition \n[%]")]
        if not x_cols:
            print(f"Skipping {file_name}: No composition columns found.")
            continue
        
        # For binary sweeps, we can just use the first component as our independent variable axis
        x_col = x_cols[0] 
        system_title = file_name.replace(".csv", "").replace("_", " vs ")

        # 4. Initialize the figure structure
        fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

        # Plot Primary Axis (e.g., SLE Melting Curve)
        color_y1 = '#1f77b4' # Deep blue
        ax1.set_xlabel(f"{x_col}", fontsize=11)
        ax1.set_ylabel(y1_label, color=color_y1, fontsize=11)
        line1 = ax1.plot(df[x_col], df[y1_col], color=color_y1, marker=".", linestyle="None", label=y1_label)
        ax1.tick_params(axis='y', labelcolor=color_y1)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # 5. Instantiates a second axes that shares the same x-axis
        ax2 = ax1.twinx()  
        # Plot Secondary Axis (e.g., Performance Characteristic Velocity)
        color_y2 = '#d62728' # Deep red
        ax2.set_ylabel(y2_label, color=color_y2, fontsize=11)
        line2 = ax2.plot(df[x_col], df[y2_col], color=color_y2, marker=".", linestyle="None", label=y2_label)
        ax2.tick_params(axis='y', labelcolor=color_y2)

        # 6. Formatting and Titles
        plt.title(f"{system_title}", fontsize=12, fontweight='bold', pad=12)
        
        # Combined Legend configuration
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

        # 7. Save the figure alongside the data
        fig_name = file_name.replace(".csv", "--SLEvsISP.png")
        fig_path = os.path.join(results_dir, fig_name)
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        
        print(f" -> Compiled and saved plot: {fig_path}")