from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# plot_mode = "with_experimental"
plot_mode = "all_without_experimental"

# Define paths
output_dir = Path("data/output/ALL3_TestRMSE00241/CEA_SLE/testing")
save_dir = Path("data/output/ALL3_TestRMSE00241/CEA_SLE/testing")
xlsx_path = "data/input/pureComponents.xlsx"
sheet_name = "Binary Input"

# Create output folder for plots if it doesn't exist
save_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Experimental Data (only if needed)
df_exp = None
if plot_mode == "with_experimental":
    df_exp = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    # Ensure string types and clean column names
    df_exp["Chemical (1)"] = df_exp["Chemical (1)"].astype(str).str.strip()
    df_exp["Chemical (2)"] = df_exp["Chemical (2)"].astype(str).str.strip()

# 2. Iterate through predicted CSVs
csv_files = list(output_dir.glob("*.csv"))

if not csv_files:
    print(f"No CSV files found in {output_dir}")

system_groups = {}
for csv_path in csv_files:
    match = re.search(r"^(.*)_(\d+)\.csv$", csv_path.name)
    if not match:
        continue
    
    system_base_name = match.group(1)  # e.g., "CompA_CompB"
    system_groups.setdefault(system_base_name, []).append(csv_path)

# 3. Process each system across all Monte Carlo samples
for system_base_name, file_paths in system_groups.items():
    if "_" not in system_base_name:
        continue

    comp1, comp2 = [c.strip() for c in system_base_name.split("_", 1)]

    # Handle experimental data checking based on plot_mode
    df_exp_match = pd.DataFrame()
    mask_forward = pd.Series(dtype=bool)

    if plot_mode == "with_experimental" and df_exp is not None:
        # Filter matching Experimental Data (check both A_B and B_A order)
        mask_forward = (df_exp["Chemical (1)"].str.lower() == comp1.lower()) & (
            df_exp["Chemical (2)"].str.lower() == comp2.lower()
        )
        mask_reverse = (df_exp["Chemical (1)"].str.lower() == comp2.lower()) & (
            df_exp["Chemical (2)"].str.lower() == comp1.lower()
        )

        df_exp_match = df_exp[mask_forward | mask_reverse].copy()

        # Skip plotting if NO experimental data exists for this pair in this mode
        if df_exp_match.empty:
            continue

    # Aggregate predictions across all sample runs
    sample_t_list = []
    x_pred = None

    for csv_path in file_paths:
        df_pred = pd.read_csv(csv_path)
        df_pred.columns = df_pred.columns.str.strip()

        # Map composition and SLE temperature columns
        comp1_cols = [
            c for c in df_pred.columns if comp1.lower() in c.lower() and "molar composition" in c.lower()
        ]
        if not comp1_cols:
            comp1_cols = [c for c in df_pred.columns if "molar composition" in c.lower()]

        sle_temp_cols = [c for c in df_pred.columns if "solid-liquid equilibrium temperature" in c.lower()]

        if not comp1_cols or not sle_temp_cols:
            continue

        if x_pred is None:
            x_pred = df_pred[comp1_cols[0]].values

        sample_t_list.append(df_pred[sle_temp_cols[0]].values)

    if not sample_t_list:
        print(f"Skipping {system_base_name}: Could not process sample CSVs.")
        continue

    # Convert to 2D array: (num_samples, num_points)
    t_matrix = np.array(sample_t_list)

    # Calculate Monte Carlo statistics across samples
    t_mean = np.nanmean(t_matrix, axis=0)
    t_std = np.nanstd(t_matrix, axis=0)
    t_lower = t_mean - t_std
    t_upper = t_mean + t_std

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    # Plot Monte Carlo mean line
    ax.plot(x_pred, t_mean, label="Monte Carlo Mean", color="#1f77b4", linewidth=2)

    # Plot Monte Carlo uncertainty spread shading (±1 std)
    ax.fill_between(
        x_pred,
        t_lower,
        t_upper,
        color="#1f77b4",
        alpha=0.3,
        label="Monte Carlo Spread",
    )

    # Conditionally plot experimental points
    if plot_mode == "with_experimental" and not df_exp_match.empty:
        if mask_forward.any():
            x_exp = df_exp_match["Composition (1)"]
        else:
            x_exp = 1.0 - df_exp_match["Composition (2)"]

        t_exp = df_exp_match["Temperature"]

        # Plot experimental points
        ax.scatter(
            x_exp,
            t_exp,
            color="#d62728",
            edgecolor="black",
            s=50,
            zorder=5,
            label="Experimental Data",
        )

    # Formatting
    ax.set_xlabel(f"Mole Fraction {comp1} ($x_1$)", fontsize=11)
    ax.set_ylabel("Temperature [K]", fontsize=11)
    ax.set_title(f"SLE Phase Diagram: {comp1} + {comp2}", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="best")

    fig.tight_layout()

    # Save figure
    mode_suffix = "WithExp" if plot_mode == "with_experimental" else "PredOnly"
    save_path = save_dir / f"{system_base_name}_SLE_MonteCarlo_{mode_suffix}.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved plot: {save_path} (from {len(file_paths)} MC samples)")