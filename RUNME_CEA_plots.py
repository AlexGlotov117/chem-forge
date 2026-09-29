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

CEA_TARGETS = {
    "Adiabatic Flame Temperature": {
        "search_key": "adiabatic flame temperature",
        "ylabel": "Adiabatic Flame Temperature [K]",
        "filename_tag": "FlameTemp",
    },
    "Characteristic Velocity": {
        "search_key": "characteristic velocity",
        "ylabel": "Characteristic Velocity [m/s]",
        "filename_tag": "CStar",
    },
    "Specific Impulse": {
        "search_key": "specific impulse",
        "ylabel": "Specific Impulse [s]",
        "filename_tag": "Isp",
    },
}

# 1. Load Experimental Data (only if needed)
df_exp = None
if plot_mode == "with_experimental":
    df_exp = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df_exp["Chemical (1)"] = df_exp["Chemical (1)"].astype(str).str.strip()
    df_exp["Chemical (2)"] = df_exp["Chemical (2)"].astype(str).str.strip()

# 2. Group predicted CSVs by system base name
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

# 3. Process each system
for system_base_name, file_paths in system_groups.items():
    if "_" not in system_base_name:
        continue

    comp1, comp2 = [c.strip() for c in system_base_name.split("_", 1)]

    # Filter experimental data check based on plot_mode
    df_exp_match = pd.DataFrame()
    mask_forward = pd.Series(dtype=bool)

    if plot_mode == "with_experimental" and df_exp is not None:
        mask_forward = (df_exp["Chemical (1)"].str.lower() == comp1.lower()) & (
            df_exp["Chemical (2)"].str.lower() == comp2.lower()
        )
        mask_reverse = (df_exp["Chemical (1)"].str.lower() == comp2.lower()) & (
            df_exp["Chemical (2)"].str.lower() == comp1.lower()
        )

        df_exp_match = df_exp[mask_forward | mask_reverse].copy()

        if df_exp_match.empty:
            continue

    # Extract data for x-axis (composition) and all CEA targets across MC samples
    x_pred = None
    target_samples = {target: [] for target in CEA_TARGETS}

    for csv_path in file_paths:
        df_pred = pd.read_csv(csv_path)
        df_pred.columns = df_pred.columns.str.strip().str.replace(r"\s+", " ", regex=True)

        # Identify composition column
        comp1_cols = [
            c for c in df_pred.columns if comp1.lower() in c.lower() and "molar composition" in c.lower()
        ]
        if not comp1_cols:
            comp1_cols = [c for c in df_pred.columns if "molar composition" in c.lower()]

        if not comp1_cols:
            continue

        if x_pred is None:
            x_pred = df_pred[comp1_cols[0]].values

        # Extract each target column value for this MC run
        for target_name, meta in CEA_TARGETS.items():
            matching_cols = [c for c in df_pred.columns if meta["search_key"] in c.lower()]
            if matching_cols:
                target_samples[target_name].append(df_pred[matching_cols[0]].values)

    # Generate plots for each CEA performance metric
    for target_name, meta in CEA_TARGETS.items():
        sample_list = target_samples[target_name]
        if not sample_list or x_pred is None:
            print(f"Skipping {target_name} for {system_base_name}: Missing column data.")
            continue

        # Convert to 2D matrix: (num_samples, num_points)
        matrix = np.array(sample_list)

        # Monte Carlo statistics
        y_mean = np.nanmean(matrix, axis=0)
        y_std = np.nanstd(matrix, axis=0)
        y_lower = y_mean - y_std
        y_upper = y_mean + y_std

        # Plotting
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

        # Plot MC Mean & Uncertainty Spread
        ax.plot(x_pred, y_mean, label="Monte Carlo Mean", color="#1f77b4", linewidth=2)
        ax.fill_between(
            x_pred,
            y_lower,
            y_upper,
            color="#1f77b4",
            alpha=0.3,
            label="Monte Carlo Spread ($\pm 1\sigma$)",
        )

        # Conditionally plot experimental data points (if present in Excel for this metric)
        if plot_mode == "with_experimental" and not df_exp_match.empty:
            exp_col_matches = [c for c in df_exp_match.columns if meta["search_key"] in c.lower()]
            if exp_col_matches:
                exp_col = exp_col_matches[0]
                if mask_forward.any():
                    x_exp = df_exp_match["Composition (1)"]
                else:
                    x_exp = 1.0 - df_exp_match["Composition (2)"]

                y_exp = df_exp_match[exp_col]

                ax.scatter(
                    x_exp,
                    y_exp,
                    color="#d62728",
                    edgecolor="black",
                    s=50,
                    zorder=5,
                    label="Experimental Data",
                )

        # Formatting
        ax.set_xlabel(f"Mole Fraction {comp1} ($x_1$)", fontsize=11)
        ax.set_ylabel(meta["ylabel"], fontsize=11)
        ax.set_title(f"CEA Performance ({target_name}): {comp1} + {comp2}", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, loc="best")

        fig.tight_layout()

        # Save figure
        mode_suffix = "WithExp" if plot_mode == "with_experimental" else "PredOnly"
        save_filename = f"{system_base_name}_CEA_{meta['filename_tag']}_{mode_suffix}.png"
        save_path = save_dir / save_filename
        
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

        print(f"Saved plot: {save_path} (from {len(sample_list)} MC samples)")