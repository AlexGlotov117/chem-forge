from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Define paths
output_dir = Path("data/output/20230921_WithHANNA_WithGP")
save_dir = Path("data/output/20230921_WithHANNA_WithGP/comparison_plots")
xlsx_path = "data/input/pureComponents.xlsx"
sheet_name = "Binary Input"

# Create output folder for plots if it doesn't exist
save_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Experimental Data
df_exp = pd.read_excel(xlsx_path, sheet_name=sheet_name)

# Ensure string types and clean column names
df_exp["Chemical (1)"] = df_exp["Chemical (1)"].astype(str).str.strip()
df_exp["Chemical (2)"] = df_exp["Chemical (2)"].astype(str).str.strip()

# 2. Iterate through predicted CSVs
csv_files = list(output_dir.glob("*.csv"))

if not csv_files:
    print(f"No CSV files found in {output_dir}")

for csv_path in csv_files:
    pair_name = csv_path.stem
    if "_" not in pair_name:
        continue

    comp1, comp2 = [c.strip() for c in pair_name.split("_", 1)]

    # Filter matching Experimental Data (check both A_B and B_A order)
    mask_forward = (df_exp["Chemical (1)"].str.lower() == comp1.lower()) & (
        df_exp["Chemical (2)"].str.lower() == comp2.lower()
    )
    mask_reverse = (df_exp["Chemical (1)"].str.lower() == comp2.lower()) & (
        df_exp["Chemical (2)"].str.lower() == comp1.lower()
    )

    df_exp_match = df_exp[mask_forward | mask_reverse].copy()

    # Skip plotting if NO experimental data exists for this pair
    if df_exp_match.empty:
        continue

    # Load predicted CSV
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
        print(f"Skipping {csv_path.name}: Could not map required CSV columns.")
        continue

    x_pred = df_pred[comp1_cols[0]] 
    t_pred = df_pred[sle_temp_cols[0]]

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    # Plot predicted curve
    ax.plot(x_pred, t_pred, label="Predicted (HANNA / Model)", color="#1f77b4", linewidth=2)

    # Extract experimental x and T
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

    # Save figure and close to free memory
    save_path = save_dir / f"{pair_name}_SLE_Comparison.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved plot: {save_path}")