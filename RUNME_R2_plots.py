import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# USER CONFIGURATION
# ==========================================
# 1. File paths
actual_xlsx = Path("data/input/pureComponents_2026DecJANNAF_training_actual.xlsx")
predicted_xlsx = Path("data/output/MTGPR_Tm_Hfus_TestRSME000223/pureComponents_filled_training_predicted.xlsx")

# 2. Sheets to process (Set to None to process all sheets, or specify list: ["Sheet1", "Sheet2"])
actual_sheets = None
predicted_sheets = None

# 3. Key column to match rows across files (e.g., Chemical Name, CAS Number, ID)
id_column = "Full Name"

pred_scale_factor = 1.00
# 4. Target column search keywords
# The script will look for columns containing these strings (case-insensitive)
actual_target_keyword = "Melting Temperature [K]"
predicted_target_keyword = "Melting Temperature [K]"
dataset = "train"

# actual_target_keyword = "Enthalpy of Fusion [kJ/mol]"
# predicted_target_keyword = "Enthalpy of Fusion [kJ/mol]"

# 5. Global Plot Formatting & Style Options
PLOT_CONFIG = {
    "save_dir": Path("data/output/parity_plots"),
    "fig_size": (3, 3),
    "dpi": 300,
    "display_dpi": 120,
    # Scatter plot style
    "scatter_color": "#000000",
    "scatter_edgecolor": "black",
    "scatter_alpha": 0.75,
    "scatter_size": 45,
    "scatter_label": "Data Points",
    # 1:1 Parity line style
    "line_color": "#CFB991",
    "line_style": "--",
    "line_width": 3.0,
    "line_label": "1:1 Parity Line",
    # Stats annotation box
    "stats_box_loc": (0.05, 0.92),
    "stats_fontsize": 8,
    "stats_box_style": dict(
        boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"
    ),
    # Axes & Labels
    "legend_fontsize": 8,
    "label_fontsize": 11,
    "title_fontsize": 12,
    "grid_linestyle": "--",
    "grid_alpha": 0.5,
    "legend_loc": "lower right",
}

PLOT_CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
# ==========================================


def parse_property_and_unit(target_str):
    """Extracts property name and unit from strings formatted as 'Property Name [Unit]'."""
    match = re.search(r"^(.*?)(?:\s*\[(.*?)\])?$", target_str.strip())
    if match:
        prop_name = match.group(1).strip()
        unit = match.group(2).strip() if match.group(2) else ""
        return prop_name, unit
    return target_str.strip(), ""


def load_and_combine_sheets(xlsx_path, sheets_to_load):
    """Loads specified or all sheets from an Excel file into a single DataFrame."""
    excel_file = pd.ExcelFile(xlsx_path)
    available = excel_file.sheet_names

    target_sheets = available if sheets_to_load is None else sheets_to_load
    df_list = []

    for sheet in target_sheets:
        if sheet in available:
            df = pd.read_excel(excel_file, sheet_name=sheet)
            df.columns = df.columns.astype(str).str.strip()
            df_list.append(df)
        else:
            print(
                f"Warning: Sheet '{sheet}' not found in {xlsx_path.name}. Skipping."
            )

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


def find_matching_column(df, keyword, col_type_name):
    """Finds the first column containing the keyword (case-insensitive)."""
    matches = [c for c in df.columns if keyword.lower() in c.lower()]
    if matches:
        return matches[0]
    raise KeyError(
        f"Could not find any column matching keyword '{keyword}' for {col_type_name} in columns: {list(df.columns)}"
    )


# 1. Parse Target Property and Unit
prop_name_act, unit_act = parse_property_and_unit(actual_target_keyword)
prop_name_pred, unit_pred = parse_property_and_unit(predicted_target_keyword)

# Fallback unit handling
unit_display = f" [{unit_act}]" if unit_act else ""

# 2. Load data from both Excel files
print("Loading Excel files...")
df_actual = load_and_combine_sheets(actual_xlsx, actual_sheets)
df_pred = load_and_combine_sheets(predicted_xlsx, predicted_sheets)

if df_actual.empty or df_pred.empty:
    raise ValueError("One or both input Excel files yielded no data.")

# 3. Identify target value columns
actual_orig_col = find_matching_column(
    df_actual, actual_target_keyword, "actual values"
)
pred_orig_col = find_matching_column(
    df_pred, predicted_target_keyword, "predicted values"
)

# 4. Clean Missing Data and Identifiers
df_actual.replace(["—", "–", "-"], np.nan, inplace=True)
df_pred.replace(["—", "–", "-"], np.nan, inplace=True)

# Find exact ID column case-insensitively
actual_id_col = find_matching_column(df_actual, id_column, "ID column (actual)")
pred_id_col = find_matching_column(df_pred, id_column, "ID column (predicted)")

# Normalize the ID column for reliable merging
df_actual["_merge_id"] = (
    df_actual[actual_id_col].astype(str).str.strip().str.lower()
)
df_pred["_merge_id"] = (
    df_pred[pred_id_col].astype(str).str.strip().str.lower()
)

# Extract and rename target columns explicitly to avoid merge conflict suffixes
df_act_sub = df_actual[["_merge_id", actual_orig_col]].rename(
    columns={actual_orig_col: "y_actual_val"}
)
df_pred_sub = df_pred[["_merge_id", pred_orig_col]].rename(
    columns={pred_orig_col: "y_pred_val"}
)

# Merge datasets on normalized ID
df_merged = pd.merge(df_act_sub, df_pred_sub, on="_merge_id", how="inner")

# Force columns to numeric and drop NaN pairs
df_merged["y_actual_val"] = pd.to_numeric(
    df_merged["y_actual_val"], errors="coerce"
)
df_merged["y_pred_val"] = pd.to_numeric(
    df_merged["y_pred_val"], errors="coerce"
)
df_clean = df_merged.dropna(subset=["y_actual_val", "y_pred_val"])

# 5. Compute Metrics & Dynamic Parity Plot
y_actual = df_clean["y_actual_val"].values
y_pred = (df_clean["y_pred_val"].values) / pred_scale_factor

if len(y_actual) == 0:
    print(
        "No matching data points found between actual and predicted files after dropping missing values."
    )
else:
    # Calculate R^2 and RMSE
    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    # Plot Layout
    fig, ax = plt.subplots(
        figsize=PLOT_CONFIG["fig_size"], dpi=PLOT_CONFIG["display_dpi"]
    )

    # Scatter plot
    ax.scatter(
        y_actual,
        y_pred,
        color=PLOT_CONFIG["scatter_color"],
        edgecolor=PLOT_CONFIG["scatter_edgecolor"],
        alpha=PLOT_CONFIG["scatter_alpha"],
        s=PLOT_CONFIG["scatter_size"],
        zorder=3,
        label=PLOT_CONFIG["scatter_label"],
    )

    # 1:1 Parity Line
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    padding = (max_val - min_val) * 0.05
    line_min, line_max = min_val - padding, max_val + padding

    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        color=PLOT_CONFIG["line_color"],
        linestyle=PLOT_CONFIG["line_style"],
        linewidth=PLOT_CONFIG["line_width"],
        label=PLOT_CONFIG["line_label"],
        zorder=2,
    )

    # Annotate R^2 and RMSE dynamically with extracted units
    rmse_str = f"{rmse:.2f}" if unit_act == "" else f"{rmse:.2f} {unit_act}"
    stats_text = f"$R^2 = {r2:.3f}$\n$\mathrm{{RMSE}}$ = {rmse_str}"

    ax.text(
        PLOT_CONFIG["stats_box_loc"][0],
        PLOT_CONFIG["stats_box_loc"][1],
        stats_text,
        transform=ax.transAxes,
        fontsize=PLOT_CONFIG["stats_fontsize"],
        verticalalignment="top",
        bbox=PLOT_CONFIG["stats_box_style"],
    )

    # Dynamic Formatting
    ax.set_xlabel(f"Actual {unit_display}", fontsize=PLOT_CONFIG["label_fontsize"])
    ax.set_ylabel(f"Predicted {unit_display}", fontsize=PLOT_CONFIG["label_fontsize"])
    ax.set_title(f"{prop_name_act}", fontsize=PLOT_CONFIG["title_fontsize"], fontweight="bold")
    
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=PLOT_CONFIG["grid_linestyle"], alpha=PLOT_CONFIG["grid_alpha"])
    ax.legend(
        frameon=True,
        loc=PLOT_CONFIG["legend_loc"],
        fontsize=PLOT_CONFIG["legend_fontsize"]
    )

    fig.tight_layout()

    # Save figure with sanitized property name in filename
    clean_filename = re.sub(r"[^\w\-]", "_", prop_name_act)
    save_path = PLOT_CONFIG["save_dir"] / f"Parity_Plot_{dataset}_{clean_filename}.png"
    fig.savefig(save_path, dpi=PLOT_CONFIG["dpi"])
    plt.close(fig)

    print(f"Successfully matched and plotted {len(y_actual)} data points.")
    print(f"Property: {prop_name_act}")
    print(f"R^2: {r2:.4f} | RMSE: {rmse:.4f} {unit_act}".strip())
    print(f"Saved parity plot to: {save_path}")