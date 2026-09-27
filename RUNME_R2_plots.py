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
actual_xlsx = Path("data/input/pureComponents_2026DecJANNAF_testing_actual.xlsx")
predicted_xlsx = Path("data/output/ALL3_TestRMSE00241/pureComponents_filled_testing_predicted_v2.xlsx")

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
dataset = "testing"

# actual_target_keyword = "Enthalpy of Fusion [kJ/mol]"
# predicted_target_keyword = "Enthalpy of Fusion [kJ/mol]"

# actual_target_keyword = "Enthalpy of Formation [kJ/mol]"
# predicted_target_keyword = "Enthalpy of Formation [kJ/mol]"

show_error_bars = True

# 5. Global Plot Formatting & Style Options
PLOT_CONFIG = {
    "save_dir": Path("data/output/parity_plots"),
    "fig_size": (3, 3),
    "dpi": 300,
    "display_dpi": 120,
    # Scatter plot style
    "scatter_color": "#8E6F3E", # melting
    # "scatter_color": "#DDB945", # enthlapy of fusion
    # "scatter_color": "#9D9795", # enthalpy of formation
    # "scatter_edgecolor": "black",
    "scatter_alpha": 1.0,
    "scatter_size": 45,
    "scatter_label": "Mean Value",
    # Error bars style
    # "errorbar_color": "#8E6F3E",
    "errorbar_alpha": 0.7,
    "errorbar_capsize": 10,
    "errorbar_linewidth": 3.0,
    # 1:1 Parity line style
    "line_color": "#000000",
    "line_style": "--",
    "line_width": 3.0,
    "line_label": "1:1 Parity Line",
    # Stats annotation box
    "stats_box_loc": (0.05, 0.92),
    "stats_fontsize": 8,
    "stats_box_style": dict(
        boxstyle="round,pad=0.5", facecolor="white", alpha=0.5, edgecolor="gray"
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

def construct_stdev_keyword(target_str):
    """Constructs property search string with STDEV placed before units, e.g., 'Melting Temperature STDEV [K]'."""
    prop_name, unit = parse_property_and_unit(target_str)
    return f"{prop_name} STDEV [{unit}]" if unit else f"{prop_name} STDEV"

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


def find_matching_column(df, keyword, col_type_name, optional=False):
    """Finds the first column containing the keyword (case-insensitive)."""
    matches = [c for c in df.columns if keyword.lower() in c.lower()]
    if matches:
        return matches[0]
    if optional:
        return None
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

tual_stdev_col = None
pred_stdev_col = None

if show_error_bars:
    act_stdev_kw = construct_stdev_keyword(actual_target_keyword)
    pred_stdev_kw = construct_stdev_keyword(predicted_target_keyword)

    actual_stdev_col = find_matching_column(
        df_actual, act_stdev_kw, "actual STDEV", optional=True
    )
    pred_stdev_col = find_matching_column(
        df_pred, pred_stdev_kw, "predicted STDEV", optional=True
    )

    if not actual_stdev_col:
        print(f"Warning: STDEV column '{act_stdev_kw}' not found in actual dataset.")
    if not pred_stdev_col:
        print(f"Warning: STDEV column '{pred_stdev_kw}' not found in predicted dataset.")

# 4. Clean Missing Data and Identifiers
df_actual.replace(["—", "–", "-"], np.nan, inplace=True)
df_pred.replace(["—", "–", "-"], np.nan, inplace=True)

# Find exact ID column case-insensitively
actual_id_col = find_matching_column(df_actual, id_column, "ID column (actual)")
pred_id_col = find_matching_column(df_pred, id_column, "ID column (predicted)")

df_actual["_merge_id"] = (
    df_actual[actual_id_col].astype(str).str.strip().str.lower()
)
df_pred["_merge_id"] = (
    df_pred[pred_id_col].astype(str).str.strip().str.lower()
)

# Extract columns for merge
act_cols = ["_merge_id", actual_orig_col]
if actual_stdev_col:
    act_cols.append(actual_stdev_col)

pred_cols = ["_merge_id", pred_orig_col]
if pred_stdev_col:
    pred_cols.append(pred_stdev_col)

df_act_sub = df_actual[act_cols].copy()
df_pred_sub = df_pred[pred_cols].copy()

# Rename columns explicitly
rename_act = {actual_orig_col: "y_actual_val"}
if actual_stdev_col:
    rename_act[actual_stdev_col] = "x_err_val"
df_act_sub.rename(columns=rename_act, inplace=True)

rename_pred = {pred_orig_col: "y_pred_val"}
if pred_stdev_col:
    rename_pred[pred_stdev_col] = "y_err_val"
df_pred_sub.rename(columns=rename_pred, inplace=True)

# Merge datasets on normalized ID
df_merged = pd.merge(df_act_sub, df_pred_sub, on="_merge_id", how="inner")

df_merged["y_actual_val"] = pd.to_numeric(df_merged["y_actual_val"], errors="coerce")
df_merged["y_pred_val"] = pd.to_numeric(df_merged["y_pred_val"], errors="coerce")

if "x_err_val" in df_merged.columns:
    df_merged["x_err_val"] = pd.to_numeric(df_merged["x_err_val"], errors="coerce").fillna(0)
if "y_err_val" in df_merged.columns:
    df_merged["y_err_val"] = pd.to_numeric(df_merged["y_err_val"], errors="coerce").fillna(0)

df_clean = df_merged.dropna(subset=["y_actual_val", "y_pred_val"])

# 5. Compute Metrics & Dynamic Parity Plot
y_actual = df_clean["y_actual_val"].values
y_pred = df_clean["y_pred_val"].values / pred_scale_factor

x_err = df_clean["x_err_val"].values if "x_err_val" in df_clean.columns else None
y_err = (df_clean["y_err_val"].values / pred_scale_factor) if "y_err_val" in df_clean.columns else None
print(y_err)
if len(y_actual) == 0:
    print("No matching data points found between actual and predicted files after dropping missing values.")
else:
    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    fig, ax = plt.subplots(
        figsize=PLOT_CONFIG["fig_size"], dpi=PLOT_CONFIG["display_dpi"]
    )

    # Plot Error Bars if enabled and present
    if show_error_bars and (x_err is not None or y_err is not None):
        ax.errorbar(
            y_actual,
            y_pred,
            xerr=x_err,
            yerr=y_err,
            fmt="none",
            ecolor=PLOT_CONFIG.get("errorbar_color", PLOT_CONFIG["scatter_color"]),
            alpha=PLOT_CONFIG.get("errorbar_alpha", 0.5),
            capsize=PLOT_CONFIG.get("errorbar_capsize", 2),
            linewidth=PLOT_CONFIG.get("errorbar_linewidth", 1.0),
            zorder=2,
        )

    # Scatter plot
    ax.scatter(
        y_actual,
        y_pred,
        color=PLOT_CONFIG["scatter_color"],
        alpha=PLOT_CONFIG["scatter_alpha"],
        # edgecolors="none",
        s=PLOT_CONFIG["scatter_size"],
        # zorder=3,
        label=PLOT_CONFIG["scatter_label"],
    )

    # Calculate Symmetric Bounds for a Perfect Square Axis Range
    global_min = min(y_actual.min(), y_pred.min())
    global_max = max(y_actual.max(), y_pred.max())
    padding = (global_max - global_min) * 0.6
    axis_min, axis_max = global_min - padding, global_max + padding

    # 1:1 Parity Line
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color=PLOT_CONFIG["line_color"],
        linestyle=PLOT_CONFIG["line_style"],
        linewidth=PLOT_CONFIG["line_width"],
        label=PLOT_CONFIG["line_label"],
        zorder=1,
    )

    # Annotate R^2 and RMSE
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

    # Formatting
    ax.set_xlabel(f"Actual {unit_display}", fontsize=PLOT_CONFIG["label_fontsize"])
    ax.set_ylabel(f"Predicted {unit_display}", fontsize=PLOT_CONFIG["label_fontsize"])
    ax.set_title(f"{prop_name_act}", fontsize=PLOT_CONFIG["title_fontsize"], fontweight="bold")

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=PLOT_CONFIG["grid_linestyle"], alpha=PLOT_CONFIG["grid_alpha"])
    ax.legend(
        frameon=True,
        loc=PLOT_CONFIG["legend_loc"],
        fontsize=PLOT_CONFIG["legend_fontsize"],
    )

    fig.tight_layout()

    # Save figure
    clean_filename = re.sub(r"[^\w\-]", "_", prop_name_act)
    save_path = PLOT_CONFIG["save_dir"] / f"Parity_Plot_{dataset}_{clean_filename}.png"
    fig.savefig(save_path, dpi=PLOT_CONFIG["dpi"])
    plt.close(fig)

    print(f"Successfully matched and plotted {len(y_actual)} data points.")
    print(f"Property: {prop_name_act}")
    print(f"R^2: {r2:.4f} | RMSE: {rmse:.4f} {unit_act}".strip())
    print(f"Saved parity plot to: {save_path}")