import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==========================================
# USER CONFIGURATION
# ==========================================
# 1. File Path
csv_log_path = Path("data/output/ALL3_RSME000223_v2/history_60000.csv")


# 2. Log Scale Toggle
use_log_scale = True  # Set to True for log scale on y-axis, False for linear scale

# 3. Task Definitions (Map task column index to display name, unit, and unique color)
TASK_METRICS = {
    0: {"name": "Entropy of Fusion", "unit": "kJ/(mol K)", "color": "#8E6F3E"},
    1: {"name": "Enthalpy of Fusion", "unit": "kJ/mol", "color": "#DDB945"},
    2: {"name": "Enthalpy of Formation", "unit": "kJ/mol", "color": "#9D9795"},
}

selected_epochs = [50000]

# 4. Global Plot Formatting Options
PLOT_CONFIG = {
    "save_dir": Path("data/output/ALL3_RSME000223_v2"),
    "fig_size": (3, 3),
    "dpi": 300,
    "display_dpi": 120,
    # Default colors for loss plot
    "loss_train_color": "#000000",
    "loss_test_color": "#CFB991",
    # Line styles
    "train_style": "-",
    "train_label": "Training",
    "test_style": "--",
    "test_label": "Testing",
    "line_width": 2.5,
    # Vertical line style for selected epoch(s)
    "epoch_line_color": "#000000",
    "epoch_line_style": ":",
    "epoch_line_width": 1.5,
    "epoch_line_alpha": 0.8,
    # Font Sizes & Axes
    "label_fontsize": 11,
    "title_fontsize": 12,
    "legend_fontsize": 8,
    "grid_linestyle": "--",
    "grid_alpha": 0.5,
    "legend_loc": "upper right",
}

PLOT_CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
# ==========================================

def add_epoch_vertical_lines(ax):
    """Adds vertical lines for user-specified selected epochs and returns legend labels."""
    if not selected_epochs:
        return

    for idx, ep in enumerate(selected_epochs):
        # Label only the first line as "Selected Epoch" (or list specific values if multiple)
        label = (
            "Selected Epoch"
        )

        ax.axvline(
            x=ep,
            color=PLOT_CONFIG["epoch_line_color"],
            linestyle=PLOT_CONFIG["epoch_line_style"],
            linewidth=PLOT_CONFIG["epoch_line_width"],
            alpha=PLOT_CONFIG["epoch_line_alpha"],
            label=label,
            zorder=1,
        )

# 1. Load Training Log CSV
print(f"Loading training log: {csv_log_path}")
df_log = pd.read_csv(csv_log_path)
df_log.columns = df_log.columns.str.strip()

epochs = df_log["epoch"].values

# 2. Plot Overall Losses (train_loss vs test_loss)
if "train_loss" in df_log.columns and "test_loss" in df_log.columns:
    fig, ax = plt.subplots(
        figsize=PLOT_CONFIG["fig_size"], dpi=PLOT_CONFIG["display_dpi"]
    )

    ax.plot(
        epochs,
        df_log["train_loss"],
        color=PLOT_CONFIG["loss_train_color"],
        linestyle=PLOT_CONFIG["train_style"],
        linewidth=PLOT_CONFIG["line_width"],
        label=f"{PLOT_CONFIG['train_label']}",
    )
    ax.plot(
        epochs,
        df_log["test_loss"],
        color=PLOT_CONFIG["loss_test_color"],
        linestyle=PLOT_CONFIG["test_style"],
        linewidth=PLOT_CONFIG["line_width"],
        label=f"{PLOT_CONFIG['test_label']}",
    )
    add_epoch_vertical_lines(ax)

    if use_log_scale:
        ax.set_yscale("log")
    

    ax.set_xlabel("Epoch [-]", fontsize=PLOT_CONFIG["label_fontsize"])
    ax.set_ylabel("Negative Log Likelihood Loss [-]", fontsize=PLOT_CONFIG["label_fontsize"])
    ax.set_title(
        "Model Loss",
        fontsize=PLOT_CONFIG["title_fontsize"],
        fontweight="bold",
    )
    ax.grid(
        True,
        which="both" if use_log_scale else "major",
        linestyle=PLOT_CONFIG["grid_linestyle"],
        alpha=PLOT_CONFIG["grid_alpha"],
    )
    ax.legend(
        framealpha=1.0,
        frameon=True,
        loc="upper right",
        fontsize=PLOT_CONFIG["legend_fontsize"],
    )

    fig.tight_layout()
    loss_save_path = PLOT_CONFIG["save_dir"] / "Training_Testing_Loss.png"
    fig.savefig(loss_save_path, dpi=PLOT_CONFIG["dpi"])
    plt.close(fig)
    print(f"Saved loss plot: {loss_save_path}")

# 3. Plot Per-Task RMSE (train_rmse_Task_X vs test_rmse_Task_X)
for task_id, info in TASK_METRICS.items():
    train_col = f"train_rmse_Task_{task_id}"
    test_col = f"test_rmse_Task_{task_id}"

    # Check if train column exists
    if train_col not in df_log.columns or df_log[train_col].dropna().empty:
        print(f"Skipping Task {task_id}: No training data found in column '{train_col}'.")
        continue

    fig, ax = plt.subplots(
        figsize=PLOT_CONFIG["fig_size"], dpi=PLOT_CONFIG["display_dpi"]
    )

    task_color = info.get("color", "#1f77b4")

    # Plot Training (Solid Line, Task Color)
    ax.plot(
        epochs,
        df_log[train_col],
        color=task_color,
        linestyle=PLOT_CONFIG["train_style"],
        linewidth=PLOT_CONFIG["line_width"],
        label=PLOT_CONFIG["train_label"],
    )

    # Check if test column exists AND has non-NaN values
    has_test_data = (
        test_col in df_log.columns 
        and not df_log[test_col].dropna().empty
    )

    # Only plot testing if valid test data exists
    if has_test_data:
        ax.plot(
            epochs,
            df_log[test_col],
            color=task_color,
            linestyle=PLOT_CONFIG["test_style"],
            linewidth=PLOT_CONFIG["line_width"],
            label=PLOT_CONFIG["test_label"],
        )
    add_epoch_vertical_lines(ax)
    if use_log_scale:
        ax.set_yscale("log")

    y_vals = df_log[train_col].dropna().values
    if has_test_data:
        y_vals = np.concatenate([y_vals, df_log[test_col].dropna().values])

    y_min, y_max = y_vals.min(), y_vals.max()

    if use_log_scale:
        # Multiplicative margin in log space
        pad_factor = 1.5  # 20% expansion on log limits
        ax.set_ylim(y_min / pad_factor, y_max * pad_factor)

    unit_str = f" [{info['unit']}]" if info["unit"] else ""

    ax.set_xlabel("Epoch [-]", fontsize=PLOT_CONFIG["label_fontsize"])
    ax.set_ylabel(f"RMSE{unit_str}", fontsize=PLOT_CONFIG["label_fontsize"])
    ax.set_title(
        f"{info['name']}",
        fontsize=PLOT_CONFIG["title_fontsize"],
        fontweight="bold",
    )
    ax.grid(
        True,
        which="both" if use_log_scale else "major",
        linestyle=PLOT_CONFIG["grid_linestyle"],
        alpha=PLOT_CONFIG["grid_alpha"],
    )
    
    # Legend automatically displays only labeled lines
    ax.legend(
        framealpha=1.0,
        frameon=True,
        loc=PLOT_CONFIG["legend_loc"],
        fontsize=PLOT_CONFIG["legend_fontsize"],
    )

    if task_id == 0:
        ax.set_ylim(5e-5, 5e-1)
    elif task_id == 1:
        ax.set_ylim(1e-2, 200)
    elif task_id == 2:
        ax.set_ylim(0.0008, 5000)

    fig.tight_layout()

    # Save sanitized per-task plot
    clean_task_name = re.sub(r"[^\w\-]", "_", info["name"])
    task_save_path = (
        PLOT_CONFIG["save_dir"] / f"Task_{task_id}_{clean_task_name}_RMSE.png"
    )
    fig.savefig(task_save_path, dpi=PLOT_CONFIG["dpi"])
    plt.close(fig)
    print(f"Saved Task {task_id} plot: {task_save_path}")