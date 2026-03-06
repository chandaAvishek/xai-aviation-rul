# standard
from __future__ import annotations

from pathlib import Path

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# configure seaborn globally
sns.set(style="whitegrid")


def plot_rul_distribution(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """Plot the distribution of engine lifespans with RUL cap reference line."""
    # get RUL column name
    col = "RUL" if "RUL" in df.columns else "RUL_capped"
    # calculate max RUL per engine
    max_rul_per_engine = df.groupby("unit_number")[col].max()
    plt.figure(figsize=(8, 5))
    plt.hist(max_rul_per_engine, bins=20, edgecolor="black")
    plt.axvline(125, color="red", linestyle="--", linewidth=1.5, label="RUL cap = 125")
    plt.title("Distribution of Engine Lifespans")
    plt.xlabel("Maximum RUL per Engine")
    plt.ylabel("Number of Engines")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_sensor_trends(
    df: pd.DataFrame,
    engine_ids: list[int],
    sensor_cols: list[str],
    save_path: str | Path | None = None,
) -> None:
    """Plot time-series of sensor readings for selected engines in a grid."""
    n_engines = len(engine_ids)
    n_sensors = len(sensor_cols)
    fig, axes = plt.subplots(
        n_sensors, n_engines, figsize=(5 * n_engines, 3 * n_sensors)
    )

    # ensure axes is always 2d
    if n_sensors == 1 and n_engines == 1:
        axes = [[axes]]
    elif n_sensors == 1:
        axes = [axes]
    elif n_engines == 1:
        axes = [[ax] for ax in axes]

    for col_idx, uid in enumerate(engine_ids):
        sub = df[df["unit_number"] == uid]
        for row_idx, col in enumerate(sensor_cols):
            ax = axes[row_idx][col_idx]
            ax.plot(sub["time_in_cycles"].values, sub[col].values, linewidth=1)
            ax.set_title(f"Engine {uid} — {col}", fontsize=9)
            ax.set_xlabel("Cycle", fontsize=8)
            ax.set_ylabel(col, fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_sensor_variance(
    df: pd.DataFrame,
    threshold: float = 0.01,
    save_path: str | Path | None = None,
) -> None:
    """Bar chart of the standard deviation of each sensor column."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    std_vals = df[sensor_cols].std().sort_values()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(std_vals)), np.asarray(std_vals.values))
    ax.set_xticks(range(len(std_vals)))
    ax.set_xticklabels(std_vals.index.tolist(), rotation=45, ha="right")
    ax.set_title("Standard Deviation of Each Sensor")
    ax.set_ylabel("Std Dev")
    ax.set_xlabel("Sensor")

    # determine x-position of threshold line
    x_pos = len(std_vals) - 0.5
    for idx, val in enumerate(std_vals.values):
        if val >= threshold:
            x_pos = idx
            break
    ax.axvline(
        x_pos,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"threshold={threshold}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """Heatmap of correlations between sensor and RUL columns."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    cols_to_plot = sensor_cols + ["RUL_capped"]
    corr = df[cols_to_plot].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Correlation Matrix: Sensors & RUL")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def save_table_as_figure(
    df: pd.DataFrame,
    save_path: str | Path,
    figsize: tuple = (6, 4),
    fontsize: int = 11,
    scale: tuple = (1.5, 1.8),
    title: str | None = None,
) -> None:
    """Render a DataFrame as a styled matplotlib table and save it as an image."""
    col_labels = ["Metric", "Value"]
    cell_data = [
        [str(idx), str(int(val)) if float(val).is_integer() else str(val)]
        for idx, val in zip(df.index, df["Value"])
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    if title:
        ax.set_title(
            title,
            fontsize=fontsize + 2,
            fontweight="bold",
            pad=15,
            color="#2C3E50",
        )

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(*scale)

    # style header row
    for col in range(2):
        cell = table[0, col]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#2C3E50")

    # style data rows with alternating colours
    for row in range(1, len(cell_data) + 1):
        for col in range(2):
            cell = table[row, col]
            cell.set_facecolor("#EAF0FB" if row % 2 == 0 else "white")
            cell.set_edgecolor("#D5D8DC")
            cell.set_text_props(color="#2C3E50")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


def save_results_table(
    df: pd.DataFrame,
    save_path: str | Path,
    figsize: tuple = (8, 3),
    fontsize: int = 11,
    scale: tuple = (1.5, 1.8),
    title: str | None = None,
) -> None:
    """Render a model results DataFrame as a styled table.

    Index=model names, columns=metrics.
    """
    col_labels = ["Model"] + list(df.columns)
    cell_data = [
        [str(idx)] + [f"{v:.3f}" for v in row]
        for idx, row in zip(df.index, df.values)
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    if title:
        ax.set_title(
            title,
            fontsize=fontsize + 2,
            fontweight="bold",
            pad=15,
            color="#2C3E50",
        )

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(*scale)

    for col in range(len(col_labels)):
        cell = table[0, col]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#2C3E50")

    for row in range(1, len(cell_data) + 1):
        for col in range(len(col_labels)):
            cell = table[row, col]
            cell.set_facecolor("#EAF0FB" if row % 2 == 0 else "white")
            cell.set_edgecolor("#D5D8DC")
            cell.set_text_props(color="#2C3E50")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()