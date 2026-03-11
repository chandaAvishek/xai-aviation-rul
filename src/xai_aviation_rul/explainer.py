# standard
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 3rd party
import pandas as pd
import shap


def compute_shap_values(
    model, X: pd.DataFrame | np.ndarray
) -> tuple[np.ndarray, shap.TreeExplainer]:
    """Calculate SHAP values to explain model predictions.

    Returns values and explainer object.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap_values, explainer

def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """Draw a summary plot showing which features are most important
    overall."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="dot", show=False)
    plt.title(
        "SHAP Summary Plot — Global Feature Importance",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    plt.show()

def plot_shap_waterfall(
        explainer: shap.TreeExplainer,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        engine_idx: int,
        save_path: str | Path | None = None,
) -> None:
    """Draw a waterfall plot showing how each sensor value influences one prediction."""

    # extract base value as scalar (handle both scalar and array cases)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value.item() if base_value.size == 1 else float(base_value[0])
    else:
        base_value = float(base_value)

    # create SHAP explanation object for the specific instance
    explanation = shap.Explanation(
        values=shap_values[engine_idx],
        base_values=base_value,
        data=X.iloc[engine_idx].to_numpy(),
        feature_names=X.columns.tolist(),
    )

    # create waterfall plot
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title(
        f"SHAP Waterfall Plot — Engine {engine_idx} RUL Prediction Explanation",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    plt.show()