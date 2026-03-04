import pytest
import pandas as pd
import numpy as np
from xai_aviation_rul import visualizer
from pathlib import Path

def test_plot_rul_distribution(tmp_path):
    """Plot the distribution of engine lifespans with RUL cap reference line."""
    df = pd.DataFrame({"unit_number": [1, 2], "RUL": [100, 120]})
    out_path = tmp_path / "rul_dist.png"
    visualizer.plot_rul_distribution(df, save_path=out_path)
    assert out_path.exists()


def test_plot_sensor_trends(tmp_path):
    """Plot time-series of sensor readings for selected engines in a grid."""
    df = pd.DataFrame({"unit_number": [1]*5, "time_in_cycles": range(5), "sensor_1": np.arange(5)})
    out_path = tmp_path / "sensor_trends.png"
    visualizer.plot_sensor_trends(df, [1], ["sensor_1"], save_path=out_path)
    assert out_path.exists()


def test_plot_sensor_variance(tmp_path):
    """Bar chart of the standard deviation of each sensor column."""
    df = pd.DataFrame({"sensor_1": [1, 2, 3], "sensor_2": [2, 2, 2]})
    out_path = tmp_path / "sensor_var.png"
    visualizer.plot_sensor_variance(df, save_path=out_path)
    assert out_path.exists()


def test_plot_correlation_heatmap(tmp_path):
    """Heatmap of correlations between sensor and RUL columns."""
    df = pd.DataFrame({"sensor_1": [1, 2, 3], "sensor_2": [2, 3, 4], "RUL_capped": [10, 20, 30]})
    out_path = tmp_path / "corr_heatmap.png"
    visualizer.plot_correlation_heatmap(df, save_path=out_path)
    assert out_path.exists()


def test_save_table_as_figure(tmp_path):
    """Render a DataFrame as a styled matplotlib table and save it as an image."""
    df = pd.DataFrame({"Metric": ["A", "B"], "Value": [1, 2]})
    out_path = tmp_path / "table.png"
    visualizer.save_table_as_figure(df, save_path=out_path)
    assert out_path.exists()
