# XAI for Aviation RUL Prediction

This project implements an explainable AI pipeline for Remaining Useful Life (RUL) prediction of turbofan aircraft engines using the NASA C-MAPSS dataset. The system trains and compares three regression models — Linear Regression, Random Forest, and XGBoost — and applies SHAP (SHapley Additive exPlanations) to interpret the predictions at both global and local levels.

The work is developed as part of an M.Sc. research report at Technische Universität Darmstadt.

## Dataset

This project uses the **FD001 subset** of the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. The FD001 subset covers a single operating condition and a single fault mode (High Pressure Compressor degradation), making it well-suited for targeted RUL regression.

Download the dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place the files in `data/CMAPSS_dataset/`. 

## Notebooks

### `notebooks/data-exploration.ipynb`

Performs full EDA on the FD001 training and test sets.

All figures generated here are saved to the `figures/` directory and used in the final report.

### `notebooks/model_training.ipynb`

Trains and evaluates three regression models for RUL prediction.

### `notebooks/xai_analysis.ipynb`

Applies SHAP explainability to the trained XGBoost model.

## Source Modules

### `src/xai_aviation_rul/data_loader.py`

Handles loading of C-MAPSS dataset files. The `load_cmapss` function accepts a subset name (`train` or `test`), a fault dataset index (`fd=1`), and a base path. It tries multiple filename patterns and directory locations, assigns the standard column names, and returns a `pandas.DataFrame`.

### `src/xai_aviation_rul/preprocessor.py`

Contains all data preprocessing logic:

- `compute_rul` — calculates the RUL for each engine cycle and applies a configurable cap (default 125 cycles)
- `drop_constant_sensors` — removes sensor columns whose standard deviation falls below a threshold, eliminating non-informative features
- `normalize` — fits a `MinMaxScaler` on the training sensor columns and applies it to both training and test sets, preventing data leakage
- `get_last_cycle` — extracts the final recorded observation per engine, used for test-set evaluation

### `src/xai_aviation_rul/models.py`

Provides training, evaluation, and persistence functions for all three models:

- `train_linear_regression`, `train_random_forest`, `train_xgboost` — each accepts training data and returns a fitted model; XGBoost uses 200 estimators with a learning rate of 0.05 and max depth of 6
- `evaluate_model` — computes RMSE, MAE, and R-squared for any fitted model on a given test set
- `save_model` / `load_model` — serialises and loads models using `joblib`

### `src/xai_aviation_rul/explainer.py`

Wraps SHAP functionality for tree-based models:

- `compute_shap_values` — initialises a `shap.TreeExplainer` on the model and computes SHAP values for a given input matrix; returns both the values array and the explainer object
- `plot_shap_summary` — generates a dot-style SHAP summary plot showing global feature importance across all instances
- `plot_shap_waterfall` — generates a waterfall plot for a single engine instance, showing how each sensor shifts the prediction from the baseline expected value

### `src/xai_aviation_rul/visualizer.py`

Contains all general-purpose visualisation functions used across the notebooks:

- `plot_rul_distribution` — histogram of engine lifespans with a reference line at the RUL cap
- `plot_sensor_trends` — time-series grid of selected sensor readings for selected engines
- `plot_sensor_variance` — bar chart of sensor standard deviations with a threshold reference line
- `plot_correlation_heatmap` — heatmap of Pearson correlations between sensors and RUL
- `save_table_as_figure` — renders a key-value `DataFrame` as a styled matplotlib table image
- `save_results_table` — renders a model comparison `DataFrame` as a styled matplotlib table image

---

## Installation

This project uses `pyproject.toml` for dependency management and is installable as a local package.

**Requirements:** Python 3.10 or higher.

### Step 1 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### Step 2 — Install the package and its dependencies

```bash
pip install -e .
```

This installs the `xai_aviation_rul` package in editable mode along with all required dependencies:

| Package | Minimum Version |
|---|---|
| numpy | 1.24 |
| pandas | 2.0 |
| scikit-learn | 1.3 |
| matplotlib | 3.7 |
| seaborn | 0.12 |
| shap | 0.44 |
| xgboost | 2.0 |
| scipy | 1.11 |
| joblib | 1.3 |

### Step 3 — Install development dependencies (optional)

To also install testing, linting, and Jupyter tools:

```bash
pip install -e ".[dev]"
```

This adds `pytest`, `pytest-cov`, `black`, `ruff`, `jupyter`, and `ipykernel`.

### Step 4 — Register the Jupyter kernel

```bash
python -m ipykernel install --user --name xai-aviation-rul
```

### Step 5 — Launch Jupyter and run the notebooks in order

```bash
jupyter notebook
```

Run the notebooks in the following order:

1. `notebooks/data-exploration.ipynb`
2. `notebooks/model_training.ipynb`
3. `notebooks/xai_analysis.ipynb`

The training notebook saves the fitted XGBoost model to `models/`. The XAI notebook loads this saved model, so notebook `models_training.ipynb` must be run before notebook `xai_analysis.ipynb`.

---