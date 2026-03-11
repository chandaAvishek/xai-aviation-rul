# standard
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

# 3rd party
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def train_linear_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    """Train a simple linear regression model on the data."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a random forest model with multiple decision trees."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> XGBRegressor:
    """Train an XGBoost gradient boosting model."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calculate RMSE, MAE, and R-squared metrics on test data."""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
    }

def save_model(model: Any, path: str | Path) -> None:
    """Save a trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    """Load a trained model from disk."""
    return joblib.load(path)
