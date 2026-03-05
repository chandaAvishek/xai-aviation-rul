# Standard
from __future__ import annotations
from pathlib import Path
from typing import Any

# 3rd party
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """ Train a baseline Linear Regression model. """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestRegressor:
    """ Train a Random Forest Regressor model. """
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
    """ Train an XGBoost Regressor model. """
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
    """ Evaluate a model and return RMSE, MAE, and R² scores. """
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
    """ Save a model using joblib. """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    """ Load a model using joblib. """
    return joblib.load(path)
