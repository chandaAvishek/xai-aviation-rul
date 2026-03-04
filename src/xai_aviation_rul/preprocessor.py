# Standard
from __future__ import annotations
from typing import Tuple

# 3rd-party
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    """ Return a list of columns that look like sensor_*."""
    return [c for c in df.columns if c.startswith("sensor_")]


def compute_rul(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """ Compute the remaining useful life for each row. """
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df = df.copy()
    df["RUL"] = max_cycle - df["time_in_cycles"]
    df["RUL_capped"] = df["RUL"].clip(upper=rul_cap)
    return df


def drop_constant_sensors(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """ Drop any sensor whose standard deviation is below `threshold`.A new frame with low‑variance sensors removed. """
    sensors = _sensor_columns(df)
    stds = df[sensors].std()
    keep = stds[stds >= threshold].index.tolist()
    return df.drop(columns=[c for c in sensors if c not in keep])


def normalize(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """ Apply MinMax scaling to sensor columns.  """
    sensors = _sensor_columns(train_df)
    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[sensors] = scaler.fit_transform(train_df[sensors])
    test_scaled[sensors] = scaler.transform(test_df[sensors])

    return train_scaled, test_scaled, scaler


def get_last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """ Return a dataframe containing only the last cycle for each `unit_number`. """
    idx = df.groupby("unit_number")["time_in_cycles"].idxmax()
    return df.loc[idx].reset_index(drop=True)