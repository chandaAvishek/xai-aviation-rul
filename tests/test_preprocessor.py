import pytest
import pandas as pd
import numpy as np
from xai_aviation_rul import preprocessor

def test_sensor_columns():
    """Return a list of columns that look like sensor_*."""
    df = pd.DataFrame({"sensor_1": [1], "sensor_2": [2], "other": [3]})
    cols = preprocessor._sensor_columns(df)
    assert cols == ["sensor_1", "sensor_2"]


def test_compute_rul():
    """Compute the remaining useful life for each row."""
    df = pd.DataFrame({"unit_number": [1, 1, 2], "time_in_cycles": [1, 2, 1]})
    out = preprocessor.compute_rul(df, rul_cap=10)
    assert "RUL" in out.columns
    assert "RUL_capped" in out.columns
    assert out.loc[0, "RUL"] == 1
    assert out.loc[0, "RUL_capped"] == 1


def test_drop_constant_sensors():
    """Drop any sensor whose standard deviation is below threshold."""
    df = pd.DataFrame({"sensor_1": [1, 1, 1], "sensor_2": [1, 2, 3]})
    out = preprocessor.drop_constant_sensors(df, threshold=0.01)
    assert "sensor_2" in out.columns
    assert "sensor_1" not in out.columns


def test_normalize():
    """Apply MinMax scaling to sensor columns."""
    train = pd.DataFrame({"sensor_1": [1, 2], "sensor_2": [2, 4]})
    test = pd.DataFrame({"sensor_1": [2, 3], "sensor_2": [4, 6]})
    train_scaled, test_scaled, scaler = preprocessor.normalize(train, test)
    assert np.allclose(train_scaled["sensor_1"], [0, 1])
    assert np.allclose(test_scaled["sensor_2"], [1, 2])


def test_get_last_cycle():
    """Return a dataframe containing only the last cycle for each unit_number."""
    df = pd.DataFrame({"unit_number": [1, 1, 2], "time_in_cycles": [1, 2, 1]})
    out = preprocessor.get_last_cycle(df)
    assert out.shape[0] == 2
    assert set(out["unit_number"]) == {1, 2}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover