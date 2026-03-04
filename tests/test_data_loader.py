import pytest
import pandas as pd
from xai_aviation_rul import data_loader

def test_column_names():
    """Returns the column names for the dataset."""
    cols = data_loader._column_names()
    assert isinstance(cols, list)
    assert "unit_number" in cols
    assert len(cols) == 26


def test_load_cmapps(tmp_path):
    """Loads the CMAPPS dataset."""
    # Create a dummy file
    file = tmp_path / "train_FD001.txt"
    file.write_text("1 1 0.0 0.0 0.0 " + " ".join(["100"]*21) + "\n")
    df = data_loader.load_cmapps("train", 1, tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 26
    assert df.iloc[0]["unit_number"] == 1
