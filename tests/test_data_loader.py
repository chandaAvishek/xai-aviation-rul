import pytest
import pandas as pd
from xai_aviation_rul import data_loader

def test_column_names():
    """Returns the column names for the dataset."""
    cols = data_loader._column_names()
    assert isinstance(cols, list)
    assert "unit_number" in cols
    assert len(cols) == 26


def test_load_cmapss(tmp_path):
    """Loads the CMAPSS dataset."""
    # Create a dummy file
    file = tmp_path / "train_FD001.txt"
    file.write_text("1 1 0.0 0.0 0.0 " + " ".join(["100"]*21) + "\n")
    df = data_loader.load_cmapss("train", 1, tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 26
    assert df.iloc[0]["unit_number"] == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover