# standard
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _column_names() -> list[str]:
    """Returns the column names for the dataset."""
    return (
        ["unit_number", "time_in_cycles"]
        + [f"op-setting_{i+1}" for i in range(3)]
        + [f"sensor_{i+1}" for i in range(21)]
    )

def load_cmapss(subset: str = "train", fd: int = 1, path: str | Path = "data/CMAPSS_dataset") -> pd.DataFrame:
    """Loads the CMAPSS dataset."""
    path = Path(path)
    repo_root = Path(__file__).resolve().parents[2]
    base_candidates: list[Path]
    if path.is_absolute():
        base_candidates = [path.resolve()]
    else:
        base_candidates = [(Path.cwd() / path).resolve(), (repo_root / path).resolve()]

    # build filename candidates for each base and pick the first existing file
    candidates: list[Path] = []
    for base in base_candidates:
        candidates.extend([
            base / f"{subset}_FD00{fd}.txt",
            base / f"FD00{fd}_{subset}.txt",
            base / f"{subset}_FD{fd:03d}.txt",
            base / f"FD{fd:03d}_{subset}.txt",
        ])

    existing = [p for p in candidates if p.exists()]
    if not existing:
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"None of the expected CMAPSS files were found.\nTried:\n{tried}"
        )

    filename = existing[0]
    cols = _column_names()
    df = pd.read_csv(filename, sep=r"\s+", header=None, names=cols)
    return df