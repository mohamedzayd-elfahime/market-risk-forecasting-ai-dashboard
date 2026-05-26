"""Data loading and validation for the MASI master dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


REQUIRED_MASTER_COLUMNS = {
    "date",
    "masi_close",
    "masi_log_return",
    "masi_log_return_std_5",
    "masi_log_return_std_21",
    "masi_log_return_mean_21",
    # EGARCH-LSTM hybrid features — computed by jobs/compute_egarch_features.py
    "masi_egarch_vol",
    "masi_egarch_std_resid",
    "masi_egarch_vol_ratio",
    "masi_egarch_vol_next",
}


def dataset_version(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def load_master_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Master dataset not found: {path}")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_MASTER_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required MASI master columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    numeric_cols = [col for col in df.columns if col != "date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)
