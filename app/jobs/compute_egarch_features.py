"""Recompute causal EGARCH features on app/data/final/master_dataset.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.utils.egarch_features import BURN_IN, EGARCH_COLS, compute_egarch_features


MASTER_PATH = Path("data/final/master_dataset.csv")


def main() -> None:
    print(f"Loading {MASTER_PATH} ...")
    df = pd.read_csv(MASTER_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Dataset: {len(df)} rows | {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Burn-in cutoff: {df['date'].iloc[BURN_IN - 1].date()} (obs 0..{BURN_IN - 1})\n")

    df = df.drop(columns=[col for col in EGARCH_COLS if col in df.columns], errors="ignore")
    df = compute_egarch_features(df, verbose=True)

    print("\n[EGARCH] Feature summary:")
    for col in EGARCH_COLS:
        series = df[col].dropna()
        print(f"  {col:30s} mean={series.mean():.5f} std={series.std():.5f} NaN={df[col].isna().sum()}")

    df.to_csv(MASTER_PATH, index=False)
    print(f"\nSaved -> {MASTER_PATH}")


if __name__ == "__main__":
    main()
