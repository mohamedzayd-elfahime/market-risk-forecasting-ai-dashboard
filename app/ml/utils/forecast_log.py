"""Forecast log storage and enrichment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.schemas.forecast_schema import FORECAST_LOG_COLUMNS


def normalize_forecast_log(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FORECAST_LOG_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[FORECAST_LOG_COLUMNS]


def append_forecasts(log_path: Path, forecasts: pd.DataFrame) -> pd.DataFrame:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    incoming = normalize_forecast_log(forecasts)

    if log_path.exists():
        existing = normalize_forecast_log(pd.read_csv(log_path))
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    combined = combined.drop_duplicates(
        subset=["run_date", "target_date", "model_name", "horizon", "alpha"],
        keep="last",
    )
    combined.to_csv(log_path, index=False)
    return combined


def enrich_realizations(log_path: Path, master_df: pd.DataFrame, return_col: str = "masi_log_return") -> pd.DataFrame:
    log = normalize_forecast_log(pd.read_csv(log_path))
    realized = master_df[["date", return_col]].copy()
    realized["date"] = pd.to_datetime(realized["date"], errors="coerce", format="mixed")
    realized = realized.rename(columns={"date": "target_date", return_col: "realized_return_new"})

    log["target_date"] = pd.to_datetime(log["target_date"], errors="coerce", format="mixed")
    enriched = log.merge(realized, on="target_date", how="left")
    mask = enriched["realized_return"].isna() & enriched["realized_return_new"].notna()
    enriched.loc[mask, "realized_return"] = enriched.loc[mask, "realized_return_new"]
    enriched = enriched.drop(columns=["realized_return_new"])
    enriched.to_csv(log_path, index=False)
    return enriched
