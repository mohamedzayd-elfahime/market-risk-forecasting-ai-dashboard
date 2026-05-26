"""Backtesting service based on the forecast log."""

from __future__ import annotations

import pandas as pd

from backend.core.config import DEFAULT_CONFIG, ForecastConfig
from backend.core.paths import FORECAST_LOG_PATH, MASTER_DATASET_PATH
from ml.utils.backtesting import enrich_backtest_columns, summarize_statistical_backtest
from ml.utils.data_loading import load_master_dataset
from ml.utils.forecast_log import enrich_realizations


def run_progressive_backtest(config: ForecastConfig = DEFAULT_CONFIG) -> dict[str, object]:
    master_df = load_master_dataset(MASTER_DATASET_PATH)
    enriched = enrich_realizations(FORECAST_LOG_PATH, master_df, return_col=config.return_col)
    enriched = enrich_backtest_columns(enriched)
    enriched.to_csv(FORECAST_LOG_PATH, index=False)
    summary = summarize_statistical_backtest(enriched, alpha=config.alpha)
    return {"forecast_log": enriched, "summary": summary}
