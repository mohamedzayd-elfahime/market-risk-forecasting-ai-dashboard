"""Forecast log schema definitions."""

from __future__ import annotations


FORECAST_LOG_COLUMNS = [
    "run_date",
    "target_date",
    "model_name",
    "horizon",
    "alpha",
    "mean_forecast",
    "volatility_forecast",
    "return_forecast",
    "var_forecast",
    "es_forecast",
    "regime",
    "weight",
    "model_version",
    "data_version",
    "realized_return",
    "var_breach",
    "es_calibration_error",
    "realized_strategy_return",
    "updated_wealth",
    "updated_drawdown",
]
