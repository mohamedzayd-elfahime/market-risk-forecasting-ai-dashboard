"""Read existing backend output files for the FastAPI layer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.core.paths import (
    ECON_BACKTEST_PATH,
    FORECAST_LOG_PATH,
    MASTER_DATASET_PATH,
    PLOTS_DIR,
    REPORT_DIR,
    STAT_BACKTEST_PATH,
    TEST_PREDICTIONS_PATH,
)


LATEST_REPORT_PATH = REPORT_DIR / "latest_forecast_report.md"
LATEST_PLOT_PATH = PLOTS_DIR / "latest_price_var_es_forecast.png"


def _missing(path: Path) -> HTTPException:
    return HTTPException(status_code=404, detail=f"File not found: {path}")


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _forecast_record(row: pd.Series) -> dict[str, Any]:
    fields = [
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
    ]
    record = {field: _clean_value(row[field]) if field in row else None for field in fields}
    if record["horizon"] is not None:
        record["horizon"] = int(record["horizon"])
    return record


def read_forecast_log() -> pd.DataFrame:
    if not FORECAST_LOG_PATH.exists():
        raise _missing(FORECAST_LOG_PATH)
    df = pd.read_csv(FORECAST_LOG_PATH)
    if df.empty:
        raise HTTPException(status_code=404, detail="Forecast log is empty.")
    return df


def read_latest_forecast(horizon: int | None = None) -> dict[str, Any]:
    df = read_forecast_log()
    if horizon is not None:
        df = df[pd.to_numeric(df.get("horizon"), errors="coerce") == horizon]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No forecast found for horizon={horizon}.")
    return _forecast_record(df.iloc[-1])


def read_forecast_history(limit: int = 50, horizon: int | None = None) -> list[dict[str, Any]]:
    df = read_forecast_log()
    if horizon is not None:
        df = df[pd.to_numeric(df.get("horizon"), errors="coerce") == horizon]
    df = df.tail(limit)
    return [_forecast_record(row) for _, row in df.iterrows()]


def read_price_series(history_limit: int = 260) -> dict[str, Any]:
    if not MASTER_DATASET_PATH.exists():
        raise _missing(MASTER_DATASET_PATH)

    master = pd.read_csv(MASTER_DATASET_PATH)
    required = {"date", "masi_close"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise HTTPException(status_code=500, detail=f"Master dataset missing columns: {missing}")

    master["date"] = pd.to_datetime(master["date"], errors="coerce", format="mixed")
    master["masi_close"] = pd.to_numeric(master["masi_close"], errors="coerce")
    master = master.dropna(subset=["date", "masi_close"]).sort_values("date")
    if master.empty:
        raise HTTPException(status_code=404, detail="No valid observed MASI prices found.")

    observed_df = master.tail(history_limit)
    last_price = float(master["masi_close"].iloc[-1])
    last_date = master["date"].iloc[-1]

    forecast_log = read_forecast_log()
    latest_run = forecast_log["run_date"].iloc[-1]
    latest_forecasts = forecast_log[forecast_log["run_date"] == latest_run].copy()
    latest_forecasts["horizon"] = pd.to_numeric(latest_forecasts["horizon"], errors="coerce")
    latest_forecasts = latest_forecasts.sort_values("horizon")

    forecast_points = []
    for _, row in latest_forecasts.iterrows():
        return_forecast = _clean_value(row.get("return_forecast"))
        var_forecast = _clean_value(row.get("var_forecast"))
        es_forecast = _clean_value(row.get("es_forecast"))
        forecast_points.append(
            {
                "target_date": str(_clean_value(row.get("target_date"))),
                "horizon": int(row["horizon"]),
                "mean_forecast": _clean_value(row.get("mean_forecast")),
                "volatility_forecast": _clean_value(row.get("volatility_forecast")),
                "return_forecast": return_forecast,
                "var_forecast": var_forecast,
                "es_forecast": es_forecast,
                "price_forecast_proxy": float(last_price * np.exp(return_forecast)) if return_forecast is not None else None,
                "var_price_proxy": float(last_price * np.exp(var_forecast)) if var_forecast is not None else None,
                "es_price_proxy": float(last_price * np.exp(es_forecast)) if es_forecast is not None else None,
            }
        )

    return {
        "last_observed_date": last_date.date().isoformat(),
        "last_observed_price": last_price,
        "observed": [
            {"date": row["date"].date().isoformat(), "close": float(row["masi_close"])}
            for _, row in observed_df.iterrows()
        ],
        "forecasts": forecast_points,
    }


def read_risk_series(history_limit: int = 180) -> dict[str, Any]:
    if not MASTER_DATASET_PATH.exists():
        raise _missing(MASTER_DATASET_PATH)

    master = pd.read_csv(MASTER_DATASET_PATH)
    required = {"date", "masi_egarch_mean", "masi_egarch_vol"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise HTTPException(status_code=500, detail=f"Master dataset missing risk columns: {missing}")

    master["date"] = pd.to_datetime(master["date"], errors="coerce", format="mixed")
    master["masi_egarch_mean"] = pd.to_numeric(master["masi_egarch_mean"], errors="coerce")
    master["masi_egarch_vol"] = pd.to_numeric(master["masi_egarch_vol"], errors="coerce")
    master = master.dropna(subset=["date"]).sort_values("date")
    observed_df = master.tail(history_limit)
    last_date = master["date"].iloc[-1]

    forecast_log = read_forecast_log()
    latest_run = forecast_log["run_date"].iloc[-1]
    latest_forecasts = forecast_log[forecast_log["run_date"] == latest_run].copy()
    latest_forecasts["horizon"] = pd.to_numeric(latest_forecasts["horizon"], errors="coerce")
    latest_forecasts = latest_forecasts.sort_values("horizon")

    forecast_points = []
    for _, row in latest_forecasts.iterrows():
        forecast_points.append(
            {
                "target_date": str(_clean_value(row.get("target_date"))),
                "horizon": int(row["horizon"]),
                "mean_forecast": _clean_value(row.get("mean_forecast")),
                "volatility_forecast": _clean_value(row.get("volatility_forecast")),
            }
        )

    return {
        "last_observed_date": last_date.date().isoformat(),
        "observed": [
            {
                "date": row["date"].date().isoformat(),
                "mean": _clean_value(row.get("masi_egarch_mean")),
                "volatility": _clean_value(row.get("masi_egarch_vol")),
            }
            for _, row in observed_df.iterrows()
        ],
        "forecasts": forecast_points,
    }


def read_latest_report() -> dict[str, str]:
    if not LATEST_REPORT_PATH.exists():
        raise _missing(LATEST_REPORT_PATH)
    return {
        "report_name": LATEST_REPORT_PATH.name,
        "content": LATEST_REPORT_PATH.read_text(encoding="utf-8"),
        "path": str(LATEST_REPORT_PATH),
    }


def latest_plot_path() -> Path:
    if not LATEST_PLOT_PATH.exists():
        raise _missing(LATEST_PLOT_PATH)
    return LATEST_PLOT_PATH


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise _missing(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_backtest_summary() -> dict[str, Any]:
    statistical = _read_json(STAT_BACKTEST_PATH)
    economic = _read_json(ECON_BACKTEST_PATH)
    return {
        "violation_rate": statistical.get("violation_rate"),
        "expected_violation_rate": statistical.get("expected_violation_rate"),
        "n_observations": statistical.get("n_observations"),
        "n_violations": statistical.get("n_var_breaches"),
        "es_tail_residual_mean": statistical.get("es_tail_residual_mean"),
        "final_wealth": economic.get("final_wealth"),
        "buy_hold_final_wealth": economic.get("buy_hold_final_wealth"),
        "max_drawdown": economic.get("max_drawdown"),
        "current_drawdown": economic.get("max_drawdown"),
        "buy_hold_max_drawdown": economic.get("buy_hold_max_drawdown"),
        "annualized_sharpe": economic.get("annualized_sharpe"),
        "buy_hold_sharpe": economic.get("buy_hold_sharpe"),
        "statistical": statistical,
        "economic": economic,
        "source_files": {
            "statistical": str(STAT_BACKTEST_PATH),
            "economic": str(ECON_BACKTEST_PATH),
        },
    }


def read_test_predictions(limit: int = 250) -> list[dict[str, Any]]:
    if not TEST_PREDICTIONS_PATH.exists():
        raise _missing(TEST_PREDICTIONS_PATH)
    df = pd.read_csv(TEST_PREDICTIONS_PATH)
    if df.empty:
        return []

    fields = [
        "date",
        "masi_close",
        "target_return",
        "realized_return",
        "return_pred",
        "var_pred",
        "es_pred",
        "predicted_shortfall",
        "violation",
        "buy_hold_wealth",
        "strategy_wealth",
        "buy_hold_drawdown",
        "strategy_drawdown",
        "strategy_weight",
    ]
    df = df.tail(limit)
    rows = []
    for _, row in df.iterrows():
        record = {field: _clean_value(row[field]) if field in row else None for field in fields}
        if record["violation"] is not None:
            record["violation"] = int(record["violation"])
        rows.append(record)
    return rows
