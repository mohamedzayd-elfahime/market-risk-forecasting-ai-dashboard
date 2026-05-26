"""Report service helpers."""

from __future__ import annotations

import json

import pandas as pd

from backend.core.paths import FORECAST_LOG_PATH, METADATA_PATH, REPORT_DIR
from ml.utils.reporting import write_forecast_report


def regenerate_latest_report():
    log = pd.read_csv(FORECAST_LOG_PATH)
    latest_run = log["run_date"].max()
    latest = log[log["run_date"] == latest_run].copy()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    return write_forecast_report(REPORT_DIR / "latest_forecast_report.md", latest, metadata)
