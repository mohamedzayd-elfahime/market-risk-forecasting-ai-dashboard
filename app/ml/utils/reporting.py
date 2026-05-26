"""Minimal forecast reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_forecast_report(report_path: Path, forecast_df: pd.DataFrame, metadata: dict, plot_path: Path | None = None) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    display_df = forecast_df.drop(columns=["weight"], errors="ignore")

    lines = [
        "# MASI Forecast Report",
        "",
        f"Model: {metadata.get('model_name', 'unknown')}",
        f"Version: {metadata.get('model_version', 'unknown')}",
        f"Alpha: {metadata.get('alpha', 0.05)}",
        f"Data version: {metadata.get('data_version', 'unknown')}",
        "",
        "10-day and 25-day forecasts are operational mathematical extensions of the 1-day model, not separately learned multi-horizon models.",
        "VaR and ES use sqrt(h) scaling; return uses h scaling.",
        "",
        "## Economic Backtest Interpretation",
        "",
        "The economic backtest evaluates whether the risk forecasts have historical decision-usefulness under a simulated risk-managed rule.",
        "Economic metrics should be interpreted as a simulated wealth comparison, historical maximum drawdown, and historical Sharpe ratio.",
        "The simulated risk-managed rule is used for historical evaluation only and does not constitute investment advice.",
        "",
        "## Latest Forecasts",
        "",
        "```text",
        display_df.to_string(index=False),
        "```",
    ]
    if plot_path is not None:
        lines.extend(["", f"Plot: `{plot_path}`"])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
