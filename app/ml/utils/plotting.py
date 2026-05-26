"""Simple price-scale forecast plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_price_proxy(last_price: float, return_forecast: float) -> float:
    """Reconstruct a forecast price proxy from a return forecast.

    Method: price_proxy = last_observed_price * exp(return_forecast).
    This is a visualization proxy because the model forecasts returns.
    """
    return float(last_price * np.exp(return_forecast))


def plot_latest_forecast(master_df: pd.DataFrame, forecast_df: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = master_df.copy().tail(250)
    history["date"] = pd.to_datetime(history["date"])
    one_day = forecast_df[forecast_df["horizon"].astype(int) == 1].iloc[-1]
    last_price = float(history["masi_close"].iloc[-1])
    price_proxy = build_price_proxy(last_price, float(one_day["return_forecast"]))
    target_date = pd.to_datetime(one_day["target_date"])

    var_price = build_price_proxy(last_price, float(one_day["var_forecast"]))
    es_price = build_price_proxy(last_price, float(one_day["es_forecast"]))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history["date"], history["masi_close"], color="black", linewidth=1.4, label="Observed MASI price")
    ax.scatter([target_date], [price_proxy], color="green", label="Forecast price proxy")
    ax.scatter([target_date], [var_price], color="goldenrod", label="VaR 5% price proxy")
    ax.scatter([target_date], [es_price], color="crimson", label="ES 5% price proxy")
    ax.set_title("MASI price-scale forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("MASI price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
