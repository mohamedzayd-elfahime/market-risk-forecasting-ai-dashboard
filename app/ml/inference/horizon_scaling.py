"""Operational multi-horizon scaling derived from the 1-day model."""

from __future__ import annotations

import math


def scale_one_day_forecast(return_1d: float, var_1d: float, es_1d: float, horizon: int) -> tuple[float, float, float]:
    """Scale 1-day forecasts.

    10d and 25d outputs are operational extensions, not separately learned
    multi-horizon models. The conditional mean scales linearly in h under a
    simple aggregation assumption, while VaR and ES use sqrt(h) as operational
    risk extensions.
    """
    risk_scale = math.sqrt(horizon)
    return horizon * return_1d, risk_scale * var_1d, risk_scale * es_1d


def scale_auxiliary_forecasts(mean_1d: float, volatility_1d: float, horizon: int) -> tuple[float, float]:
    """Scale auxiliary EGARCH mean and volatility forecasts to the chosen horizon."""
    return horizon * mean_1d, math.sqrt(horizon) * volatility_1d
