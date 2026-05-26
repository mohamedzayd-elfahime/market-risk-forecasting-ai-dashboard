"""Leakage-safe calibration for return forecasts."""

from __future__ import annotations

import numpy as np


def fit_return_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Fit a simple validation-only linear calibration y = intercept + slope * pred."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5 or np.std(y_pred[mask]) <= 1e-12:
        return {"intercept": 0.0, "slope": 1.0, "raw_std_ratio": 0.0}

    slope, intercept = np.polyfit(y_pred[mask], y_true[mask], deg=1)
    # Keep validation calibration conservative: it can correct scale drift,
    # but should not manufacture a high-amplitude return signal.
    slope = float(np.clip(slope, 0.25, 2.0))
    intercept = float(intercept)
    raw_std_ratio = float(np.std(y_pred[mask]) / (np.std(y_true[mask]) + 1e-12))
    return {"intercept": intercept, "slope": slope, "raw_std_ratio": raw_std_ratio}


def apply_return_calibration(y_pred: np.ndarray | float, calibration: dict[str, float] | None):
    if not calibration:
        return y_pred
    intercept = float(calibration.get("intercept", 0.0))
    slope = float(calibration.get("slope", 1.0))
    return intercept + slope * y_pred
