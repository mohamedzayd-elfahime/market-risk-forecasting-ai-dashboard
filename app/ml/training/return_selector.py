"""Validation-only selector for return forecast candidates."""

from __future__ import annotations

import numpy as np


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(y_true, y_pred)[0, 1])
    return 0.0 if np.isnan(value) else value


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _std_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.std(y_true))
    return 0.0 if denom <= 1e-12 else float(np.std(y_pred) / denom)


def _dynamic_score(metrics: dict[str, float | str]) -> float:
    """Validation-only score that avoids selecting an over-flat return curve."""

    corr = float(metrics["corr"])
    directional = float(metrics["directional_accuracy"])
    std_ratio = float(metrics["std_ratio"])
    noisy_low_signal_penalty = 0.010 if corr < 0.10 and std_ratio > 0.75 else 0.0
    return (
        float(metrics["rmse"])
        + 0.018 * max(0.0, 0.22 - corr)
        + 0.006 * max(0.0, 0.55 - directional)
        + 0.006 * abs(std_ratio - 0.60)
        + 0.012 * max(0.0, 0.35 - std_ratio)
        + 0.010 * max(0.0, std_ratio - 1.00)
        + noisy_low_signal_penalty
    )


def fit_return_selector(y_true: np.ndarray, candidates: dict[str, np.ndarray]) -> dict[str, object]:
    """Choose the return forecast candidate using validation data only."""

    y_true = np.asarray(y_true, dtype=float)
    rows: list[dict[str, float | str]] = []
    for name, values in candidates.items():
        y_pred = np.asarray(values, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 5:
            continue
        metrics = {
            "name": name,
            "rmse": _rmse(y_true[mask], y_pred[mask]),
            "mae": _mae(y_true[mask], y_pred[mask]),
            "corr": _corr(y_true[mask], y_pred[mask]),
            "directional_accuracy": _directional_accuracy(y_true[mask], y_pred[mask]),
            "std_ratio": _std_ratio(y_true[mask], y_pred[mask]),
        }
        metrics["dynamic_score"] = _dynamic_score(metrics)
        rows.append(metrics)

    if not rows:
        return {"selected": "zero", "metrics": []}

    rows = sorted(rows, key=lambda row: (float(row["dynamic_score"]), float(row["rmse"])))
    return {"selected": str(rows[0]["name"]), "metrics": rows}


def apply_return_selector(
    candidates: dict[str, np.ndarray | float],
    selector: dict[str, object] | None,
) -> np.ndarray | float:
    selected = "lstm_calibrated"
    if selector:
        selected = str(selector.get("selected") or selected)
    if selected not in candidates:
        selected = "zero" if "zero" in candidates else next(iter(candidates))
    return candidates[selected]
