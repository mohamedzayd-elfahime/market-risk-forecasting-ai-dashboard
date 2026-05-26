"""HMM regime detection fitted only on the training block."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from backend.core.config import ForecastConfig


HIGH_VOLATILITY_FLOOR = 0.010


def fit_hmm(train_df: pd.DataFrame, feature_col: str, config: ForecastConfig) -> dict[str, object]:
    values = pd.to_numeric(train_df[feature_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 30:
        raise ValueError("Not enough observations to fit HMM.")

    scaler = StandardScaler()
    X = scaler.fit_transform(values.to_numpy().reshape(-1, 1))
    model = GaussianHMM(
        n_components=config.hmm_n_states,
        covariance_type="full",
        n_iter=300,
        random_state=config.seed,
    )
    model.fit(X)
    state_order = np.argsort(model.means_.ravel())
    state_labels = _build_state_labels(state_order)
    raw_state_means = scaler.inverse_transform(model.means_.reshape(-1, 1)).ravel()
    ordered_means = raw_state_means[state_order]
    thresholds = _build_volatility_thresholds(ordered_means)
    high_state = int(state_order[-1])
    return {
        "model": model,
        "scaler": scaler,
        "feature_col": feature_col,
        "high_state": high_state,
        "state_labels": state_labels,
        "state_means": {int(state): float(raw_state_means[int(state)]) for state in range(len(raw_state_means))},
        "low_medium_threshold": thresholds["low_medium_threshold"],
        "medium_high_threshold": thresholds["medium_high_threshold"],
        "high_volatility_floor": HIGH_VOLATILITY_FLOOR,
    }


def detect_current_regime(bundle: dict[str, object], history_df: pd.DataFrame) -> str:
    feature_col = bundle["feature_col"]
    values = pd.to_numeric(history_df[feature_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return "unknown"

    X = bundle["scaler"].transform(values.to_numpy().reshape(-1, 1))
    _, posterior = bundle["model"].score_samples(X)
    current_state = int(np.argmax(posterior[-1]))
    label = _label_for_state(bundle, current_state)

    latest_value = float(values.iloc[-1])
    low_medium_threshold = bundle.get("low_medium_threshold")
    medium_high_threshold = bundle.get("medium_high_threshold")
    if isinstance(low_medium_threshold, (int, float)) and latest_value <= float(low_medium_threshold):
        return "low_volatility"
    if isinstance(medium_high_threshold, (int, float)) and latest_value < float(medium_high_threshold):
        return "medium_volatility"

    return label


def _build_state_labels(state_order: np.ndarray) -> dict[int, str]:
    labels_by_rank = ["low_volatility", "medium_volatility", "high_volatility"]
    if len(state_order) == 1:
        labels_by_rank = ["medium_volatility"]
    elif len(state_order) == 2:
        labels_by_rank = ["low_volatility", "high_volatility"]

    return {
        int(state): labels_by_rank[min(rank, len(labels_by_rank) - 1)]
        for rank, state in enumerate(state_order)
    }


def _build_volatility_thresholds(ordered_means: np.ndarray) -> dict[str, float | None]:
    if len(ordered_means) >= 3:
        low_medium = float((ordered_means[0] + ordered_means[1]) / 2.0)
        medium_high = float((ordered_means[1] + ordered_means[2]) / 2.0)
    elif len(ordered_means) == 2:
        low_medium = None
        medium_high = float((ordered_means[0] + ordered_means[1]) / 2.0)
    else:
        low_medium = None
        medium_high = None

    if medium_high is None:
        medium_high = HIGH_VOLATILITY_FLOOR
    else:
        medium_high = max(medium_high, HIGH_VOLATILITY_FLOOR)

    return {
        "low_medium_threshold": low_medium,
        "medium_high_threshold": medium_high,
    }


def _label_for_state(bundle: dict[str, object], state: int) -> str:
    state_labels = bundle.get("state_labels")
    if isinstance(state_labels, dict) and state in state_labels:
        return str(state_labels[state])
    return "high_volatility" if state == bundle.get("high_state") else "low_volatility"
