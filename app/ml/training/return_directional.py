"""Directional return overlay converted into an EGARCH-scaled forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.training.return_selector import fit_return_selector
from ml.training.return_tabular import build_return_tabular_features


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "date",
        "masi_close",
        "target_return",
        "target_innovation",
        "target_mean",
        "target_vol",
        "target_std_resid",
    }
    return [col for col in frame.columns if col not in blocked]


def _clean_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return frame[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


def _predict_with_scale(df_block: pd.DataFrame, probabilities: np.ndarray, scale: float) -> np.ndarray:
    signal = 2.0 * probabilities - 1.0
    mean_col = "target_mean" if "target_mean" in df_block.columns else "masi_egarch_mean_next"
    vol_col = "target_vol" if "target_vol" in df_block.columns else "masi_egarch_vol_next"
    return (
        df_block[mean_col].to_numpy(dtype=float)
        + df_block[vol_col].to_numpy(dtype=float) * float(scale) * signal
    )


def fit_directional_egarch_overlay(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
) -> dict[str, object]:
    """Fit P(r_{t+1}>0 | info_t), then tune EGARCH-scaled amplitude on validation."""

    train_frame = build_return_tabular_features(train_raw, max_lag=21).dropna().reset_index(drop=True)
    val_frame = build_return_tabular_features(val_raw, history_df=train_raw, max_lag=21)
    val_frame = val_frame.ffill().bfill().fillna(0.0).reset_index(drop=True)

    feature_cols = _feature_columns(train_frame)
    y_train = (train_frame["target_return"].to_numpy(dtype=float) > 0.0).astype(int)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.2,
            l1_ratio=0.0,
            max_iter=2000,
            class_weight="balanced",
        ),
    )
    model.fit(_clean_features(train_frame, feature_cols), y_train)

    val_proba = model.predict_proba(_clean_features(val_frame, feature_cols))[:, 1]
    y_val = val_frame["target_return"].to_numpy(dtype=float)
    best_scale = 1.0
    best_score = float("inf")
    for scale in np.linspace(0.05, 2.5, 50):
        pred = _predict_with_scale(val_frame, val_proba, float(scale))
        selector = fit_return_selector(y_val, {"candidate": pred})
        metrics = selector.get("metrics", [])
        if not metrics:
            continue
        score = float(metrics[0].get("dynamic_score", metrics[0].get("rmse", float("inf"))))
        if score < best_score:
            best_score = score
            best_scale = float(scale)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "max_lag": 21,
        "scale": best_scale,
        "validation_dynamic_score": best_score,
    }


def predict_directional_egarch_overlay(
    bundle: dict[str, object],
    df_block: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> np.ndarray:
    frame = build_return_tabular_features(
        df_block,
        history_df=history_df,
        max_lag=int(bundle.get("max_lag", 21)),
    )
    frame = frame.ffill().bfill().fillna(0.0).reset_index(drop=True)
    feature_cols = list(bundle["feature_cols"])
    probabilities = bundle["model"].predict_proba(_clean_features(frame, feature_cols))[:, 1]
    return _predict_with_scale(frame, probabilities, float(bundle.get("scale", 1.0)))
