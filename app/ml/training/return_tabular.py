"""Leakage-safe tabular return model used as a candidate forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


LAG_SOURCE_COLS = (
    "masi_log_return",
    "masi_log_open_close",
    "masi_log_high_low",
    "masi_price_range_pct",
    "masi_egarch_std_resid",
    "masi_egarch_vol",
    "masi_log_return_mean_21",
    "masi_log_return_std_5",
    "masi_log_return_std_21",
)


def build_return_tabular_features(
    df_block: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    max_lag: int = 10,
) -> pd.DataFrame:
    """Build row-aligned features using only information available up to row t."""

    if history_df is not None and len(history_df):
        history = history_df.tail(max_lag + 50)
        combined = pd.concat([history, df_block], axis=0).reset_index(drop=True)
        start = len(combined) - len(df_block)
    else:
        combined = df_block.copy().reset_index(drop=True)
        start = 0

    out = combined.copy()
    for col in LAG_SOURCE_COLS:
        if col not in out.columns:
            continue
        for lag in range(1, max_lag + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    returns = out["masi_log_return"]
    for window in (2, 3, 5, 10, 21, 42):
        out[f"ret_sum_{window}"] = returns.rolling(window).sum()
        out[f"ret_mean_{window}"] = returns.rolling(window).mean()
        out[f"ret_std_{window}"] = returns.rolling(window).std()
        out[f"pos_rate_{window}"] = (returns > 0).rolling(window).mean()

    out["ret_sign"] = np.sign(returns)
    out["abs_ret"] = returns.abs()
    return out.iloc[start:].copy().reset_index(drop=True)


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


def fit_return_tabular_model(train_raw: pd.DataFrame) -> dict[str, object]:
    train_frame = build_return_tabular_features(train_raw, max_lag=21)
    feature_cols = _feature_columns(train_frame)
    train_frame = train_frame.dropna(subset=[*feature_cols, "target_return"]).reset_index(drop=True)

    models = {
        "tabular_huber": make_pipeline(
            RobustScaler(),
            HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=3000),
        ),
        "tabular_ridge": make_pipeline(
            RobustScaler(),
            Ridge(alpha=1.0),
        ),
        "tabular_hgb": HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=15,
            l2_regularization=0.001,
            random_state=42,
        ),
    }
    for model in models.values():
        model.fit(train_frame[feature_cols], train_frame["target_return"])
    return {"model": models["tabular_huber"], "models": models, "feature_cols": feature_cols, "max_lag": 21}


def predict_return_tabular_candidates(
    bundle: dict[str, object],
    df_block: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    frame = build_return_tabular_features(
        df_block,
        history_df=history_df,
        max_lag=int(bundle.get("max_lag", 21)),
    )
    feature_cols = list(bundle["feature_cols"])
    features = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0.0)
    if "models" in bundle:
        return {
            name: np.asarray(model.predict(features), dtype=float)
            for name, model in dict(bundle["models"]).items()
        }
    return {"tabular_huber": np.asarray(bundle["model"].predict(features), dtype=float)}


def predict_return_tabular_model(
    bundle: dict[str, object],
    df_block: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> np.ndarray:
    candidates = predict_return_tabular_candidates(bundle, df_block, history_df=history_df)
    return candidates.get("tabular_huber", next(iter(candidates.values())))
