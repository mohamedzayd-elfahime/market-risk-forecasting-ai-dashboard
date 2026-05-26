"""Two-step ES model: Ridge on VaR shortfalls."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from backend.core.config import ForecastConfig


def build_es_training_frame(seq_raw_df: pd.DataFrame, y_true: np.ndarray, var_pred: np.ndarray) -> pd.DataFrame:
    out = seq_raw_df.copy().reset_index(drop=True)
    out["y_true"] = y_true
    out["var_pred"] = var_pred
    out["abs_var_pred"] = np.abs(var_pred)
    out["violation"] = (out["y_true"] < out["var_pred"]).astype(int)
    out["shortfall"] = np.maximum(out["var_pred"] - out["y_true"], 0.0)
    return out


def fit_es_ridge_model(es_train_df: pd.DataFrame, config: ForecastConfig) -> dict[str, object]:
    viol_df = es_train_df[es_train_df["violation"] == 1].copy()
    if len(viol_df) < config.es_min_violations:
        raise ValueError(f"Not enough VaR violations for ES Ridge: {len(viol_df)}")

    feature_names = [
        "abs_var_pred",
        "masi_log_return",
        "masi_log_return_std_5",
        "masi_log_return_std_21",
        "masi_log_return_mean_21",
    ]
    X = viol_df[feature_names].copy()
    y_shortfall = viol_df["shortfall"].clip(lower=config.es_shortfall_floor).to_numpy()
    y_log = np.log1p(y_shortfall)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=config.es_ridge_alpha)
    model.fit(X_scaled, y_log)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "shortfall_floor": config.es_shortfall_floor,
        "n_violations_used": len(viol_df),
    }


def predict_es_ridge(bundle: dict[str, object], seq_raw_df: pd.DataFrame, var_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = pd.DataFrame(
        {
            "abs_var_pred": np.abs(var_pred),
            "masi_log_return": seq_raw_df["masi_log_return"].to_numpy(),
            "masi_log_return_std_5": seq_raw_df["masi_log_return_std_5"].to_numpy(),
            "masi_log_return_std_21": seq_raw_df["masi_log_return_std_21"].to_numpy(),
            "masi_log_return_mean_21": seq_raw_df["masi_log_return_mean_21"].to_numpy(),
        }
    )
    X_scaled = bundle["scaler"].transform(X)
    pred_log_shortfall = bundle["model"].predict(X_scaled)
    predicted_shortfall = np.maximum(np.expm1(pred_log_shortfall), bundle["shortfall_floor"])
    es_pred = np.minimum(var_pred - predicted_shortfall, var_pred)
    return es_pred, predicted_shortfall
