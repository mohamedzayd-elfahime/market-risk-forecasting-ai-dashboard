"""Chronological preprocessing with explicit anti-leakage rules."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import RobustScaler

from backend.core.config import ForecastConfig


@dataclass
class ChronologicalSplit:
    train_raw: pd.DataFrame
    val_raw: pd.DataFrame
    train_full_raw: pd.DataFrame
    test_raw: pd.DataFrame


def build_model_frame(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    data = df.copy().sort_values(config.date_col).reset_index(drop=True)
    data[config.target_col] = data[config.return_col].shift(-1)
    data["target_mean"] = data["masi_egarch_mean_next"]
    data["target_vol"] = data["masi_egarch_vol_next"]
    data["target_innovation"] = (data[config.target_col] - data["target_mean"]) / data["target_vol"]
    data["target_std_resid"] = data[config.target_col] / data["target_vol"]
    feature_cols = sorted(set(config.var_feature_cols).union(config.return_feature_cols))

    required = [
        config.date_col,
        config.close_col,
        config.target_col,
        "target_mean",
        "target_vol",
        "target_innovation",
        "target_std_resid",
        *feature_cols,
    ]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")

    model = data[required].replace([float("inf"), float("-inf")], pd.NA)
    model = model.dropna().reset_index(drop=True)
    return model


def build_inference_frame(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    data = df.copy().sort_values(config.date_col).reset_index(drop=True)
    feature_cols = sorted(set(config.var_feature_cols).union(config.return_feature_cols))

    required = [config.date_col, config.close_col, "masi_egarch_mean_next", "masi_egarch_vol_next", *feature_cols]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing inference columns: {missing}")

    model = data[required].replace([float("inf"), float("-inf")], pd.NA)
    model = model.dropna().reset_index(drop=True)
    return model


def chronological_train_test_split(model_df: pd.DataFrame, config: ForecastConfig) -> ChronologicalSplit:
    ordered = model_df.sort_values(config.date_col).reset_index(drop=True)
    test_window = max(1, int(round(config.train_window * config.test_window_ratio)))
    required = config.train_window + test_window

    if len(ordered) < required:
        raise ValueError(
            f"Need at least {required} valid rows for the dynamic train/test window "
            f"({config.train_window} train + {test_window} test). Available rows: {len(ordered)}."
        )

    selected = ordered.iloc[-required:].copy().reset_index(drop=True)
    train_full = selected.iloc[: config.train_window].copy().reset_index(drop=True)
    test = selected.iloc[config.train_window :].copy().reset_index(drop=True)

    n_val = max(int(round(len(train_full) * config.val_fraction_within_train)), config.seq_len + 5)
    if n_val >= len(train_full):
        raise ValueError("Validation block is too large for the fixed train window.")

    train = train_full.iloc[: -n_val].copy().reset_index(drop=True)
    val = train_full.iloc[-n_val:].copy().reset_index(drop=True)
    return ChronologicalSplit(train, val, train_full, test)


def fit_feature_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def transform_features(df: pd.DataFrame, scaler: RobustScaler, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out.loc[:, feature_cols] = scaler.transform(out[feature_cols])
    return out
