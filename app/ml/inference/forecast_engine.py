"""1-day-ahead MASI inference plus operational horizon extensions."""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch

from backend.core.config import ForecastConfig
from backend.core.paths import (
    ES_MODEL_PATH,
    HMM_MODEL_PATH,
    METADATA_PATH,
    RETURN_CALIBRATION_PATH,
    RETURN_DIRECTIONAL_PATH,
    RETURN_EGARCH_AR1_PATH,
    RETURN_MODEL_PATH,
    RETURN_SCALER_PATH,
    RETURN_SELECTOR_PATH,
    RETURN_TABULAR_PATH,
    VAR_MODEL_PATH,
    VAR_SCALER_PATH,
)
from ml.inference.horizon_scaling import scale_auxiliary_forecasts, scale_one_day_forecast
from ml.training.deep_models import QuantileLSTM, ReturnCnnLstm, ReturnPatchTransformer, _model_hyperparameters, predict_model
from ml.training.es_model import predict_es_ridge
from ml.training.hmm_model import detect_current_regime
from ml.training.return_calibration import apply_return_calibration
from ml.training.return_directional import predict_directional_egarch_overlay
from ml.training.return_egarch import predict_latest_egarch_return_candidates
from ml.training.return_selector import apply_return_selector
from ml.training.return_tabular import predict_return_tabular_candidates
from ml.utils.preprocessing import build_inference_frame, transform_features
from ml.utils.sequences import latest_inference_sequence


FIXED_MARKET_HOLIDAYS = (
    (1, 1),    # New Year
    (1, 11),   # Independence Manifesto Day
    (5, 1),    # Labour Day
    (7, 30),   # Throne Day
    (8, 14),   # Oued Ed-Dahab Day
    (8, 20),   # Revolution Day
    (8, 21),   # Youth Day
    (11, 6),   # Green March Day
    (11, 18),  # Independence Day
)


def _moroccan_market_holidays(years: range) -> list[pd.Timestamp]:
    return [
        pd.Timestamp(year=year, month=month, day=day)
        for year in years
        for month, day in FIXED_MARKET_HOLIDAYS
    ]


def _target_market_date(last_date: pd.Timestamp, horizon: int) -> pd.Timestamp:
    years = range(last_date.year, last_date.year + 3)
    market_day = pd.offsets.CustomBusinessDay(
        weekmask="Mon Tue Wed Thu Fri",
        holidays=_moroccan_market_holidays(years),
    )
    return last_date + horizon * market_day


def _load_lstm(path, input_size: int, config: ForecastConfig, model_role: str = "var") -> QuantileLSTM:
    hyperparams = _model_hyperparameters(config, model_role=model_role)
    model = QuantileLSTM(
        input_size=input_size,
        lstm_hidden_1=int(hyperparams["lstm_hidden_1"]),
        lstm_hidden_2=int(hyperparams["lstm_hidden_2"]),
        dense_hidden=int(hyperparams["dense_hidden"]),
        dropout=float(hyperparams["dropout"]),
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def _load_return_model(path, input_size: int, seq_len: int, config: ForecastConfig):
    model_type = config.return_model_type.strip().lower()
    if model_type in {"patch_transformer", "transformer", "patchtst"}:
        model = ReturnPatchTransformer(
            input_size=input_size,
            seq_len=seq_len,
            patch_len=config.return_transformer_patch_len,
            d_model=config.return_transformer_d_model,
            n_heads=config.return_transformer_heads,
            n_layers=config.return_transformer_layers,
            ff_mult=config.return_transformer_ff_mult,
            dropout=config.return_dropout if config.return_dropout is not None else config.dropout,
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
    if model_type in {"cnn_lstm", "cnnlstm", "egarch_cnn_lstm"}:
        hyperparams = _model_hyperparameters(config, model_role="return")
        model = ReturnCnnLstm(
            input_size=input_size,
            cnn_filters=config.return_cnn_filters,
            kernel_size=config.return_cnn_kernel_size,
            lstm_hidden_1=int(hyperparams["lstm_hidden_1"]),
            lstm_hidden_2=int(hyperparams["lstm_hidden_2"]),
            dense_hidden=int(hyperparams["dense_hidden"]),
            dropout=float(hyperparams["dropout"]),
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
    return _load_lstm(path, input_size, config, model_role="return")


def make_forecast(master_df: pd.DataFrame, config: ForecastConfig, data_version: str) -> pd.DataFrame:
    var_feature_cols = list(config.var_feature_cols)
    return_feature_cols = list(config.return_feature_cols)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    var_scaler = joblib.load(VAR_SCALER_PATH)
    return_selector = joblib.load(RETURN_SELECTOR_PATH) if RETURN_SELECTOR_PATH.exists() else None
    return_tabular_bundle = joblib.load(RETURN_TABULAR_PATH) if RETURN_TABULAR_PATH.exists() else None
    return_egarch_ar1_bundle = joblib.load(RETURN_EGARCH_AR1_PATH) if RETURN_EGARCH_AR1_PATH.exists() else None
    return_directional_bundle = joblib.load(RETURN_DIRECTIONAL_PATH) if RETURN_DIRECTIONAL_PATH.exists() else None
    es_bundle = joblib.load(ES_MODEL_PATH)
    hmm_bundle = joblib.load(HMM_MODEL_PATH)

    model_df = build_inference_frame(master_df, config)
    var_scaled = transform_features(model_df, var_scaler, var_feature_cols)
    X_latest_var = latest_inference_sequence(var_scaled, var_feature_cols, config.seq_len)

    var_model = _load_lstm(VAR_MODEL_PATH, len(var_feature_cols), config, model_role="var")

    var_1d = float(predict_model(var_model, X_latest_var)[0])
    latest_raw = model_df.tail(1).copy().reset_index(drop=True)
    es_1d = float(predict_es_ridge(es_bundle, latest_raw, np.array([var_1d]))[0][0])
    regime = detect_current_regime(hmm_bundle, model_df)

    mean_1d = float(latest_raw["masi_egarch_mean_next"].iloc[0])
    target_vol = float(latest_raw["masi_egarch_vol_next"].iloc[0])
    innovation_t = float(latest_raw["masi_egarch_std_resid"].iloc[0])
    if not np.isfinite(innovation_t):
        innovation_t = 0.0
    egarch_innovation_t = mean_1d + target_vol * innovation_t
    candidates = {
        "egarch_mean": mean_1d,
        "egarch_innovation_t": egarch_innovation_t,
        "zero": 0.0,
    }
    candidates.update(predict_latest_egarch_return_candidates(latest_raw, return_egarch_ar1_bundle))
    if return_directional_bundle is not None:
        candidates["directional_logit_egarch"] = float(
            predict_directional_egarch_overlay(
                return_directional_bundle,
                model_df.tail(1),
                history_df=model_df.iloc[:-1],
            )[0]
        )
    if return_tabular_bundle is not None:
        tabular_candidates = predict_return_tabular_candidates(
            return_tabular_bundle,
            model_df.tail(1),
            history_df=model_df.iloc[:-1],
        )
        candidates.update({name: float(values[0]) for name, values in tabular_candidates.items()})
    return_1d = float(apply_return_selector(candidates, return_selector))

    last_date = pd.to_datetime(model_df[config.date_col].iloc[-1])
    run_date = datetime.now().isoformat(timespec="seconds")
    rows = []
    for horizon in config.horizons:
        mean_h, vol_h = scale_auxiliary_forecasts(mean_1d, target_vol, horizon)
        _, var_h, es_h = scale_one_day_forecast(return_1d, var_1d, es_1d, horizon)
        ret_h = return_1d if horizon == 1 else mean_h
        rows.append(
            {
                "run_date": run_date,
                "target_date": _target_market_date(last_date, horizon),
                "model_name": config.model_name,
                "horizon": horizon,
                "alpha": config.alpha,
                "mean_forecast": mean_h,
                "volatility_forecast": vol_h,
                "return_forecast": ret_h,
                "var_forecast": var_h,
                "es_forecast": es_h,
                "regime": regime,
                "weight": np.nan,
                "model_version": metadata.get("model_version", config.model_version),
                "data_version": data_version,
                "realized_return": np.nan,
            }
        )
    return pd.DataFrame(rows)
