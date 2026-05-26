"""End-to-end MASI model training with chronological anti-leakage controls."""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import torch

from backend.core.config import ForecastConfig
from backend.core.paths import (
    ES_MODEL_PATH,
    HMM_MODEL_PATH,
    METADATA_PATH,
    RETURN_CALIBRATION_PATH,
    RETURN_EGARCH_AR1_PATH,
    RETURN_MODEL_PATH,
    RETURN_SCALER_PATH,
    RETURN_SELECTOR_PATH,
    RETURN_TABULAR_PATH,
    VAR_MODEL_PATH,
    VAR_SCALER_PATH,
    ensure_backend_dirs,
)
from ml.training.deep_models import predict_model, train_return_model, train_var_model
from ml.training.es_model import build_es_training_frame, fit_es_ridge_model, predict_es_ridge
from ml.training.hmm_model import fit_hmm
from ml.training.return_calibration import apply_return_calibration, fit_return_calibration
from ml.training.return_egarch import fit_egarch_innovation_ar1, predict_egarch_return_candidates
from ml.training.return_selector import apply_return_selector, fit_return_selector
from ml.training.return_tabular import (
    fit_return_tabular_model,
    predict_return_tabular_candidates,
    predict_return_tabular_model,
)
from ml.utils.preprocessing import (
    build_model_frame,
    chronological_train_test_split,
    fit_feature_scaler,
    transform_features,
)
from ml.utils.sequences import make_sequences_from_block, make_sequences_with_context


def _aligned_rows_no_context(df_block, seq_len: int):
    return df_block.iloc[seq_len - 1 :].copy().reset_index(drop=True)


def _aligned_rows_with_context(df_block):
    return df_block.copy().reset_index(drop=True)


def train_forecast_models(master_df, data_version: str, config: ForecastConfig) -> dict[str, object]:
    ensure_backend_dirs()
    var_feature_cols = list(config.var_feature_cols)
    return_feature_cols = list(config.return_feature_cols)

    model_df = build_model_frame(master_df, config)
    split = chronological_train_test_split(model_df, config)

    var_scaler = fit_feature_scaler(split.train_raw, var_feature_cols)
    return_scaler = fit_feature_scaler(split.train_raw, return_feature_cols)

    train_var_scaled = transform_features(split.train_raw, var_scaler, var_feature_cols)
    val_var_scaled = transform_features(split.val_raw, var_scaler, var_feature_cols)
    train_full_var_scaled = transform_features(split.train_full_raw, var_scaler, var_feature_cols)
    test_var_scaled = transform_features(split.test_raw, var_scaler, var_feature_cols)

    train_return_scaled = transform_features(split.train_raw, return_scaler, return_feature_cols)
    val_return_scaled = transform_features(split.val_raw, return_scaler, return_feature_cols)
    train_full_return_scaled = transform_features(split.train_full_raw, return_scaler, return_feature_cols)
    test_return_scaled = transform_features(split.test_raw, return_scaler, return_feature_cols)

    X_train_var, y_train_var, _ = make_sequences_from_block(
        train_var_scaled, var_feature_cols, config.target_col, config.date_col, config.seq_len
    )
    X_val_var, y_val_var, _ = make_sequences_with_context(
        train_var_scaled, val_var_scaled, var_feature_cols, config.target_col, config.date_col, config.seq_len
    )

    X_train_return, y_train_innovation, _ = make_sequences_from_block(
        train_return_scaled, return_feature_cols, "target_innovation", config.date_col, config.seq_len
    )
    X_val_return, y_val_innovation, _ = make_sequences_with_context(
        train_return_scaled,
        val_return_scaled,
        return_feature_cols,
        "target_innovation",
        config.date_col,
        config.seq_len,
    )

    var_result = train_var_model(X_train_var, y_train_var, X_val_var, y_val_var, config)
    return_result = train_return_model(
        X_train_return, y_train_innovation, X_val_return, y_val_innovation, config
    )

    X_dev_var, y_dev, _ = make_sequences_from_block(
        train_full_var_scaled, var_feature_cols, config.target_col, config.date_col, config.seq_len
    )
    X_test_var, y_test, d_test = make_sequences_with_context(
        train_full_var_scaled, test_var_scaled, var_feature_cols, config.target_col, config.date_col, config.seq_len
    )
    X_test_return, _, _ = make_sequences_with_context(
        train_full_return_scaled,
        test_return_scaled,
        return_feature_cols,
        "target_innovation",
        config.date_col,
        config.seq_len,
    )

    var_dev_pred = predict_model(var_result.model, X_dev_var)
    var_test_pred = predict_model(var_result.model, X_test_var) if len(X_test_var) else np.array([])
    return_val_innovation = predict_model(return_result.model, X_val_return) if len(X_val_return) else np.array([])
    return_test_innovation = (
        predict_model(return_result.model, X_test_return) if len(X_test_return) else np.array([])
    )

    dev_seq_raw = _aligned_rows_no_context(split.train_full_raw, config.seq_len)
    test_seq_raw = _aligned_rows_with_context(split.test_raw)

    es_dev_df = build_es_training_frame(dev_seq_raw, y_dev, var_dev_pred)
    es_bundle = fit_es_ridge_model(es_dev_df, config)
    es_test_pred, shortfall_test_pred = (
        predict_es_ridge(es_bundle, test_seq_raw, var_test_pred)
        if len(X_test_var)
        else (np.array([]), np.array([]))
    )

    hmm_feature = "masi_log_return_std_21"
    hmm_bundle = fit_hmm(split.train_raw, hmm_feature, config)

    val_raw = split.val_raw.reset_index(drop=True)
    raw_val_return_pred = (
        val_raw["target_mean"].to_numpy(dtype=float)
        + val_raw["target_vol"].to_numpy(dtype=float) * return_val_innovation
        if len(return_val_innovation)
        else np.array([])
    )
    return_calibration = fit_return_calibration(
        val_raw[config.target_col].to_numpy(dtype=float) if len(raw_val_return_pred) else np.array([]),
        raw_val_return_pred,
    )
    val_lstm_calibrated = apply_return_calibration(raw_val_return_pred, return_calibration)
    return_tabular_bundle = fit_return_tabular_model(split.train_raw)
    return_egarch_ar1_bundle = fit_egarch_innovation_ar1(split.train_raw)
    val_tabular_candidates = predict_return_tabular_candidates(
        return_tabular_bundle,
        split.val_raw,
        history_df=split.train_raw,
    )
    val_egarch_candidates = predict_egarch_return_candidates(val_raw, return_egarch_ar1_bundle)
    return_selector = fit_return_selector(
        val_raw[config.target_col].to_numpy(dtype=float),
        {
            **val_tabular_candidates,
            **val_egarch_candidates,
            "model_raw": raw_val_return_pred,
            "model_calibrated": val_lstm_calibrated,
            "lstm_raw": raw_val_return_pred,
            "lstm_calibrated": val_lstm_calibrated,
            "zero": np.zeros(len(val_raw), dtype=float),
        },
    )

    joblib.dump(var_scaler, VAR_SCALER_PATH)
    joblib.dump(return_scaler, RETURN_SCALER_PATH)
    joblib.dump(return_calibration, RETURN_CALIBRATION_PATH)
    joblib.dump(return_selector, RETURN_SELECTOR_PATH)
    joblib.dump(return_tabular_bundle, RETURN_TABULAR_PATH)
    joblib.dump(return_egarch_ar1_bundle, RETURN_EGARCH_AR1_PATH)
    joblib.dump(es_bundle, ES_MODEL_PATH)
    joblib.dump(hmm_bundle, HMM_MODEL_PATH)
    torch.save(var_result.model.state_dict(), VAR_MODEL_PATH)
    torch.save(return_result.model.state_dict(), RETURN_MODEL_PATH)

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": config.model_name,
        "model_version": config.model_version,
        "data_version": data_version,
        "alpha": config.alpha,
        "train_window": config.train_window,
        "seq_len": config.seq_len,
        "lstm_hidden_1": config.lstm_hidden_1,
        "lstm_hidden_2": config.lstm_hidden_2,
        "dense_hidden": config.dense_hidden,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "return_lstm_hidden_1": config.return_lstm_hidden_1 or config.lstm_hidden_1,
        "return_lstm_hidden_2": config.return_lstm_hidden_2 or config.lstm_hidden_2,
        "return_dense_hidden": config.return_dense_hidden or config.dense_hidden,
        "return_dropout": config.return_dropout if config.return_dropout is not None else config.dropout,
        "return_batch_size": config.return_batch_size or config.batch_size,
        "return_lr": config.return_lr if config.return_lr is not None else config.lr,
        "return_weight_decay": (
            config.return_weight_decay if config.return_weight_decay is not None else config.weight_decay
        ),
        "return_model_type": config.return_model_type,
        "return_transformer_d_model": config.return_transformer_d_model,
        "return_transformer_heads": config.return_transformer_heads,
        "return_transformer_layers": config.return_transformer_layers,
        "return_transformer_ff_mult": config.return_transformer_ff_mult,
        "return_transformer_patch_len": config.return_transformer_patch_len,
        "return_corr_weight": config.return_corr_weight,
        "return_var_weight": config.return_var_weight,
        "return_calibration_intercept": return_calibration.get("intercept"),
        "return_calibration_slope": return_calibration.get("slope"),
        "return_calibration_raw_std_ratio": return_calibration.get("raw_std_ratio"),
        "return_selected_candidate": return_selector.get("selected"),
        "return_selection_metrics": return_selector.get("metrics", []),
        "return_egarch_ar1_intercept": return_egarch_ar1_bundle.get("intercept"),
        "return_egarch_ar1_phi": return_egarch_ar1_bundle.get("phi"),
        "epochs": config.epochs,
        "patience": config.patience,
        "es_ridge_alpha": config.es_ridge_alpha,
        "var_feature_cols": var_feature_cols,
        "return_feature_cols": return_feature_cols,
        "train_start": str(split.train_raw[config.date_col].iloc[0]),
        "train_end": str(split.train_raw[config.date_col].iloc[-1]),
        "validation_start": str(split.val_raw[config.date_col].iloc[0]),
        "validation_end": str(split.val_raw[config.date_col].iloc[-1]),
        "test_start": str(split.test_raw[config.date_col].iloc[0]) if len(split.test_raw) else None,
        "test_end": str(split.test_raw[config.date_col].iloc[-1]) if len(split.test_raw) else None,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    test_predictions = test_seq_raw[[config.date_col, config.close_col, config.target_col]].copy()
    if len(test_predictions):
        raw_test_return_pred = (
            test_seq_raw["target_mean"].to_numpy()
            + test_seq_raw["target_vol"].to_numpy() * return_test_innovation
        )
        test_tabular_candidates = predict_return_tabular_candidates(
            return_tabular_bundle,
            test_seq_raw,
            history_df=split.train_full_raw,
        )
        test_egarch_candidates = predict_egarch_return_candidates(test_seq_raw, return_egarch_ar1_bundle)
        selected_test_return_pred = apply_return_selector(
            {
                **test_tabular_candidates,
                **test_egarch_candidates,
                "model_raw": raw_test_return_pred,
                "model_calibrated": apply_return_calibration(raw_test_return_pred, return_calibration),
                "lstm_raw": raw_test_return_pred,
                "lstm_calibrated": apply_return_calibration(raw_test_return_pred, return_calibration),
                "zero": np.zeros(len(test_seq_raw), dtype=float),
            },
            return_selector,
        )
        test_predictions = test_predictions.assign(
            realized_return=y_test,
            return_pred=selected_test_return_pred,
            var_pred=var_test_pred,
            es_pred=es_test_pred,
            predicted_shortfall=shortfall_test_pred,
            violation=(y_test < var_test_pred).astype(int),
        )

    return {
        "model_df": model_df,
        "split": split,
        "test_predictions": test_predictions,
        "metadata": metadata,
        "var_history": var_result.history,
        "return_history": return_result.history,
        "var_scaler": var_scaler,
        "return_scaler": return_scaler,
        "return_calibration": return_calibration,
        "return_selector": return_selector,
        "return_tabular_bundle": return_tabular_bundle,
        "es_bundle": es_bundle,
        "hmm_bundle": hmm_bundle,
        "var_model": var_result.model,
        "return_model": return_result.model,
    }
