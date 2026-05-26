"""Retrain only the MASI return branch and refresh forecasts.

This job intentionally leaves the VaR, ES and HMM artifacts untouched. It is
useful when experimenting with the return/innovation model without replacing
the risk backtest artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import DEFAULT_CONFIG, ForecastConfig
from backend.core.paths import (
    MASTER_DATASET_PATH,
    METADATA_PATH,
    RETURN_CALIBRATION_PATH,
    RETURN_DIRECTIONAL_PATH,
    RETURN_EGARCH_AR1_PATH,
    RETURN_MODEL_PATH,
    RETURN_SCALER_PATH,
    RETURN_SELECTOR_PATH,
    RETURN_TABULAR_PATH,
    TEST_PREDICTIONS_PATH,
    TRAINING_HISTORY_PATH,
    ensure_backend_dirs,
)
from backend.services.forecast_service import run_forecast_pipeline
from jobs.hyperparameter_config import TUNABLE_RETURN_CONFIG_KEYS
from ml.training.deep_models import predict_model, train_return_model
from ml.training.return_directional import fit_directional_egarch_overlay, predict_directional_egarch_overlay
from ml.training.return_calibration import apply_return_calibration, fit_return_calibration
from ml.training.return_egarch import fit_egarch_innovation_ar1, predict_egarch_return_candidates
from ml.training.return_selector import apply_return_selector, fit_return_selector
from ml.training.return_tabular import fit_return_tabular_model, predict_return_tabular_model
from ml.training.return_tabular import predict_return_tabular_candidates
from ml.utils.backtesting import compute_wealth_curves
from ml.utils.data_loading import dataset_version, load_master_dataset
from ml.utils.preprocessing import (
    build_model_frame,
    chronological_train_test_split,
    fit_feature_scaler,
    transform_features,
)
from ml.utils.sequences import make_sequences_from_block, make_sequences_with_context


def _metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "corr": float("nan"), "directional_accuracy": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 2 and np.std(yp) > 1e-12 else 0.0
    return {
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "mae": float(np.mean(np.abs(yt - yp))),
        "corr": 0.0 if np.isnan(corr) else corr,
        "directional_accuracy": float(np.mean(np.sign(yt) == np.sign(yp))),
        "prediction_std_ratio": float(np.std(yp) / (np.std(yt) + 1e-12)),
    }


def _config_from_metadata() -> ForecastConfig:
    if not METADATA_PATH.exists():
        return DEFAULT_CONFIG
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(ForecastConfig)}
    overrides: dict[str, Any] = {}
    for key in allowed:
        if key in metadata:
            overrides[key] = metadata[key]
    return replace(DEFAULT_CONFIG, **overrides)


def _apply_return_params(config: ForecastConfig, path: Path | None) -> ForecastConfig:
    if path is None:
        return config
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    if not isinstance(params, dict):
        raise ValueError(f"No best_params object found in {path}.")
    return_params = {key: params[key] for key in TUNABLE_RETURN_CONFIG_KEYS if key in params}
    if not return_params:
        raise ValueError(f"No return hyperparameters found in {path}.")
    return replace(config, **return_params)


def _return_model_candidate_names(config: ForecastConfig) -> tuple[str, str]:
    model_type = config.return_model_type.strip().lower()
    if model_type in {"patch_transformer", "transformer", "patchtst"}:
        return ("transformer_raw", "transformer_calibrated")
    if model_type in {"cnn_lstm", "cnnlstm", "egarch_cnn_lstm"}:
        return ("cnn_lstm_raw", "cnn_lstm_calibrated")
    return ("lstm_raw", "lstm_calibrated")


def _update_training_history(return_history: dict[str, list[float]]) -> None:
    payload: dict[str, Any] = {}
    if TRAINING_HISTORY_PATH.exists():
        payload = json.loads(TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
    payload["return_history"] = return_history
    TRAINING_HISTORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_return_only(config: ForecastConfig, return_params_source: str | None = None) -> dict[str, Any]:
    ensure_backend_dirs()
    master_df = load_master_dataset(MASTER_DATASET_PATH)
    data_version = dataset_version(MASTER_DATASET_PATH)
    return_feature_cols = list(config.return_feature_cols)

    model_df = build_model_frame(master_df, config)
    split = chronological_train_test_split(model_df, config)

    return_scaler = fit_feature_scaler(split.train_raw, return_feature_cols)
    train_return_scaled = transform_features(split.train_raw, return_scaler, return_feature_cols)
    val_return_scaled = transform_features(split.val_raw, return_scaler, return_feature_cols)
    train_full_return_scaled = transform_features(split.train_full_raw, return_scaler, return_feature_cols)
    test_return_scaled = transform_features(split.test_raw, return_scaler, return_feature_cols)

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
    X_test_return, _, _ = make_sequences_with_context(
        train_full_return_scaled,
        test_return_scaled,
        return_feature_cols,
        "target_innovation",
        config.date_col,
        config.seq_len,
    )

    return_result = train_return_model(
        X_train_return,
        y_train_innovation,
        X_val_return,
        y_val_innovation,
        config,
    )
    raw_candidate_name, calibrated_candidate_name = _return_model_candidate_names(config)

    val_raw = split.val_raw.reset_index(drop=True)
    return_val_innovation = predict_model(return_result.model, X_val_return) if len(X_val_return) else np.array([])
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
    return_directional_bundle = fit_directional_egarch_overlay(split.train_raw, split.val_raw)
    val_tabular_candidates = predict_return_tabular_candidates(
        return_tabular_bundle,
        split.val_raw,
        history_df=split.train_raw,
    )
    val_egarch_candidates = predict_egarch_return_candidates(val_raw, return_egarch_ar1_bundle)
    val_directional_pred = predict_directional_egarch_overlay(
        return_directional_bundle,
        split.val_raw,
        history_df=split.train_raw,
    )
    return_selector = fit_return_selector(
        val_raw[config.target_col].to_numpy(dtype=float),
        {
            **val_tabular_candidates,
            **val_egarch_candidates,
            "directional_logit_egarch": val_directional_pred,
            raw_candidate_name: raw_val_return_pred,
            calibrated_candidate_name: val_lstm_calibrated,
            "model_raw": raw_val_return_pred,
            "model_calibrated": val_lstm_calibrated,
            "zero": np.zeros(len(val_raw), dtype=float),
        },
    )

    joblib.dump(return_scaler, RETURN_SCALER_PATH)
    joblib.dump(return_calibration, RETURN_CALIBRATION_PATH)
    joblib.dump(return_selector, RETURN_SELECTOR_PATH)
    joblib.dump(return_tabular_bundle, RETURN_TABULAR_PATH)
    joblib.dump(return_egarch_ar1_bundle, RETURN_EGARCH_AR1_PATH)
    joblib.dump(return_directional_bundle, RETURN_DIRECTIONAL_PATH)
    torch.save(return_result.model.state_dict(), RETURN_MODEL_PATH)

    test_seq_raw = split.test_raw.reset_index(drop=True)
    return_test_innovation = predict_model(return_result.model, X_test_return) if len(X_test_return) else np.array([])
    raw_test_return_pred = (
        test_seq_raw["target_mean"].to_numpy(dtype=float)
        + test_seq_raw["target_vol"].to_numpy(dtype=float) * return_test_innovation
        if len(return_test_innovation)
        else np.array([])
    )
    test_tabular_candidates = predict_return_tabular_candidates(
        return_tabular_bundle,
        test_seq_raw,
        history_df=split.train_full_raw,
    )
    test_egarch_candidates = predict_egarch_return_candidates(test_seq_raw, return_egarch_ar1_bundle)
    test_directional_pred = predict_directional_egarch_overlay(
        return_directional_bundle,
        test_seq_raw,
        history_df=split.train_full_raw,
    )
    selected_test_return_pred = np.asarray(
        apply_return_selector(
            {
                **test_tabular_candidates,
                **test_egarch_candidates,
                "directional_logit_egarch": test_directional_pred,
                raw_candidate_name: raw_test_return_pred,
                calibrated_candidate_name: apply_return_calibration(raw_test_return_pred, return_calibration),
                "model_raw": raw_test_return_pred,
                "model_calibrated": apply_return_calibration(raw_test_return_pred, return_calibration),
                "zero": np.zeros(len(test_seq_raw), dtype=float),
            },
            return_selector,
        ),
        dtype=float,
    )

    if TEST_PREDICTIONS_PATH.exists():
        test_predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
        if len(test_predictions) == len(selected_test_return_pred):
            test_predictions["return_pred"] = selected_test_return_pred
        else:
            update_frame = pd.DataFrame(
                {
                    config.date_col: pd.to_datetime(test_seq_raw[config.date_col]).dt.strftime("%Y-%m-%d"),
                    "return_pred_new": selected_test_return_pred,
                }
            )
            test_predictions[config.date_col] = pd.to_datetime(test_predictions[config.date_col]).dt.strftime("%Y-%m-%d")
            test_predictions = test_predictions.merge(update_frame, on=config.date_col, how="left")
            test_predictions["return_pred"] = test_predictions["return_pred_new"].fillna(test_predictions["return_pred"])
            test_predictions = test_predictions.drop(columns=["return_pred_new"])
    else:
        test_predictions = test_seq_raw[[config.date_col, config.close_col, config.target_col]].copy()
        test_predictions = test_predictions.assign(
            realized_return=test_seq_raw[config.target_col].to_numpy(dtype=float),
            return_pred=selected_test_return_pred,
        )
    test_predictions = compute_wealth_curves(test_predictions)
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    metadata.update(
        {
            "return_retrained_at": datetime.now().isoformat(timespec="seconds"),
            "return_update_mode": "return_only",
            "return_params_source": return_params_source,
            "data_version": data_version,
            "seq_len": config.seq_len,
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
            "return_cnn_filters": config.return_cnn_filters,
            "return_cnn_kernel_size": config.return_cnn_kernel_size,
            "return_corr_weight": config.return_corr_weight,
            "return_var_weight": config.return_var_weight,
            "return_epochs": config.epochs,
            "return_patience": config.patience,
            "return_calibration_intercept": return_calibration.get("intercept"),
            "return_calibration_slope": return_calibration.get("slope"),
            "return_calibration_raw_std_ratio": return_calibration.get("raw_std_ratio"),
            "return_selected_candidate": return_selector.get("selected"),
            "return_selection_metrics": return_selector.get("metrics", []),
            "return_egarch_ar1_intercept": return_egarch_ar1_bundle.get("intercept"),
            "return_egarch_ar1_phi": return_egarch_ar1_bundle.get("phi"),
            "return_directional_overlay_scale": return_directional_bundle.get("scale"),
            "return_directional_overlay_validation_dynamic_score": return_directional_bundle.get(
                "validation_dynamic_score"
            ),
            "return_feature_cols": return_feature_cols,
            "train_start": str(split.train_raw[config.date_col].iloc[0]),
            "train_end": str(split.train_raw[config.date_col].iloc[-1]),
            "validation_start": str(split.val_raw[config.date_col].iloc[0]),
            "validation_end": str(split.val_raw[config.date_col].iloc[-1]),
            "test_start": str(split.test_raw[config.date_col].iloc[0]) if len(split.test_raw) else None,
            "test_end": str(split.test_raw[config.date_col].iloc[-1]) if len(split.test_raw) else None,
        }
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _update_training_history(return_result.history)

    return {
        "metadata": metadata,
        "return_selector": return_selector,
        "validation": _metric_summary(val_raw[config.target_col].to_numpy(dtype=float), val_lstm_calibrated),
        "test": _metric_summary(test_seq_raw[config.target_col].to_numpy(dtype=float), selected_test_return_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain only the MASI return branch.")
    parser.add_argument(
        "--return-params-json",
        type=Path,
        default=None,
        help="Optional JSON file containing best_params for return hyperparameters.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override return training epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override return early-stopping patience.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length for the return branch.")
    parser.add_argument("--return-lstm-hidden-1", type=int, default=None)
    parser.add_argument("--return-lstm-hidden-2", type=int, default=None)
    parser.add_argument("--return-dense-hidden", type=int, default=None)
    parser.add_argument("--return-dropout", type=float, default=None)
    parser.add_argument("--return-batch-size", type=int, default=None)
    parser.add_argument("--return-lr", type=float, default=None)
    parser.add_argument("--return-weight-decay", type=float, default=None)
    parser.add_argument(
        "--model-type",
        choices=("lstm", "patch_transformer", "cnn_lstm"),
        default=None,
        help="Return branch architecture to train.",
    )
    parser.add_argument("--transformer-d-model", type=int, default=None)
    parser.add_argument("--transformer-heads", type=int, default=None)
    parser.add_argument("--transformer-layers", type=int, default=None)
    parser.add_argument("--transformer-patch-len", type=int, default=None)
    parser.add_argument("--cnn-filters", type=int, default=None)
    parser.add_argument("--cnn-kernel-size", type=int, default=None)
    parser.add_argument("--no-forecast", action="store_true", help="Do not run inference after updating return artifacts.")
    args = parser.parse_args()

    config = _config_from_metadata()
    config = _apply_return_params(config, args.return_params_json)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.patience is not None:
        config = replace(config, patience=args.patience)
    explicit_return_overrides = {}
    if args.seq_len is not None:
        explicit_return_overrides["seq_len"] = args.seq_len
    if args.return_lstm_hidden_1 is not None:
        explicit_return_overrides["return_lstm_hidden_1"] = args.return_lstm_hidden_1
    if args.return_lstm_hidden_2 is not None:
        explicit_return_overrides["return_lstm_hidden_2"] = args.return_lstm_hidden_2
    if args.return_dense_hidden is not None:
        explicit_return_overrides["return_dense_hidden"] = args.return_dense_hidden
    if args.return_dropout is not None:
        explicit_return_overrides["return_dropout"] = args.return_dropout
    if args.return_batch_size is not None:
        explicit_return_overrides["return_batch_size"] = args.return_batch_size
    if args.return_lr is not None:
        explicit_return_overrides["return_lr"] = args.return_lr
    if args.return_weight_decay is not None:
        explicit_return_overrides["return_weight_decay"] = args.return_weight_decay
    if explicit_return_overrides:
        config = replace(config, **explicit_return_overrides)
    transformer_overrides = {}
    if args.model_type is not None:
        transformer_overrides["return_model_type"] = args.model_type
    if args.transformer_d_model is not None:
        transformer_overrides["return_transformer_d_model"] = args.transformer_d_model
    if args.transformer_heads is not None:
        transformer_overrides["return_transformer_heads"] = args.transformer_heads
    if args.transformer_layers is not None:
        transformer_overrides["return_transformer_layers"] = args.transformer_layers
    if args.transformer_patch_len is not None:
        transformer_overrides["return_transformer_patch_len"] = args.transformer_patch_len
    if args.cnn_filters is not None:
        transformer_overrides["return_cnn_filters"] = args.cnn_filters
    if args.cnn_kernel_size is not None:
        transformer_overrides["return_cnn_kernel_size"] = args.cnn_kernel_size
    if transformer_overrides:
        config = replace(config, **transformer_overrides)

    summary = train_return_only(config, return_params_source=str(args.return_params_json) if args.return_params_json else None)
    if not args.no_forecast:
        run_forecast_pipeline(config=config, train=False)

    print("Return-only update completed.")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
