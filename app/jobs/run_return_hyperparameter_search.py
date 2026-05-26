"""Run a leakage-safe Optuna search for the return model only.

This search never retrains or evaluates VaR/ES for selection. It uses the
chronological train block for fitting, the validation block for Optuna's
objective, and the test block only as an audit report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from backend.core.config import DEFAULT_CONFIG, ForecastConfig
from backend.core.paths import MASTER_DATASET_PATH, METADATA_PATH, REPORT_DIR
from jobs.hyperparameter_config import save_best_return_hyperparameters
from ml.training.deep_models import predict_model, train_return_model
from ml.training.return_calibration import apply_return_calibration, fit_return_calibration
from ml.utils.data_loading import dataset_version, load_master_dataset
from ml.utils.diagnostics import summarize_loss_history
from ml.utils.preprocessing import (
    build_model_frame,
    chronological_train_test_split,
    fit_feature_scaler,
    transform_features,
)
from ml.utils.sequences import make_sequences_from_block, make_sequences_with_context


SEARCH_REPORT_DIR = REPORT_DIR / "hyperparameter_search"
MIN_DIRECTIONAL_ACCURACY = 0.48


def _base_config_from_metadata(epochs: int, patience: int, model_type: str) -> ForecastConfig:
    overrides: dict[str, Any] = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(ForecastConfig)}
        overrides = {key: metadata[key] for key in allowed if key in metadata}
    overrides.update({"epochs": epochs, "patience": patience, "return_model_type": model_type})
    return replace(DEFAULT_CONFIG, **overrides)


def _suggest_return_params(trial: Any, quick: bool, model_type: str) -> dict[str, Any]:
    if model_type == "patch_transformer":
        params = {
            "return_model_type": "patch_transformer",
            "return_transformer_d_model": trial.suggest_categorical(
                "return_transformer_d_model",
                [32, 64] if quick else [32, 64, 96],
            ),
            "return_transformer_heads": trial.suggest_categorical("return_transformer_heads", [2, 4]),
            "return_transformer_layers": trial.suggest_categorical(
                "return_transformer_layers",
                [1, 2] if quick else [1, 2, 3],
            ),
            "return_transformer_patch_len": trial.suggest_categorical(
                "return_transformer_patch_len",
                [4, 5] if quick else [2, 4, 5, 10],
            ),
            "return_dropout": trial.suggest_float("return_dropout", 0.05, 0.35),
            "return_lr": trial.suggest_float("return_lr", 5e-5, 2e-3, log=True),
            "return_batch_size": trial.suggest_categorical("return_batch_size", [32, 64, 128]),
            "return_weight_decay": trial.suggest_float("return_weight_decay", 1e-7, 1e-3, log=True),
            "return_corr_weight": trial.suggest_float("return_corr_weight", 0.0, 3.0),
            "return_var_weight": trial.suggest_float("return_var_weight", 0.0, 4.0),
        }
        if params["return_transformer_d_model"] % params["return_transformer_heads"] != 0:
            params["return_transformer_heads"] = 2
        return params

    if quick:
        return {
            "return_model_type": "lstm",
            "return_lstm_hidden_1": trial.suggest_categorical("return_lstm_hidden_1", [32, 64, 96]),
            "return_lstm_hidden_2": trial.suggest_categorical("return_lstm_hidden_2", [16, 32, 48]),
            "return_dense_hidden": trial.suggest_categorical("return_dense_hidden", [16, 32]),
            "return_dropout": trial.suggest_float("return_dropout", 0.05, 0.3, step=0.05),
            "return_lr": trial.suggest_categorical("return_lr", [3e-4, 5e-4, 1e-3]),
            "return_batch_size": trial.suggest_categorical("return_batch_size", [32, 64]),
            "return_corr_weight": trial.suggest_float("return_corr_weight", 0.0, 2.0, step=0.5),
            "return_var_weight": trial.suggest_float("return_var_weight", 0.5, 3.0, step=0.5),
        }

    return {
        "return_model_type": "lstm",
        "return_lstm_hidden_1": trial.suggest_categorical("return_lstm_hidden_1", [32, 64, 96, 128]),
        "return_lstm_hidden_2": trial.suggest_categorical("return_lstm_hidden_2", [16, 32, 48, 64]),
        "return_dense_hidden": trial.suggest_categorical("return_dense_hidden", [16, 32, 48]),
        "return_dropout": trial.suggest_float("return_dropout", 0.02, 0.4),
        "return_lr": trial.suggest_float("return_lr", 1e-4, 3e-3, log=True),
        "return_batch_size": trial.suggest_categorical("return_batch_size", [16, 32, 64, 128]),
        "return_weight_decay": trial.suggest_float("return_weight_decay", 1e-7, 1e-3, log=True),
        "return_corr_weight": trial.suggest_float("return_corr_weight", 0.0, 3.0),
        "return_var_weight": trial.suggest_float("return_var_weight", 0.0, 4.0),
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    value = float(np.corrcoef(y_true, y_pred)[0, 1])
    return 0.0 if np.isnan(value) else value


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _prediction_std_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.std(y_true))
    if denom <= 0:
        return 0.0
    return float(np.std(y_pred) / denom)


def _score_metrics(metrics: dict[str, float]) -> float:
    """Lower is better; validation-only objective."""

    corr_penalty = max(0.0, 1.0 - metrics["val_corr"])
    direction_penalty = max(0.0, 0.52 - metrics["val_directional_accuracy"])
    variance_penalty = abs(metrics["val_prediction_std_ratio"] - 1.0)
    flat_penalty = max(0.0, 0.70 - metrics["val_prediction_std_ratio"])
    return (
        metrics["val_rmse"]
        + 0.0060 * corr_penalty
        + 0.0020 * direction_penalty
        + 0.0080 * variance_penalty
        + 0.0200 * flat_penalty
    )


def _evaluate_predictions(
    prefix: str,
    raw_rows: pd.DataFrame,
    y_innovation: np.ndarray,
    pred_innovation: np.ndarray,
) -> dict[str, float]:
    y_return = raw_rows["target_return"].to_numpy(dtype=float)
    pred_return = (
        raw_rows["target_mean"].to_numpy(dtype=float)
        + raw_rows["target_vol"].to_numpy(dtype=float) * pred_innovation
    )
    return {
        f"{prefix}_rmse": _rmse(y_return, pred_return),
        f"{prefix}_mae": _mae(y_return, pred_return),
        f"{prefix}_corr": _corr(y_return, pred_return),
        f"{prefix}_directional_accuracy": _directional_accuracy(y_return, pred_return),
        f"{prefix}_prediction_std_ratio": _prediction_std_ratio(y_return, pred_return),
        f"{prefix}_innovation_rmse": _rmse(y_innovation, pred_innovation),
    }


def _prepare_return_data(master_df: pd.DataFrame, config: ForecastConfig) -> dict[str, Any]:
    feature_cols = list(config.return_feature_cols)
    model_df = build_model_frame(master_df, config)
    split = chronological_train_test_split(model_df, config)

    scaler = fit_feature_scaler(split.train_raw, feature_cols)
    train_scaled = transform_features(split.train_raw, scaler, feature_cols)
    val_scaled = transform_features(split.val_raw, scaler, feature_cols)
    train_full_scaled = transform_features(split.train_full_raw, scaler, feature_cols)
    test_scaled = transform_features(split.test_raw, scaler, feature_cols)

    X_train, y_train, _ = make_sequences_from_block(
        train_scaled,
        feature_cols,
        "target_innovation",
        config.date_col,
        config.seq_len,
    )
    X_val, y_val, _ = make_sequences_with_context(
        train_scaled,
        val_scaled,
        feature_cols,
        "target_innovation",
        config.date_col,
        config.seq_len,
    )
    X_test, y_test, _ = make_sequences_with_context(
        train_full_scaled,
        test_scaled,
        feature_cols,
        "target_innovation",
        config.date_col,
        config.seq_len,
    )

    return {
        "split": split,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "val_rows": split.val_raw.reset_index(drop=True),
        "X_test": X_test,
        "y_test": y_test,
        "test_rows": split.test_raw.reset_index(drop=True),
    }


def _evaluate_config(
    prepared: dict[str, Any],
    config: ForecastConfig,
    trial_id: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    result = train_return_model(
        prepared["X_train"],
        prepared["y_train"],
        prepared["X_val"],
        prepared["y_val"],
        config,
    )
    val_pred = predict_model(result.model, prepared["X_val"])
    test_pred = predict_model(result.model, prepared["X_test"]) if len(prepared["X_test"]) else np.array([])
    raw_val_return_pred = (
        prepared["val_rows"]["target_mean"].to_numpy(dtype=float)
        + prepared["val_rows"]["target_vol"].to_numpy(dtype=float) * val_pred
    )
    return_calibration = fit_return_calibration(
        prepared["val_rows"]["target_return"].to_numpy(dtype=float),
        raw_val_return_pred,
    )
    val_return_pred = apply_return_calibration(raw_val_return_pred, return_calibration)
    loss_diag = summarize_loss_history(result.history)

    row = {
        "trial_id": trial_id,
        "status": "ok",
        **params,
        "seq_len": config.seq_len,
        "epochs": config.epochs,
        "patience": config.patience,
        "return_best_val_loss": loss_diag.get("best_val_loss"),
        "return_epochs_ran": loss_diag.get("epochs_ran"),
        "return_calibration_intercept": return_calibration.get("intercept"),
        "return_calibration_slope": return_calibration.get("slope"),
        "return_calibration_raw_std_ratio": return_calibration.get("raw_std_ratio"),
        **_evaluate_predictions("val", prepared["val_rows"], prepared["y_val"], val_pred),
    }
    row["val_raw_prediction_std_ratio"] = row["val_prediction_std_ratio"]
    row.update(_evaluate_return_series("val", prepared["val_rows"], val_return_pred))
    if len(test_pred):
        raw_test_return_pred = (
            prepared["test_rows"]["target_mean"].to_numpy(dtype=float)
            + prepared["test_rows"]["target_vol"].to_numpy(dtype=float) * test_pred
        )
        test_return_pred = apply_return_calibration(raw_test_return_pred, return_calibration)
        row.update(_evaluate_predictions("test", prepared["test_rows"], prepared["y_test"], test_pred))
        row["test_raw_prediction_std_ratio"] = row["test_prediction_std_ratio"]
        row.update(_evaluate_return_series("test", prepared["test_rows"], test_return_pred))
    row["selection_score"] = _score_metrics(row)
    return row


def _evaluate_return_series(prefix: str, raw_rows: pd.DataFrame, pred_return: np.ndarray) -> dict[str, float]:
    y_return = raw_rows["target_return"].to_numpy(dtype=float)
    return {
        f"{prefix}_rmse": _rmse(y_return, pred_return),
        f"{prefix}_mae": _mae(y_return, pred_return),
        f"{prefix}_corr": _corr(y_return, pred_return),
        f"{prefix}_directional_accuracy": _directional_accuracy(y_return, pred_return),
        f"{prefix}_prediction_std_ratio": _prediction_std_ratio(y_return, pred_return),
    }


def _acceptance_rejection_reason(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return "no successful return Optuna trial"
    if not np.isfinite(float(row.get("val_rmse", np.nan))):
        return "validation RMSE unavailable"
    if float(row.get("val_directional_accuracy", 0.0)) < MIN_DIRECTIONAL_ACCURACY:
        return (
            f"validation directional accuracy={row.get('val_directional_accuracy'):.3f} "
            f"< {MIN_DIRECTIONAL_ACCURACY:.2f}"
        )
    ratio = float(row.get("val_prediction_std_ratio", 0.0))
    if ratio < 0.70 or ratio > 1.50:
        return f"validation prediction std ratio={ratio:.3f} outside [0.70, 1.50]"
    return None


def run_search(
    max_trials: int,
    epochs: int,
    patience: int,
    quick: bool,
    study_name: str | None = None,
    storage: str | None = None,
    model_type: str = "lstm",
) -> pd.DataFrame:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna n'est pas installe.") from exc

    SEARCH_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    master_df = load_master_dataset(MASTER_DATASET_PATH)
    data_version_value = dataset_version(MASTER_DATASET_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_study_name = study_name or f"masi_return_hyperparameter_search_{timestamp}"
    base_config = _base_config_from_metadata(epochs=epochs, patience=patience, model_type=model_type)
    prepared = _prepare_return_data(master_df, base_config)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_params: dict[str, Any] | None = None

    sampler = optuna.samplers.TPESampler(seed=DEFAULT_CONFIG.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3)
    study = optuna.create_study(
        study_name=resolved_study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bool(storage),
    )

    def objective(trial: Any) -> float:
        nonlocal best_row
        params = _suggest_return_params(trial, quick=quick, model_type=model_type)
        config = replace(base_config, **params)
        print(f"\nReturn trial {trial.number + 1}/{max_trials}: {params}")
        try:
            row = _evaluate_config(prepared, config, trial.number + 1, params)
            print(
                "  ok | val_rmse="
                f"{row['val_rmse']:.6f} | val_corr={row['val_corr']:.4f} "
                f"| val_dir={row['val_directional_accuracy']:.4f} "
                f"| score={row['selection_score']:.6f}"
            )
            rows.append(row)
            trial.set_user_attr("metrics", row)
            trial.report(row["selection_score"], step=0)
            if best_row is None or row["selection_score"] < best_row["selection_score"]:
                best_row = row
            if trial.should_prune():
                raise optuna.TrialPruned()
            return float(row["selection_score"])
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            row = {
                "trial_id": trial.number + 1,
                "status": "failed",
                **params,
                "selection_score": float("nan"),
                "error": str(exc),
            }
            print(f"  failed: {exc}")
            rows.append(row)
            trial.set_user_attr("metrics", row)
            return float("inf")

    study.optimize(objective, n_trials=max_trials, gc_after_trial=True)
    if best_row is not None:
        best_params = dict(study.best_trial.params)

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["status", "selection_score"], na_position="last")
    results_path = SEARCH_REPORT_DIR / f"return_hyperparameter_search_{timestamp}.csv"
    summary_path = SEARCH_REPORT_DIR / f"return_hyperparameter_search_{timestamp}.json"
    results.to_csv(results_path, index=False)

    rejection_reason = _acceptance_rejection_reason(best_row)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "master_dataset": str(MASTER_DATASET_PATH),
        "data_version": data_version_value,
        "optimizer": "optuna",
        "selection_objective": (
            f"return {model_type} validation-only objective; chronological train/validation split; "
            "test metrics are audit-only and never used for trial selection"
        ),
        "max_trials": max_trials,
        "epochs_per_trial": epochs,
        "patience_per_trial": patience,
        "study_name": resolved_study_name,
        "storage": storage,
        "best_trial": best_row,
        "best_params": best_params,
        "best_trial_accepted_for_production": rejection_reason is None,
        "best_trial_rejection_reason": rejection_reason,
        "results_csv": str(results_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    best_params_path = None
    if best_params is not None:
        best_params_path = save_best_return_hyperparameters(
            best_params,
            summary_path,
            best_row,
            accepted=rejection_reason is None,
            rejection_reason=rejection_reason,
        )
        payload["best_params_path"] = str(best_params_path)
        summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\nReturn hyperparameter search completed.")
    print(f"Results CSV: {results_path}")
    print(f"Summary JSON: {summary_path}")
    if best_params_path is not None:
        print(f"Best return hyperparameters: {best_params_path}")
    if best_row is not None:
        print("\nBest return trial:")
        print(pd.DataFrame([best_row]).to_string(index=False))
        if rejection_reason is not None:
            print("\nBest return trial NOT accepted for production:")
            print(rejection_reason)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe return hyperparameter search with Optuna.")
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--model-type", choices=("lstm", "patch_transformer"), default="lstm")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None)
    args = parser.parse_args()

    run_search(
        max_trials=args.max_trials,
        epochs=args.epochs,
        patience=args.patience,
        quick=args.quick,
        study_name=args.study_name,
        storage=args.storage,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
