"""Utilities for loading the latest Optuna-selected hyperparameters."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from backend.core.config import DEFAULT_CONFIG, ForecastConfig
from backend.core.paths import REPORT_DIR


SEARCH_REPORT_DIR = REPORT_DIR / "hyperparameter_search"
BEST_PARAMS_PATH = SEARCH_REPORT_DIR / "best_hyperparameters.json"
BEST_RETURN_PARAMS_PATH = SEARCH_REPORT_DIR / "best_return_hyperparameters.json"

TUNABLE_CONFIG_KEYS = {
    "seq_len",
    "lstm_hidden_1",
    "lstm_hidden_2",
    "dense_hidden",
    "dropout",
    "lr",
    "batch_size",
    "es_ridge_alpha",
}

TUNABLE_RETURN_CONFIG_KEYS = {
    "return_lstm_hidden_1",
    "return_lstm_hidden_2",
    "return_dense_hidden",
    "return_dropout",
    "return_lr",
    "return_batch_size",
    "return_weight_decay",
    "return_corr_weight",
    "return_var_weight",
    "return_model_type",
    "return_transformer_d_model",
    "return_transformer_heads",
    "return_transformer_layers",
    "return_transformer_ff_mult",
    "return_transformer_patch_len",
}


def save_best_hyperparameters(
    best_params: dict[str, Any],
    source_summary_path: Path,
    best_trial: dict[str, Any] | None = None,
    accepted: bool = True,
    rejection_reason: str | None = None,
) -> Path:
    SEARCH_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_summary": str(source_summary_path),
        "best_params": _filter_tunable_params(best_params),
        "best_trial": best_trial,
        "accepted_for_production": accepted,
        "rejection_reason": rejection_reason,
    }
    BEST_PARAMS_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return BEST_PARAMS_PATH


def save_best_return_hyperparameters(
    best_params: dict[str, Any],
    source_summary_path: Path,
    best_trial: dict[str, Any] | None = None,
    accepted: bool = True,
    rejection_reason: str | None = None,
) -> Path:
    SEARCH_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_summary": str(source_summary_path),
        "best_params": _filter_return_tunable_params(best_params),
        "best_trial": best_trial,
        "accepted_for_production": accepted,
        "rejection_reason": rejection_reason,
        "selection_objective": "validation-only return forecast objective; test metrics are audit-only",
    }
    BEST_RETURN_PARAMS_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return BEST_RETURN_PARAMS_PATH


def load_best_hyperparameters(path: Path = BEST_PARAMS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No Optuna best-parameter file found at {path}. Run Optuna tuning first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"No best_params object found in {path}.")
    warning = payload.get("selection_warning")
    if warning:
        raise ValueError(
            f"The saved Optuna parameters are marked as stale: {warning}"
        )
    if payload.get("accepted_for_production") is False:
        reason = payload.get("rejection_reason") or "backtest acceptance criteria failed"
        raise ValueError(
            f"The saved Optuna parameters are not accepted for production: {reason}"
        )
    return _filter_tunable_params(params)


def build_config_from_best_hyperparameters(
    base_config: ForecastConfig = DEFAULT_CONFIG,
    path: Path = BEST_PARAMS_PATH,
    return_path: Path | None = None,
) -> ForecastConfig:
    params = load_best_hyperparameters(path)
    if return_path is not None:
        params.update(load_best_return_hyperparameters(return_path))
    return replace(base_config, **params)


def load_best_return_hyperparameters(path: Path = BEST_RETURN_PARAMS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No return Optuna best-parameter file found at {path}. Run return tuning first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"No best_params object found in {path}.")
    if payload.get("accepted_for_production") is False:
        reason = payload.get("rejection_reason") or "return validation criteria failed"
        raise ValueError(
            f"The saved return Optuna parameters are not accepted for production: {reason}"
        )
    return _filter_return_tunable_params(params)


def build_config_from_best_return_hyperparameters(
    base_config: ForecastConfig = DEFAULT_CONFIG,
    path: Path = BEST_RETURN_PARAMS_PATH,
) -> ForecastConfig:
    return replace(base_config, **load_best_return_hyperparameters(path))


def _filter_tunable_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: params[key]
        for key in TUNABLE_CONFIG_KEYS
        if key in params
    }


def _filter_return_tunable_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: params[key]
        for key in TUNABLE_RETURN_CONFIG_KEYS
        if key in params
    }
