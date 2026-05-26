"""Scientific configuration for the MASI-only backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AUTO_START_OLLAMA = _env_bool("AUTO_START_OLLAMA", True)
OLLAMA_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 4096)
OLLAMA_NUM_PREDICT = _env_int("OLLAMA_NUM_PREDICT", 1200)
OLLAMA_NUM_GPU = _env_int("OLLAMA_NUM_GPU", 999)
OLLAMA_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.1)
OLLAMA_TOP_P = _env_float("OLLAMA_TOP_P", 0.85)

# Vector RAG is the default production retrieval backend.
RAG_RETRIEVER_BACKEND = os.getenv("RAG_RETRIEVER_BACKEND", "chroma")
WARM_RAG_ON_STARTUP = _env_bool("WARM_RAG_ON_STARTUP", True)
DASHBOARD_CONTEXT_BACKEND = os.getenv("DASHBOARD_CONTEXT_BACKEND", "fast")


@dataclass(frozen=True)
class ForecastConfig:
    alpha: float = 0.05
    train_window: int = 4000
    test_window_ratio: float = 786 / 4000
    val_fraction_within_train: float = 0.15
    seq_len: int = 30

    var_feature_cols: Sequence[str] = field(
        default_factory=lambda: (
            "masi_log_return",
            "masi_log_return_std_5",
            "masi_log_return_std_21",
            "masi_log_return_mean_21",
        )
    )

    return_feature_cols: Sequence[str] = field(
        default_factory=lambda: (
            "masi_log_return",
            "masi_log_return_std_5",
            "masi_log_return_std_21",
            "masi_log_return_mean_21",
            "masi_price_range_pct",
            "masi_log_open_close",
            "masi_log_high_low",
            # --- EGARCH-LSTM hybrid features (jobs/compute_egarch_features.py) ---
            "masi_egarch_vol",         # σ_t  : conditional vol, EGARCH(2,1,1), causal
            "masi_egarch_std_resid",   # z_t = r_t/σ_t : heteroskedasticity-cleaned
            "masi_egarch_vol_ratio",   # σ_t / rolling_21(σ) : relative regime
        )
    )
    date_col: str = "date"
    target_col: str = "target_return"
    close_col: str = "masi_close"
    return_col: str = "masi_log_return"

    lstm_hidden_1: int = 128
    lstm_hidden_2: int = 64
    dense_hidden: int = 32
    dropout: float = 0.1
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    return_lstm_hidden_1: int | None = None
    return_lstm_hidden_2: int | None = None
    return_dense_hidden: int | None = None
    return_dropout: float | None = None
    return_batch_size: int | None = None
    return_lr: float | None = None
    return_weight_decay: float | None = None
    return_corr_weight: float = 1.0
    return_var_weight: float = 2.0
    return_model_type: str = "lstm"
    return_transformer_d_model: int = 64
    return_transformer_heads: int = 4
    return_transformer_layers: int = 2
    return_transformer_ff_mult: int = 2
    return_transformer_patch_len: int = 5
    return_cnn_filters: int = 32
    return_cnn_kernel_size: int = 3
    epochs: int = 80
    patience: int = 30

    es_min_violations: int = 20
    es_shortfall_floor: float = 1e-6
    es_ridge_alpha: float = 0.001

    hmm_n_states: int = 3
    seed: int = 42
    model_name: str = "masi_lstm_var_es_hmm"
    model_version: str = "0.1.0"
    horizons: tuple[int, ...] = (1, 10, 25)


DEFAULT_CONFIG = ForecastConfig()
