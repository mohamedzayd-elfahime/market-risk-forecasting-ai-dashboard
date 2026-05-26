"""EGARCH-based return candidates using only causal innovation information."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_egarch_innovation_ar1(train_raw: pd.DataFrame) -> dict[str, float]:
    """Fit z_{t+1} = intercept + phi * z_t on the training block only."""

    required = {"masi_egarch_std_resid", "target_innovation"}
    if not required.issubset(train_raw.columns):
        return {"intercept": 0.0, "phi": 0.0}

    z_t = train_raw["masi_egarch_std_resid"].to_numpy(dtype=float)
    z_next = train_raw["target_innovation"].to_numpy(dtype=float)
    mask = np.isfinite(z_t) & np.isfinite(z_next)
    if mask.sum() < 20 or np.std(z_t[mask]) <= 1e-12:
        return {"intercept": 0.0, "phi": 0.0}

    phi, intercept = np.polyfit(z_t[mask], z_next[mask], deg=1)
    return {
        "intercept": float(np.clip(intercept, -0.5, 0.5)),
        "phi": float(np.clip(phi, -0.75, 0.75)),
    }


def predict_egarch_return_candidates(
    df_block: pd.DataFrame,
    ar1_bundle: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Return causal EGARCH candidates aligned to df_block rows."""

    mean_next = df_block["target_mean"].to_numpy(dtype=float)
    vol_next = df_block["target_vol"].to_numpy(dtype=float)
    z_t = df_block["masi_egarch_std_resid"].to_numpy(dtype=float)
    z_t = np.nan_to_num(z_t, nan=0.0, posinf=0.0, neginf=0.0)

    candidates = {
        "egarch_mean": mean_next,
        "egarch_innovation_t": mean_next + vol_next * z_t,
    }
    if ar1_bundle is not None:
        intercept = float(ar1_bundle.get("intercept", 0.0))
        phi = float(ar1_bundle.get("phi", 0.0))
        candidates["egarch_innovation_ar1"] = mean_next + vol_next * (intercept + phi * z_t)
    return candidates


def predict_latest_egarch_return_candidates(
    latest_raw: pd.DataFrame,
    ar1_bundle: dict[str, float] | None = None,
) -> dict[str, float]:
    """Inference variant for the latest row with inference-frame column names."""

    mean_next = float(latest_raw["masi_egarch_mean_next"].iloc[0])
    vol_next = float(latest_raw["masi_egarch_vol_next"].iloc[0])
    z_t = float(latest_raw["masi_egarch_std_resid"].iloc[0])
    if not np.isfinite(z_t):
        z_t = 0.0

    candidates = {
        "egarch_mean": mean_next,
        "egarch_innovation_t": mean_next + vol_next * z_t,
    }
    if ar1_bundle is not None:
        intercept = float(ar1_bundle.get("intercept", 0.0))
        phi = float(ar1_bundle.get("phi", 0.0))
        candidates["egarch_innovation_ar1"] = mean_next + vol_next * (intercept + phi * z_t)
    return candidates
