"""Causal EGARCH feature engineering for the MASI forecast dataset."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model


BURN_IN = 500
EGARCH_COLS = [
    "masi_egarch_mean",
    "masi_egarch_vol",
    "masi_egarch_std_resid",
    "masi_egarch_vol_ratio",
    "masi_egarch_mean_next",
    "masi_egarch_vol_next",
]


def compute_egarch_features(
    df: pd.DataFrame,
    return_col: str = "masi_log_return",
    burn_in: int = BURN_IN,
    verbose: bool = False,
) -> pd.DataFrame:
    """Enrich a MASI master dataset with causal EGARCH features."""
    if return_col not in df.columns:
        raise ValueError(f"Missing return column for EGARCH features: {return_col}")

    out = df.copy().sort_values("date").reset_index(drop=True)
    returns_pct = out[return_col].fillna(0).to_numpy(dtype=float) * 100.0

    if burn_in >= len(returns_pct):
        raise ValueError(f"burn_in={burn_in} >= n={len(returns_pct)}")

    if verbose:
        print(f"[EGARCH] Fitting on burn-in ({burn_in} obs)...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        burnin_model = arch_model(
            returns_pct[:burn_in],
            mean="AR",
            lags=1,
            vol="EGARCH",
            p=2,
            o=1,
            q=1,
            dist="t",
            rescale=False,
        )
        burnin_result = burnin_model.fit(
            disp="off",
            show_warning=False,
            options={"maxiter": 500, "ftol": 1e-9},
        )

    if verbose:
        print(f"[EGARCH] Burn-in params:\n{burnin_result.params.round(6).to_string()}")
        print(f"[EGARCH] Applying fixed params to full {len(returns_pct)}-obs series...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_model = arch_model(
            returns_pct,
            mean="AR",
            lags=1,
            vol="EGARCH",
            p=2,
            o=1,
            q=1,
            dist="t",
            rescale=False,
        )
        fixed_result = full_model.fix(burnin_result.params)

    mean_const = float(burnin_result.params.get("Const", 0.0))
    mean_ar1 = float(burnin_result.params.get("y[1]", 0.0))

    cond_mean = (returns_pct - fixed_result.resid) / 100.0
    mean_next = (mean_const + mean_ar1 * returns_pct) / 100.0
    cond_vol = fixed_result.conditional_volatility / 100.0
    std_resid = fixed_result.std_resid

    vol_series = pd.Series(cond_vol)
    vol_ma21 = vol_series.rolling(21, min_periods=5).mean().clip(lower=1e-8)
    vol_ratio = (vol_series / vol_ma21).to_numpy()

    forecast_result = fixed_result.forecast(horizon=1, start=0, reindex=True)
    forecast_var = forecast_result.variance.iloc[:, 0].to_numpy(dtype=float)
    forecast_mean = forecast_result.mean.iloc[:, 0].to_numpy(dtype=float)

    # Row t must contain the one-step-ahead EGARCH forecast made with
    # information available through t. This keeps the innovation target:
    # (r_{t+1} - mu_{t+1|t}) / sigma_{t+1|t}
    # without leaking the realized t+1 return into row t.
    cond_vol_next = np.sqrt(forecast_var) / 100.0
    mean_next = forecast_mean / 100.0
    fallback_mask = ~np.isfinite(cond_vol_next) | (cond_vol_next <= 0)
    cond_vol_next[fallback_mask] = cond_vol[fallback_mask]
    mean_next[~np.isfinite(mean_next)] = ((mean_const + mean_ar1 * returns_pct) / 100.0)[~np.isfinite(mean_next)]

    out["masi_egarch_mean"] = cond_mean
    out["masi_egarch_vol"] = cond_vol
    out["masi_egarch_std_resid"] = std_resid
    out["masi_egarch_vol_ratio"] = vol_ratio
    out["masi_egarch_mean_next"] = mean_next
    out["masi_egarch_vol_next"] = cond_vol_next
    return out
