"""Reusable functions extracted from the research notebooks.

The notebooks keep the narrative and result cells; this module keeps shared code.
"""

import numpy as np

import pandas as pd

from scipy.stats import chi2, ttest_1samp

ALPHA = 0.05

EPS = 1e-12

def clip_probability(p, eps=EPS):
    return np.clip(p, eps, 1.0 - eps)

def safe_loglik_bernoulli(successes, trials, prob):
    prob = clip_probability(prob)
    return successes * np.log(prob) + (trials - successes) * np.log(1.0 - prob)

def validate_inputs(realized_returns, var_forecasts, es_forecasts=None):
    data = pd.DataFrame({
        "realized_return": pd.Series(realized_returns),
        "var_forecast": pd.Series(var_forecasts),
    })

    if es_forecasts is not None:
        data["es_forecast"] = pd.Series(es_forecasts)

    data = data.dropna().copy()

    if data.empty:
        raise ValueError("Aucune observation valide aprÃ¨s suppression des NaN.")

    if es_forecasts is not None:
        invalid_es = (data["es_forecast"] > data["var_forecast"]).sum()
        if invalid_es > 0:
            raise ValueError(
                f"IncohÃ©rence dÃ©tectÃ©e : {invalid_es} observations ont ES > VaR. "
                "Sur l'Ã©chelle des rendements, on attend gÃ©nÃ©ralement ES <= VaR."
            )

    return data

def kupiec_pof_test(violations, alpha=ALPHA):
    hits = np.asarray(violations, dtype=int)
    n = len(hits)

    if n == 0:
        return np.nan, np.nan

    x = int(hits.sum())
    p_hat = x / n

    loglik_null = safe_loglik_bernoulli(x, n, alpha)
    loglik_alt = safe_loglik_bernoulli(x, n, p_hat)

    lr_pof = -2.0 * (loglik_null - loglik_alt)
    p_value = 1.0 - chi2.cdf(lr_pof, df=1)

    return lr_pof, p_value

def christoffersen_independence_test(violations):
    hits = np.asarray(violations, dtype=int)
    n = len(hits)

    if n < 2:
        return np.nan, np.nan

    prev_hits = hits[:-1]
    curr_hits = hits[1:]

    n00 = int(((prev_hits == 0) & (curr_hits == 0)).sum())
    n01 = int(((prev_hits == 0) & (curr_hits == 1)).sum())
    n10 = int(((prev_hits == 1) & (curr_hits == 0)).sum())
    n11 = int(((prev_hits == 1) & (curr_hits == 1)).sum())

    total_0 = n00 + n01
    total_1 = n10 + n11
    total_all = n00 + n01 + n10 + n11

    if total_all == 0 or total_0 == 0 or total_1 == 0:
        return np.nan, np.nan

    pi0 = n01 / total_0
    pi1 = n11 / total_1
    pi = (n01 + n11) / total_all

    loglik_null = (
        n00 * np.log(clip_probability(1.0 - pi))
        + n01 * np.log(clip_probability(pi))
        + n10 * np.log(clip_probability(1.0 - pi))
        + n11 * np.log(clip_probability(pi))
    )

    loglik_alt = (
        n00 * np.log(clip_probability(1.0 - pi0))
        + n01 * np.log(clip_probability(pi0))
        + n10 * np.log(clip_probability(1.0 - pi1))
        + n11 * np.log(clip_probability(pi1))
    )

    lr_ind = -2.0 * (loglik_null - loglik_alt)
    p_value = 1.0 - chi2.cdf(lr_ind, df=1)

    return lr_ind, p_value

def es_tail_calibration_test(realized_returns, var_forecasts, es_forecasts):
    rr = np.asarray(realized_returns, dtype=float)
    vf = np.asarray(var_forecasts, dtype=float)
    ef = np.asarray(es_forecasts, dtype=float)

    tail_mask = rr < vf
    tail_residuals = rr[tail_mask] - ef[tail_mask]

    n_tail = int(tail_mask.sum())

    if n_tail < 2:
        return n_tail, np.nan, np.nan, np.nan

    test_stat, p_value = ttest_1samp(tail_residuals, popmean=0.0, nan_policy="omit")
    tail_mean = float(np.nanmean(tail_residuals))

    return n_tail, test_stat, p_value, tail_mean

def backtest_var_es_model(model_data, alpha=ALPHA):
    required_columns = ["realized_return", "var_5pct", "es_5pct"]
    missing_cols = [c for c in required_columns if c not in model_data.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    clean = validate_inputs(
        realized_returns=model_data["realized_return"],
        var_forecasts=model_data["var_5pct"],
        es_forecasts=model_data["es_5pct"],
    )

    realized_returns = clean["realized_return"]
    var_forecasts = clean["var_forecast"]
    es_forecasts = clean["es_forecast"]

    violations = (realized_returns < var_forecasts).astype(int)

    n_obs = int(len(violations))
    n_var_violations = int(violations.sum())
    violation_rate = n_var_violations / n_obs if n_obs > 0 else np.nan

    kupiec_stat, kupiec_p_value = kupiec_pof_test(violations, alpha=alpha)
    ind_stat, ind_p_value = christoffersen_independence_test(violations)

    n_es_tail, es_stat, es_p_value, es_tail_mean = es_tail_calibration_test(
        realized_returns=realized_returns,
        var_forecasts=var_forecasts,
        es_forecasts=es_forecasts,
    )

    return {
        "n_observations": n_obs,
        "n_var_violations": n_var_violations,
        "violation_rate": violation_rate,
        "expected_violation_rate": alpha,
        "kupiec_pof_stat": kupiec_stat,
        "kupiec_pof_p_value": kupiec_p_value,
        "christoffersen_independence_stat": ind_stat,
        "christoffersen_independence_p_value": ind_p_value,
        "n_es_tail_observations": n_es_tail,
        "es_tail_calibration_stat": es_stat,
        "es_tail_calibration_p_value": es_p_value,
        "es_tail_residual_mean": es_tail_mean,
    }

def backtest_forecast_results(forecast_results, alpha=ALPHA):
    required_columns = ["model_family", "realized_return", "var_5pct", "es_5pct"]
    missing_cols = [c for c in required_columns if c not in forecast_results.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans forecast_results : {missing_cols}")

    backtest_rows = []

    for model_family, model_data in forecast_results.groupby("model_family"):
        summary = backtest_var_es_model(model_data=model_data, alpha=alpha)
        summary["model_family"] = model_family
        backtest_rows.append(summary)

    result = pd.DataFrame(backtest_rows)

    ordered_cols = [
        "model_family",
        "n_observations",
        "n_var_violations",
        "violation_rate",
        "expected_violation_rate",
        "kupiec_pof_stat",
        "kupiec_pof_p_value",
        "christoffersen_independence_stat",
        "christoffersen_independence_p_value",
        "n_es_tail_observations",
        "es_tail_calibration_stat",
        "es_tail_calibration_p_value",
        "es_tail_residual_mean",
    ]

    return result[ordered_cols].sort_values("model_family").reset_index(drop=True)
