"""Progressive statistical and economic backtesting from the forecast log."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_1samp


def _clip_probability(p: float, eps: float = 1e-12) -> float:
    return float(np.clip(p, eps, 1.0 - eps))


def _safe_loglik_bernoulli(x: int, n: int, p: float) -> float:
    p = _clip_probability(p)
    return x * np.log(p) + (n - x) * np.log(1.0 - p)


def kupiec_pof_test(violations, alpha: float = 0.05) -> tuple[float, float]:
    hits = np.asarray(violations, dtype=int)
    n = len(hits)
    if n == 0:
        return np.nan, np.nan
    x = int(hits.sum())
    p_hat = x / n
    lr_pof = -2.0 * (_safe_loglik_bernoulli(x, n, alpha) - _safe_loglik_bernoulli(x, n, p_hat))
    return float(lr_pof), float(1.0 - chi2.cdf(lr_pof, df=1))


def christoffersen_independence_test(violations) -> tuple[float, float]:
    hits = np.asarray(violations, dtype=int)
    if len(hits) < 2:
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
        n00 * np.log(_clip_probability(1.0 - pi))
        + n01 * np.log(_clip_probability(pi))
        + n10 * np.log(_clip_probability(1.0 - pi))
        + n11 * np.log(_clip_probability(pi))
    )
    loglik_alt = (
        n00 * np.log(_clip_probability(1.0 - pi0))
        + n01 * np.log(_clip_probability(pi0))
        + n10 * np.log(_clip_probability(1.0 - pi1))
        + n11 * np.log(_clip_probability(pi1))
    )

    lr_ind = -2.0 * (loglik_null - loglik_alt)
    return float(lr_ind), float(1.0 - chi2.cdf(lr_ind, df=1))


def christoffersen_conditional_coverage_test(violations, alpha: float = 0.05) -> tuple[float, float]:
    kupiec_stat, _ = kupiec_pof_test(violations, alpha)
    independence_stat, _ = christoffersen_independence_test(violations)
    if np.isnan(kupiec_stat) or np.isnan(independence_stat):
        return np.nan, np.nan
    cc_stat = kupiec_stat + independence_stat
    return float(cc_stat), float(1.0 - chi2.cdf(cc_stat, df=2))


def es_tail_calibration(realized_returns, var_forecasts, es_forecasts) -> tuple[int, float, float, float]:
    rr = np.asarray(realized_returns, dtype=float)
    vf = np.asarray(var_forecasts, dtype=float)
    ef = np.asarray(es_forecasts, dtype=float)
    tail = rr < vf
    n_tail = int(tail.sum())
    if n_tail < 2:
        return n_tail, np.nan, np.nan, np.nan
    residuals = rr[tail] - ef[tail]
    stat, p_value = ttest_1samp(residuals, popmean=0.0, nan_policy="omit")
    return n_tail, float(stat), float(p_value), float(np.nanmean(residuals))


def compute_var_budget_weights(var_forecast, window: int = 60, quantile: float = 0.7, cap: float = 1.0) -> pd.Series:
    var_abs = pd.Series(np.abs(pd.to_numeric(var_forecast, errors="coerce")))
    budget = var_abs.rolling(window=window, min_periods=max(5, window // 4)).quantile(quantile)
    weights = (budget / var_abs.replace(0, np.nan)).clip(lower=0.0, upper=cap)
    return weights.fillna(1.0)


def _compute_economic_paths(
    realized_return: pd.Series,
    var_pred: pd.Series,
    initial_wealth: float = 1.0,
    transaction_cost: float = 0.0,
) -> dict[str, pd.Series]:
    asset_simple = pd.Series(np.expm1(pd.to_numeric(realized_return, errors="coerce")), index=realized_return.index)

    buy_hold_wealth = initial_wealth * (1.0 + asset_simple).cumprod()
    buy_hold_drawdown = buy_hold_wealth / buy_hold_wealth.cummax() - 1.0

    weights_signal = compute_var_budget_weights(var_pred)
    weights_signal.index = realized_return.index
    weights_used = weights_signal.shift(1).fillna(1.0)
    turnover = weights_used.diff().abs().fillna(0.0)

    strategy_simple = weights_used * asset_simple - transaction_cost * turnover
    strategy_wealth = initial_wealth * (1.0 + strategy_simple).cumprod()
    strategy_drawdown = strategy_wealth / strategy_wealth.cummax() - 1.0

    return {
        "asset_simple": asset_simple,
        "buy_hold_wealth": buy_hold_wealth,
        "buy_hold_drawdown": buy_hold_drawdown,
        "weights_signal": weights_signal,
        "weights_used": weights_used,
        "turnover": turnover,
        "strategy_simple": strategy_simple,
        "strategy_wealth": strategy_wealth,
        "strategy_drawdown": strategy_drawdown,
    }


def enrich_backtest_columns(
    log_df: pd.DataFrame,
    initial_wealth: float = 1.0,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    out = log_df.copy()
    ready = out["realized_return"].notna()
    out.loc[ready, "var_breach"] = (
        pd.to_numeric(out.loc[ready, "realized_return"]) < pd.to_numeric(out.loc[ready, "var_forecast"])
    ).astype(int)
    out.loc[ready, "es_calibration_error"] = (
        pd.to_numeric(out.loc[ready, "realized_return"]) - pd.to_numeric(out.loc[ready, "es_forecast"])
    )

    one_day = out["horizon"].astype(str).eq("1") & ready
    if one_day.any():
        paths = _compute_economic_paths(
            realized_return=out.loc[one_day, "realized_return"],
            var_pred=out.loc[one_day, "var_forecast"],
            initial_wealth=initial_wealth,
            transaction_cost=transaction_cost,
        )
        out.loc[one_day, "weight"] = paths["weights_used"].to_numpy()
        out.loc[one_day, "realized_strategy_return"] = paths["strategy_simple"].to_numpy()
        out.loc[one_day, "updated_wealth"] = paths["strategy_wealth"].to_numpy()
        out.loc[one_day, "updated_drawdown"] = paths["strategy_drawdown"].to_numpy()
    return out


def summarize_statistical_backtest(log_df: pd.DataFrame, alpha: float = 0.05) -> dict[str, float]:
    data = log_df[(log_df["horizon"].astype(str) == "1") & log_df["realized_return"].notna()].copy()
    if data.empty:
        return {"n_observations": 0}

    violations = pd.to_numeric(data["var_breach"], errors="coerce").fillna(0).astype(int)
    kupiec_stat, kupiec_p_value = kupiec_pof_test(violations, alpha)
    ind_stat, ind_p_value = christoffersen_independence_test(violations)
    cc_stat, cc_p_value = christoffersen_conditional_coverage_test(violations, alpha)
    n_tail, es_stat, es_p_value, es_mean = es_tail_calibration(
        data["realized_return"], data["var_forecast"], data["es_forecast"]
    )
    return {
        "n_observations": int(len(data)),
        "n_var_breaches": int(violations.sum()),
        "violation_rate": float(violations.mean()),
        "expected_violation_rate": alpha,
        "kupiec_pof_stat": kupiec_stat,
        "kupiec_pof_p_value": kupiec_p_value,
        "christoffersen_independence_stat": ind_stat,
        "christoffersen_independence_p_value": ind_p_value,
        "christoffersen_cc_stat": cc_stat,
        "christoffersen_cc_p_value": cc_p_value,
        "n_es_tail_observations": n_tail,
        "es_tail_calibration_stat": es_stat,
        "es_tail_calibration_p_value": es_p_value,
        "es_tail_residual_mean": es_mean,
    }


def summarize_test_predictions(test_predictions: pd.DataFrame, alpha: float = 0.05) -> dict[str, float]:
    data = test_predictions.dropna(subset=["realized_return", "var_pred", "es_pred"]).copy()
    if data.empty:
        return {"n_observations": 0}

    violations = (data["realized_return"] < data["var_pred"]).astype(int)
    kupiec_stat, kupiec_p_value = kupiec_pof_test(violations, alpha)
    ind_stat, ind_p_value = christoffersen_independence_test(violations)
    cc_stat, cc_p_value = christoffersen_conditional_coverage_test(violations, alpha)
    n_tail, es_stat, es_p_value, es_mean = es_tail_calibration(
        data["realized_return"], data["var_pred"], data["es_pred"]
    )
    pinball = np.maximum(
        alpha * (data["realized_return"] - data["var_pred"]),
        (alpha - 1.0) * (data["realized_return"] - data["var_pred"]),
    )
    return {
        "n_observations": int(len(data)),
        "n_var_breaches": int(violations.sum()),
        "violation_rate": float(violations.mean()),
        "expected_violation_rate": alpha,
        "kupiec_pof_stat": kupiec_stat,
        "kupiec_pof_p_value": kupiec_p_value,
        "christoffersen_independence_stat": ind_stat,
        "christoffersen_independence_p_value": ind_p_value,
        "christoffersen_cc_stat": cc_stat,
        "christoffersen_cc_p_value": cc_p_value,
        "n_es_tail_observations": n_tail,
        "es_tail_calibration_stat": es_stat,
        "es_tail_calibration_p_value": es_p_value,
        "es_tail_residual_mean": es_mean,
        "mean_pinball_loss": float(pinball.mean()),
        "mean_return_prediction_error": float((data["realized_return"] - data["return_pred"]).mean()),
        "rmse_return_prediction": float(np.sqrt(np.mean((data["realized_return"] - data["return_pred"]) ** 2))),
    }


def compute_wealth_curves(
    test_predictions: pd.DataFrame,
    initial_wealth: float = 1.0,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Attach Buy&Hold and Risk-Managed wealth/drawdown columns to test_predictions."""
    data = test_predictions.copy()
    valid = data["realized_return"].notna() & data["var_pred"].notna()
    if valid.any():
        paths = _compute_economic_paths(
            realized_return=data.loc[valid, "realized_return"],
            var_pred=data.loc[valid, "var_pred"],
            initial_wealth=initial_wealth,
            transaction_cost=transaction_cost,
        )
        data.loc[valid, "buy_hold_wealth"] = paths["buy_hold_wealth"].values
        data.loc[valid, "buy_hold_drawdown"] = paths["buy_hold_drawdown"].values
        data.loc[valid, "strategy_wealth"] = paths["strategy_wealth"].values
        data.loc[valid, "strategy_drawdown"] = paths["strategy_drawdown"].values
        data.loc[valid, "strategy_weight"] = paths["weights_used"].values
    return data


def summarize_economic_backtest(
    test_predictions: pd.DataFrame,
    initial_wealth: float = 1.0,
    transaction_cost: float = 0.0,
) -> dict[str, float]:
    data = test_predictions.dropna(subset=["realized_return", "var_pred"]).copy().reset_index(drop=True)
    if data.empty:
        return {"n_observations": 0}

    paths = _compute_economic_paths(
        realized_return=data["realized_return"],
        var_pred=data["var_pred"],
        initial_wealth=initial_wealth,
        transaction_cost=transaction_cost,
    )

    asset_simple = paths["asset_simple"]
    weights_used = paths["weights_used"]
    turnover = paths["turnover"]
    strategy_simple = paths["strategy_simple"]
    wealth = paths["strategy_wealth"]
    drawdown = paths["strategy_drawdown"]
    buy_hold_wealth = paths["buy_hold_wealth"]
    buy_hold_dd = paths["buy_hold_drawdown"]

    periods_per_year = 252
    vol_strat = strategy_simple.std(ddof=1)
    sharpe_strat = np.sqrt(periods_per_year) * strategy_simple.mean() / vol_strat if vol_strat and vol_strat > 0 else np.nan
    vol_bh = asset_simple.std(ddof=1)
    sharpe_bh = np.sqrt(periods_per_year) * asset_simple.mean() / vol_bh if vol_bh and vol_bh > 0 else np.nan

    return {
        "n_observations": int(len(data)),
        "final_wealth": float(wealth.iloc[-1]),
        "buy_hold_final_wealth": float(buy_hold_wealth.iloc[-1]),
        "max_drawdown": float(drawdown.min()),
        "buy_hold_max_drawdown": float(buy_hold_dd.min()),
        "annualized_sharpe": float(sharpe_strat),
        "buy_hold_sharpe": float(sharpe_bh),
        "average_weight_used": float(weights_used.mean()),
        "average_turnover": float(turnover.mean()),
        "transaction_cost": float(transaction_cost),
        "weight_rule": "w_t uses a 1-period lag of a rolling VaR-budget weight to avoid look-ahead bias",
    }
