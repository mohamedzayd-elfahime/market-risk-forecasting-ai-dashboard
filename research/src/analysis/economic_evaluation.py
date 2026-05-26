"""Reusable functions extracted from the research notebooks.

The notebooks keep the narrative and result cells; this module keeps shared code.
"""

import numpy as np

import pandas as pd

def _to_series(x, name=None):
    """Convert input to pandas Series."""
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    if name is not None:
        s.name = name
    return s

def align_inputs(log_returns, weights, rf=0.0):
    """
    Align log returns, weights, and optional risk-free rate on common index.

    Parameters
    ----------
    log_returns : array-like or pd.Series
        Realized asset log returns.
    weights : array-like or pd.Series
        Strategy exposure/weight applied to the risky asset.
        Same length/index as log_returns or alignable.
    rf : float, array-like, or pd.Series, default 0.0
        Risk-free simple return per period.
        If float, assumed constant per period.

    Returns
    -------
    pd.DataFrame
        Columns: ['log_r', 'w', 'rf']
    """
    log_r = _to_series(log_returns, "log_r")
    w = _to_series(weights, "w")

    if np.isscalar(rf):
        rf_s = pd.Series(rf, index=log_r.index, name="rf")
    else:
        rf_s = _to_series(rf, "rf")

    df = pd.concat([log_r, w, rf_s], axis=1).dropna()
    return df

def simple_returns_from_log(log_returns):
    """
    Convert log returns to simple returns.

    Formula
    -------
    simple_return = exp(log_return) - 1
    """
    log_r = _to_series(log_returns, "log_r")
    return np.expm1(log_r)

def strategy_simple_returns(
    log_returns,
    weights,
    rf=0.0,
    transaction_cost=0.0,
    weight_lag=1,
):
    """
    Compute simple strategy returns from realized log returns and weights.

    Parameters
    ----------
    log_returns : array-like or pd.Series
        Realized asset log returns.
    weights : array-like or pd.Series
        Exposure signal. Usually in [0, 1] for long-only risk targeting.
    rf : float, array-like, or pd.Series, default 0.0
        Risk-free simple return per period.
    transaction_cost : float, default 0.0
        Proportional cost per unit turnover.
        Example: 0.001 means 10 bps per unit of absolute change in weight.
    weight_lag : int, default 1
        Number of periods by which weights are lagged.
        Use 1 to avoid look-ahead bias when weight at t is formed using info up to t-1.

    Returns
    -------
    pd.DataFrame
        Columns:
        - asset_simple_return
        - weight_used
        - turnover
        - transaction_cost
        - strategy_simple_return
    """
    df = align_inputs(log_returns, weights, rf=rf)

    asset_simple = np.expm1(df["log_r"])
    w_used = df["w"].shift(weight_lag)

    # Long-only risky asset + cash interpretation
    # strategy return = w * risky simple return + (1 - w) * rf
    strat_simple = w_used * asset_simple + (1.0 - w_used) * df["rf"]

    turnover = w_used.diff().abs().fillna(0.0)
    tc = transaction_cost * turnover

    strat_simple_net = strat_simple - tc

    out = pd.DataFrame(
        {
            "asset_simple_return": asset_simple,
            "weight_used": w_used,
            "turnover": turnover,
            "transaction_cost": tc,
            "strategy_simple_return": strat_simple_net,
        }
    ).dropna()

    return out

def cumulative_wealth(simple_returns, initial_wealth=1.0):
    """
    Compute cumulative wealth path from simple returns.

    Formula
    -------
    W_t = W_0 * prod(1 + R_t)
    """
    r = _to_series(simple_returns, "simple_r").dropna()
    return initial_wealth * (1.0 + r).cumprod()

def drawdown_from_wealth(wealth):
    """
    Compute drawdown series from wealth path.

    Formula
    -------
    DD_t = W_t / max_{s<=t}(W_s) - 1
    """
    w = _to_series(wealth, "wealth").dropna()
    peak = w.cummax()
    dd = w / peak - 1.0
    dd.name = "drawdown"
    return dd

def max_drawdown(wealth):
    """Return maximum drawdown from wealth series."""
    dd = drawdown_from_wealth(wealth)
    return dd.min()

def annualized_return_from_wealth(wealth, periods_per_year=252):
    """
    Annualized geometric return from wealth path.

    Formula
    -------
    (W_T / W_0)^(periods_per_year / n_periods) - 1
    """
    w = _to_series(wealth, "wealth").dropna()
    n = len(w)
    if n < 2:
        return np.nan
    total_growth = w.iloc[-1] / w.iloc[0]
    return total_growth ** (periods_per_year / (n - 1)) - 1.0

def annualized_volatility(simple_returns, periods_per_year=252):
    """
    Annualized volatility from simple returns.
    """
    r = _to_series(simple_returns, "simple_r").dropna()
    if len(r) < 2:
        return np.nan
    return r.std(ddof=1) * np.sqrt(periods_per_year)

def sharpe_ratio(simple_returns, rf=0.0, periods_per_year=252):
    """
    Annualized Sharpe ratio based on simple returns.

    rf must be the per-period simple risk-free rate.
    """
    r = _to_series(simple_returns, "simple_r").dropna()

    if np.isscalar(rf):
        rf_s = pd.Series(rf, index=r.index)
    else:
        rf_s = _to_series(rf).reindex(r.index)

    excess = (r - rf_s).dropna()
    vol = excess.std(ddof=1)
    if len(excess) < 2 or vol == 0:
        return np.nan
    return excess.mean() / vol * np.sqrt(periods_per_year)

def sortino_ratio(simple_returns, rf=0.0, periods_per_year=252):
    """
    Annualized Sortino ratio based on downside deviation.
    """
    r = _to_series(simple_returns, "simple_r").dropna()

    if np.isscalar(rf):
        rf_s = pd.Series(rf, index=r.index)
    else:
        rf_s = _to_series(rf).reindex(r.index)

    excess = (r - rf_s).dropna()
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)

    if len(excess) < 2 or pd.isna(downside_std) or downside_std == 0:
        return np.nan

    return excess.mean() / downside_std * np.sqrt(periods_per_year)

def calmar_ratio(wealth, periods_per_year=252):
    """
    Calmar ratio = annualized return / abs(max drawdown)
    """
    ann_ret = annualized_return_from_wealth(wealth, periods_per_year=periods_per_year)
    mdd = max_drawdown(wealth)
    if pd.isna(mdd) or mdd == 0:
        return np.nan
    return ann_ret / abs(mdd)

def evaluate_strategy(
    log_returns,
    weights,
    rf=0.0,
    transaction_cost=0.0,
    weight_lag=1,
    periods_per_year=252,
    initial_wealth=1.0,
    strategy_name="strategy",
):
    """
    End-to-end evaluation pipeline independent of model type.

    Returns
    -------
    summary : pd.Series
        Core economic metrics.
    details : pd.DataFrame
        Period-by-period aligned results.
    """
    details = strategy_simple_returns(
        log_returns=log_returns,
        weights=weights,
        rf=rf,
        transaction_cost=transaction_cost,
        weight_lag=weight_lag,
    )

    wealth = cumulative_wealth(
        details["strategy_simple_return"],
        initial_wealth=initial_wealth
    )
    dd = drawdown_from_wealth(wealth)

    details = details.copy()
    details["wealth"] = wealth
    details["drawdown"] = dd

    summary = pd.Series(
        {
            "strategy_name": strategy_name,
            "n_observations": len(details),
            "annualized_return": annualized_return_from_wealth(
                wealth, periods_per_year=periods_per_year
            ),
            "annualized_volatility": annualized_volatility(
                details["strategy_simple_return"],
                periods_per_year=periods_per_year
            ),
            "sharpe_ratio": sharpe_ratio(
                details["strategy_simple_return"],
                rf=rf,
                periods_per_year=periods_per_year
            ),
            "sortino_ratio": sortino_ratio(
                details["strategy_simple_return"],
                rf=rf,
                periods_per_year=periods_per_year
            ),
            "max_drawdown": max_drawdown(wealth),
            "calmar_ratio": calmar_ratio(
                wealth, periods_per_year=periods_per_year
            ),
            "final_wealth": wealth.iloc[-1] if len(wealth) else np.nan,
            "average_turnover": details["turnover"].mean() if len(details) else np.nan,
            "average_weight": details["weight_used"].mean() if len(details) else np.nan,
        }
    )

    return summary, details

def to_series(x, name=None):
    s = x.copy() if isinstance(x, pd.Series) else pd.Series(x)
    if name is not None:
        s.name = name
    return s

def positive_var(var_forecast, eps=1e-12):
    """
    Convert left-tail VaR forecast to a strictly positive magnitude.
    """
    v = to_series(var_forecast, "var_forecast")
    return v.abs().clip(lower=eps)

def compute_weights_from_budget_and_var(var_forecast, B, cap=1.0, floor=0.0):
    """
    Common allocation rule:
        w_t = min(cap, B_t / |VaR_t|)
    """
    var_pos = positive_var(var_forecast)

    if np.isscalar(B):
        B = pd.Series(B, index=var_pos.index, name="B_t")
    else:
        B = to_series(B, "B_t").reindex(var_pos.index)

    w = B / var_pos
    w = w.clip(lower=floor, upper=cap)
    w.name = "weight"
    return w

def rolling_var_quantiles(var_forecast, window=60, q_low=0.30, q_high=0.70):
    """
    Rolling quantiles of past positive VaR magnitudes.
    Quantiles at t use only information up to t-1.
    """
    var_pos = positive_var(var_forecast)
    shifted = var_pos.shift(1)

    q30 = shifted.rolling(window=window, min_periods=window).quantile(q_low)
    q70 = shifted.rolling(window=window, min_periods=window).quantile(q_high)

    q30.name = f"q{int(q_low*100)}"
    q70.name = f"q{int(q_high*100)}"
    return q30, q70

def weights_benchmark_econometric(var_forecast, B=0.01, cap=1.0, floor=0.0):
    """
    EGARCH / GJR-GARCH benchmark:
        w_t = min(1, B / |VaR_t|)
    """
    return compute_weights_from_budget_and_var(
        var_forecast=var_forecast,
        B=B,
        cap=cap,
        floor=floor,
    )

def weights_lstm(var_forecast, B=0.01, cap=1.0, floor=0.0):
    """
    LSTM without regime adaptation:
        w_t = min(1, B / |VaR_t|)
    """
    return compute_weights_from_budget_and_var(
        var_forecast=var_forecast,
        B=B,
        cap=cap,
        floor=floor,
    )

def weights_hmm_lstm_quantile_budget(
    lstm_var_forecast,
    stress_flag,
    window=60,
    q_stress=0.30,
    q_normal=0.70,
    cap=1.0,
    floor=0.0,
):
    """
    HMM-LSTM:
    - VaR comes from LSTM
    - B_t depends on HMM regime
    - in stress regime -> rolling 30% quantile
    - in normal regime -> rolling 70% quantile

    Formula:
        w_t = min(1, B_t / |VaR_t^LSTM|)
    """
    v = to_series(lstm_var_forecast, "lstm_var")
    s = to_series(stress_flag, "stress_flag").astype(bool).reindex(v.index)

    q30, q70 = rolling_var_quantiles(
        var_forecast=v,
        window=window,
        q_low=q_stress,
        q_high=q_normal,
    )

    B_t = pd.Series(index=v.index, dtype=float, name="B_t")
    B_t.loc[s] = q30.loc[s]
    B_t.loc[~s] = q70.loc[~s]

    w = compute_weights_from_budget_and_var(
        var_forecast=v,
        B=B_t,
        cap=cap,
        floor=floor,
    )

    out = pd.DataFrame({
        "lstm_var": v,
        "var_abs": positive_var(v),
        "stress_flag": s,
        "B_stress_q30": q30,
        "B_normal_q70": q70,
        "B_t": B_t,
        "weight": w,
    })

    return out
