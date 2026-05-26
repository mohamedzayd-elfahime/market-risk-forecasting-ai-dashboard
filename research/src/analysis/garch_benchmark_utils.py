"""Reusable functions extracted from the research notebooks.

The notebooks keep the narrative and result cells; this module keeps shared code.
"""

from dataclasses import dataclass

import numpy as np

import pandas as pd

@dataclass
class RollingVaRSplit:
    train_initial_df: pd.DataFrame
    test_df: pd.DataFrame
    rolling_windows_info: pd.DataFrame
    train_size: int
    test_size: int
    total_observations: int

def prepare_rolling_var_benchmark_split(
    df: pd.DataFrame,
    date_column: str = "date",
    return_column: str = "masi_log_return",
    train_size: int = 4000,
    test_size: int = 764,
) -> RollingVaRSplit:
    """
    Prepare a strict chronological fixed rolling-window split for a VaR benchmark.

    Design:
    - total expected observations = 4766
    - initial estimation window = 4000 observations
    - out-of-sample backtesting window = 766 observations
    - rolling one-step-ahead forecasting
    - fixed training window size at each forecast date
    - no shuffle, no random split, no future leakage

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame already sorted or intended to be sorted chronologically.
    date_column : str, default="date"
        Name of the date column.
    return_column : str, default="masi_log_return"
        Name of the return column.
    train_size : int, default=4000
        Number of observations in each rolling estimation window.
    test_size : int, default=766
        Number of out-of-sample one-step-ahead forecasts.

    Returns
    -------
    RollingVaRSplit
        Object containing:
        - train_initial_df
        - test_df
        - rolling_windows_info
        - train_size
        - test_size
        - total_observations

    Notes
    -----
    For each forecast index t in the test period:
    - training window = [t - train_size, ..., t - 1]
    - forecast target   = t
    This guarantees a strict rolling one-step-ahead design.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = [date_column, return_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    expected_total = train_size + test_size
    if expected_total != 4764:
        raise ValueError(
            f"Protocol inconsistency: train_size + test_size must equal 4764, "
            f"got {train_size} + {test_size} = {expected_total}."
        )

    working_df = df[[date_column, return_column]].copy()
    working_df[date_column] = pd.to_datetime(working_df[date_column], errors="coerce")

    if working_df[date_column].isna().any():
        raise ValueError(f"Column '{date_column}' contains invalid or missing dates.")

    if working_df[return_column].isna().any():
        raise ValueError(f"Column '{return_column}' contains missing values.")

    if not working_df[date_column].is_monotonic_increasing:
        raise ValueError(
            f"Data are not sorted chronologically by '{date_column}'. "
            "Please sort the DataFrame in ascending date order before continuing."
        )

    total_observations = len(working_df)
    if total_observations < expected_total:
        raise ValueError(
            f"Insufficient observations: need at least {expected_total}, "
            f"got {total_observations}."
        )

    if total_observations != expected_total:
        raise ValueError(
            f"Unexpected sample size: expected exactly {expected_total} observations, "
            f"got {total_observations}."
        )

    # Initial estimation sample: first 4000 observations
    train_initial_df = working_df.iloc[:train_size].copy().reset_index(drop=True)

    # Out-of-sample backtesting sample: last 766 observations
    test_df = working_df.iloc[train_size:train_size + test_size].copy().reset_index(drop=True)

    # Rolling one-step-ahead windows:
    # for each forecast index t in [4000, ..., 4765],
    # use exactly 4000 observations from [t-4000, ..., t-1]
    rolling_windows = []
    forecast_indices = np.arange(train_size, train_size + test_size)

    for window_number, forecast_index in enumerate(forecast_indices, start=1):
        train_start_index = forecast_index - train_size
        train_end_index = forecast_index - 1

        rolling_windows.append(
            {
                "window_number": window_number,
                "train_start_index": int(train_start_index),
                "train_end_index": int(train_end_index),
                "forecast_index": int(forecast_index),
                "train_start_date": working_df.loc[train_start_index, date_column],
                "train_end_date": working_df.loc[train_end_index, date_column],
                "forecast_date": working_df.loc[forecast_index, date_column],
            }
        )

    rolling_windows_info = pd.DataFrame(rolling_windows)

    first_window = rolling_windows_info.iloc[0]
    last_window = rolling_windows_info.iloc[-1]

    print("Rolling VaR benchmark split prepared")
    print(f"Check total observations:          {train_size} + {test_size} = {expected_total}")
    print(f"Total observations in dataset:     {total_observations}")
    print(f"Initial estimation sample size:    {len(train_initial_df)}")
    print(f"Backtesting sample size:           {len(test_df)}")
    print(f"Initial train index range:         0 to {train_size - 1}")
    print(f"Test index range:                  {train_size} to {train_size + test_size - 1}")
    print(f"Expected out-of-sample forecasts:  {len(rolling_windows_info)}")
    print()
    print("First rolling window")
    print(f"  Train indices:                   {first_window['train_start_index']} to {first_window['train_end_index']}")
    print(f"  Train dates:                     {first_window['train_start_date']} to {first_window['train_end_date']}")
    print(f"  Forecast index:                  {first_window['forecast_index']}")
    print(f"  Forecast date:                   {first_window['forecast_date']}")
    print()
    print("Last rolling window")
    print(f"  Train indices:                   {last_window['train_start_index']} to {last_window['train_end_index']}")
    print(f"  Train dates:                     {last_window['train_start_date']} to {last_window['train_end_date']}")
    print(f"  Forecast index:                  {last_window['forecast_index']}")
    print(f"  Forecast date:                   {last_window['forecast_date']}")

    return RollingVaRSplit(
        train_initial_df=train_initial_df,
        test_df=test_df,
        rolling_windows_info=rolling_windows_info,
        train_size=train_size,
        test_size=test_size,
        total_observations=total_observations,
    )

from arch import arch_model

def fit_forecast_egarch(
    training_returns,
    mean_spec,
    innovation_dist,
    p,
    o,
    q,
    scale_factor=100.0,
):
    """Fit a 1-step-ahead EGARCH model with reversible multiplicative scaling."""
    try:
        y = training_returns.dropna().astype(float)

        if mean_spec not in {"zero", "constant", "ar1"}:
            raise ValueError("mean_spec must be one of {'zero', 'constant', 'ar1'}")
        if innovation_dist not in {"normal", "t", "ged"}:
            raise ValueError("innovation_dist must be one of {'normal', 't', 'ged'}")
        if min(p, o, q) < 0:
            raise ValueError("EGARCH orders must be non-negative")
        if scale_factor <= 0:
            raise ValueError("scale_factor must be strictly positive")

        y_scaled = y * scale_factor

        mean_kwargs = {
            "zero": {"mean": "Zero", "lags": None},
            "constant": {"mean": "Constant", "lags": None},
            "ar1": {"mean": "ARX", "lags": 1},
        }[mean_spec]

        model = arch_model(
            y_scaled,
            mean=mean_kwargs["mean"],
            lags=mean_kwargs["lags"],
            vol="EGARCH",
            p=p,
            o=o,
            q=q,
            dist=innovation_dist,
            rescale=False,
        )

        fitted_result = model.fit(disp="off", show_warning=False)
        forecast = fitted_result.forecast(horizon=1, reindex=False)

        mean_forecast_scaled = float(forecast.mean.iloc[-1, 0])
        residual_variance_scaled = float(forecast.residual_variance.iloc[-1, 0])
        volatility_forecast_scaled = float(np.sqrt(residual_variance_scaled))

        mean_forecast = mean_forecast_scaled / scale_factor
        volatility_forecast = volatility_forecast_scaled / scale_factor

        return {
            "success": True,
            "fitted_result": fitted_result,
            "mean_forecast": mean_forecast,
            "volatility_forecast": volatility_forecast,
            "error": None,
        }

    except Exception as error:
        return {
            "success": False,
            "fitted_result": None,
            "mean_forecast": np.nan,
            "volatility_forecast": np.nan,
            "error": str(error),
        }

def fit_forecast_gjr_garch(
    training_returns,
    mean_spec,
    innovation_dist,
    p,
    o,
    q,
    scale_factor=100.0,
):
    """Fit a 1-step-ahead GJR-GARCH model on scaled returns."""
    try:
        y = training_returns.dropna().astype(float)

        if mean_spec not in {"zero", "constant", "ar1"}:
            raise ValueError("mean_spec must be one of {'zero', 'constant', 'ar1'}")
        if innovation_dist not in {"normal", "t", "ged"}:
            raise ValueError("innovation_dist must be one of {'normal', 't', 'ged'}")
        if min(p, o, q) < 0:
            raise ValueError("GJR-GARCH orders must be non-negative")
        if scale_factor <= 0:
            raise ValueError("scale_factor must be strictly positive")

        # Simple reversible scaling for numerical stability
        y_scaled = y * scale_factor

        mean_kwargs = {
            "zero": {"mean": "Zero", "lags": None},
            "constant": {"mean": "Constant", "lags": None},
            "ar1": {"mean": "ARX", "lags": 1},
        }[mean_spec]

        model = arch_model(
            y_scaled,
            mean=mean_kwargs["mean"],
            lags=mean_kwargs["lags"],
            vol="GARCH",
            p=p,
            o=o,
            q=q,
            power=2.0,
            dist=innovation_dist,
            rescale=False,
        )

        fitted_result = model.fit(disp="off", show_warning=False)
        forecast = fitted_result.forecast(horizon=1, reindex=False)

        mean_forecast_scaled = float(forecast.mean.iloc[-1, 0])
        error_variance_forecast_scaled = float(forecast.residual_variance.iloc[-1, 0])
        volatility_forecast_scaled = float(np.sqrt(error_variance_forecast_scaled))

        # Convert forecasts back to the original return scale
        mean_forecast = mean_forecast_scaled / scale_factor
        volatility_forecast = volatility_forecast_scaled / scale_factor

        return {
            "success": True,
            "fitted_result": fitted_result,
            "mean_forecast": mean_forecast,
            "volatility_forecast": volatility_forecast,
            "metadata": {
                "model_family": "GJR-GARCH",
                "mean_spec": mean_spec,
                "innovation_dist": innovation_dist,
                "p": p,
                "o": o,
                "q": q,
                "scale_factor": scale_factor,
            },
            "error": None,
        }

    except Exception as error:
        return {
            "success": False,
            "fitted_result": None,
            "mean_forecast": np.nan,
            "volatility_forecast": np.nan,
            "metadata": {
                "model_family": "GJR-GARCH",
                "mean_spec": mean_spec,
                "innovation_dist": innovation_dist,
                "p": p,
                "o": o,
                "q": q,
                "scale_factor": scale_factor,
            },
            "error": str(error),
        }

from scipy.integrate import quad

from scipy.special import gamma

from scipy.stats import norm, t, gennorm

class ParametricRiskError(ValueError):
    """Raised when parametric VaR/ES computation fails or is inconsistent."""

def _validate_finite_scalar(name, value):
    """Validate that value is a finite scalar."""
    if not np.isscalar(value):
        raise ParametricRiskError(f"{name} must be a scalar")
    if not np.isfinite(value):
        raise ParametricRiskError(f"{name} must be finite")

def _validate_positive_scalar(name, value):
    """Validate that value is finite and strictly positive."""
    _validate_finite_scalar(name, value)
    if value <= 0:
        raise ParametricRiskError(f"{name} must be strictly positive")

def _validate_non_negative_scalar(name, value):
    """Validate that value is finite and non-negative."""
    _validate_finite_scalar(name, value)
    if value < 0:
        raise ParametricRiskError(f"{name} must be non-negative")

def _normalize_distribution_name(distribution_name):
    """Normalize accepted aliases for distribution names."""
    if distribution_name is None:
        raise ParametricRiskError("distribution_name must not be None")

    name = str(distribution_name).strip().lower()
    aliases = {
        "normal": "normal",
        "gaussian": "normal",
        "norm": "normal",
        "t": "t",
        "student": "t",
        "student-t": "t",
        "student_t": "t",
        "ged": "ged",
        "generalized error": "ged",
        "generalized_error": "ged",
        "generalized-error": "ged",
    }

    if name not in aliases:
        raise ParametricRiskError(
            "distribution_name must be one of "
            "{'normal', 't', 'ged'} or a supported alias"
        )
    return aliases[name]

def _standardized_t_quantile_and_es(alpha, degrees_of_freedom):
    """
    Return q_alpha and E[Z | Z <= q_alpha] for a variance-1 standardized Student-t.
    """
    _validate_positive_scalar("degrees_of_freedom", degrees_of_freedom)
    if degrees_of_freedom <= 2:
        raise ParametricRiskError("For standardized Student-t, degrees_of_freedom must be > 2")

    nu = float(degrees_of_freedom)
    scale = np.sqrt((nu - 2.0) / nu)

    raw_quantile = t.ppf(alpha, df=nu)
    standardized_quantile = scale * raw_quantile

    raw_tail_mean = -(
        t.pdf(raw_quantile, df=nu) * (nu + raw_quantile**2)
    ) / ((nu - 1.0) * alpha)

    standardized_tail_mean = scale * raw_tail_mean
    return standardized_quantile, standardized_tail_mean

def _standardized_ged_quantile_and_es(alpha, ged_shape, quad_epsabs=1e-12, quad_epsrel=1e-10):
    """
    Return q_alpha and E[Z | Z <= q_alpha] for a variance-1 standardized GED.
    """
    _validate_positive_scalar("ged_shape", ged_shape)

    beta = float(ged_shape)
    raw_variance = gamma(3.0 / beta) / gamma(1.0 / beta)
    raw_std = np.sqrt(raw_variance)

    raw_quantile = gennorm.ppf(alpha, beta=beta)
    standardized_quantile = raw_quantile / raw_std

    def standardized_density(z):
        return raw_std * gennorm.pdf(raw_std * z, beta=beta)

    cdf_check = gennorm.cdf(raw_std * standardized_quantile, beta=beta)
    if not np.isfinite(cdf_check) or abs(cdf_check - alpha) > 1e-8:
        raise ParametricRiskError(
            "GED quantile consistency check failed; verify parameterization."
        )

    tail_first_moment, integration_error = quad(
        lambda z: z * standardized_density(z),
        -np.inf,
        standardized_quantile,
        epsabs=quad_epsabs,
        epsrel=quad_epsrel,
        limit=500,
    )

    if not np.isfinite(tail_first_moment):
        raise ParametricRiskError("GED ES integration returned a non-finite result")

    if not np.isfinite(integration_error):
        raise ParametricRiskError("GED ES integration error estimate is non-finite")

    standardized_tail_mean = tail_first_moment / alpha
    return standardized_quantile, standardized_tail_mean

def compute_parametric_var_es(
    mean_forecast,
    volatility_forecast,
    distribution_name,
    distribution_parameters=None,
    alpha=0.05,
    check_monotonicity=True,
):
    """
    Compute 1-step-ahead parametric VaR and ES on the original return scale.

    Model:
        R_{t+1} = mu_{t+1} + sigma_{t+1} Z_{t+1}

    where:
        - mu_{t+1}    = mean_forecast
        - sigma_{t+1} = volatility_forecast
        - Z_{t+1}     = standardized innovation with E[Z]=0 and Var[Z]=1
    """
    _validate_finite_scalar("mean_forecast", mean_forecast)
    _validate_non_negative_scalar("volatility_forecast", volatility_forecast)
    _validate_positive_scalar("alpha", alpha)

    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ParametricRiskError("alpha must be in (0, 1)")

    distribution = _normalize_distribution_name(distribution_name)
    params = {} if distribution_parameters is None else dict(distribution_parameters)

    mean_forecast = float(mean_forecast)
    volatility_forecast = float(volatility_forecast)

    if volatility_forecast == 0.0:
        return mean_forecast, mean_forecast

    if distribution == "normal":
        innovation_quantile = norm.ppf(alpha)
        expected_shortfall_innovation = -norm.pdf(innovation_quantile) / alpha

    elif distribution == "t":
        if "degrees_of_freedom" not in params:
            raise ParametricRiskError(
                "For Student-t, distribution_parameters must include 'degrees_of_freedom'"
            )
        innovation_quantile, expected_shortfall_innovation = _standardized_t_quantile_and_es(
            alpha=alpha,
            degrees_of_freedom=params["degrees_of_freedom"],
        )

    elif distribution == "ged":
        if "ged_shape" not in params:
            raise ParametricRiskError(
                "For GED, distribution_parameters must include 'ged_shape'"
            )
        innovation_quantile, expected_shortfall_innovation = _standardized_ged_quantile_and_es(
            alpha=alpha,
            ged_shape=params["ged_shape"],
        )

    else:
        raise ParametricRiskError("Internal error: unsupported distribution after normalization")

    value_at_risk = mean_forecast + volatility_forecast * innovation_quantile
    expected_shortfall = mean_forecast + volatility_forecast * expected_shortfall_innovation

    if not np.isfinite(value_at_risk):
        raise ParametricRiskError("Computed VaR is non-finite")
    if not np.isfinite(expected_shortfall):
        raise ParametricRiskError("Computed ES is non-finite")

    if check_monotonicity and expected_shortfall > value_at_risk + 1e-12:
        raise ParametricRiskError(
            "Consistency check failed: for left-tail risk, ES should satisfy ES <= VaR"
        )

    return value_at_risk, expected_shortfall

def run_parametric_var_es_self_tests(verbose=True):
    """
    Basic deterministic self-tests for publication-grade sanity checking.
    """
    test_cases = [
        {
            "name": "normal_5pct",
            "kwargs": dict(
                mean_forecast=0.0,
                volatility_forecast=0.01,
                distribution_name="normal",
                alpha=0.05,
            ),
        },
        {
            "name": "student_t_df8_5pct",
            "kwargs": dict(
                mean_forecast=0.0,
                volatility_forecast=0.01,
                distribution_name="t",
                distribution_parameters={"degrees_of_freedom": 8},
                alpha=0.05,
            ),
        },
        {
            "name": "ged_beta_1_5_5pct",
            "kwargs": dict(
                mean_forecast=0.0,
                volatility_forecast=0.01,
                distribution_name="ged",
                distribution_parameters={"ged_shape": 1.5},
                alpha=0.05,
            ),
        },
        {
            "name": "zero_volatility",
            "kwargs": dict(
                mean_forecast=0.001,
                volatility_forecast=0.0,
                distribution_name="normal",
                alpha=0.05,
            ),
        },
        {
            "name": "negative_mean_forecast",
            "kwargs": dict(
                mean_forecast=-0.001,
                volatility_forecast=0.01,
                distribution_name="normal",
                alpha=0.05,
            ),
        },
    ]

    results = []
    for case in test_cases:
        var_value, es_value = compute_parametric_var_es(**case["kwargs"])

        assert np.isfinite(var_value)
        assert np.isfinite(es_value)

        if case["kwargs"]["volatility_forecast"] == 0.0:
            assert var_value == case["kwargs"]["mean_forecast"]
            assert es_value == case["kwargs"]["mean_forecast"]
        else:
            assert es_value <= var_value + 1e-12

        results.append((case["name"], var_value, es_value))

    if verbose:
        print("Self-tests passed.\n")
        for name, var_value, es_value in results:
            print(f"{name:<24} VaR={var_value: .8f}   ES={es_value: .8f}")

    return results
