"""Pydantic response schemas for the MASI API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    message: str


class ForecastResponse(BaseModel):
    run_date: str | None = None
    target_date: str | None = None
    model_name: str | None = None
    horizon: int | None = None
    alpha: float | None = None
    mean_forecast: float | None = None
    volatility_forecast: float | None = None
    return_forecast: float | None = None
    var_forecast: float | None = None
    es_forecast: float | None = None
    regime: str | None = None
    weight: float | None = None
    model_version: str | None = None
    data_version: str | None = None
    realized_return: float | None = None


class ForecastHistoryResponse(BaseModel):
    count: int
    forecasts: list[ForecastResponse]


class ObservedPricePoint(BaseModel):
    date: str
    close: float


class ForecastPricePoint(BaseModel):
    target_date: str
    horizon: int
    mean_forecast: float | None = None
    volatility_forecast: float | None = None
    return_forecast: float | None = None
    var_forecast: float | None = None
    es_forecast: float | None = None
    price_forecast_proxy: float | None = None
    var_price_proxy: float | None = None
    es_price_proxy: float | None = None


class PriceSeriesResponse(BaseModel):
    last_observed_date: str
    last_observed_price: float
    observed: list[ObservedPricePoint]
    forecasts: list[ForecastPricePoint]


class ObservedRiskPoint(BaseModel):
    date: str
    mean: float | None = None
    volatility: float | None = None


class ForecastRiskPoint(BaseModel):
    target_date: str
    horizon: int
    mean_forecast: float | None = None
    volatility_forecast: float | None = None


class RiskSeriesResponse(BaseModel):
    last_observed_date: str
    observed: list[ObservedRiskPoint]
    forecasts: list[ForecastRiskPoint]


class ReportResponse(BaseModel):
    report_name: str
    content: str
    path: str


class DashboardContextResponse(BaseModel):
    context: str


class ChatAskRequest(BaseModel):
    question: str
    debug: bool = False
    conversation_history: list[dict[str, str]] | None = None
    current_dashboard_state: dict[str, Any] | None = None


class ChatAskResponse(BaseModel):
    answer: str
    used_rag: bool
    used_dashboard_context: bool
    llm_backend: str | None = None
    model_name: str
    intent: str | None = None
    question: str | None = None
    rag_context: str | None = None
    dashboard_context: str | None = None
    prompt: str | None = None


class BacktestSummaryResponse(BaseModel):
    violation_rate: float | None = None
    expected_violation_rate: float | None = None
    n_observations: int | None = None
    n_violations: int | None = None
    es_tail_residual_mean: float | None = None
    final_wealth: float | None = None
    buy_hold_final_wealth: float | None = None
    max_drawdown: float | None = None
    current_drawdown: float | None = None
    buy_hold_max_drawdown: float | None = None
    annualized_sharpe: float | None = None
    buy_hold_sharpe: float | None = None
    statistical: dict[str, Any]
    economic: dict[str, Any]
    source_files: dict[str, str]


class TestPredictionResponse(BaseModel):
    date: str | None = None
    masi_close: float | None = None
    target_return: float | None = None
    realized_return: float | None = None
    return_pred: float | None = None
    var_pred: float | None = None
    es_pred: float | None = None
    predicted_shortfall: float | None = None
    violation: int | None = None
    buy_hold_wealth: float | None = None
    strategy_wealth: float | None = None
    buy_hold_drawdown: float | None = None
    strategy_drawdown: float | None = None
    strategy_weight: float | None = None


class TestPredictionsResponse(BaseModel):
    count: int
    predictions: list[TestPredictionResponse]
