"""Forecast routes."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.schemas.api_schema import (
    ForecastHistoryResponse,
    ForecastResponse,
    PriceSeriesResponse,
    RiskSeriesResponse,
)
from backend.dashboard_state.api_file_service import (
    read_forecast_history,
    read_latest_forecast,
    read_price_series,
    read_risk_series,
)


router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/latest", response_model=ForecastResponse)
def latest_forecast(horizon: int | None = Query(default=None, ge=1)) -> dict:
    return read_latest_forecast(horizon=horizon)


@router.get("/history", response_model=ForecastHistoryResponse)
def forecast_history(
    limit: int = Query(default=50, ge=1, le=1000),
    horizon: int | None = Query(default=None, ge=1),
) -> ForecastHistoryResponse:
    rows = read_forecast_history(limit=limit, horizon=horizon)
    return ForecastHistoryResponse(count=len(rows), forecasts=rows)


@router.get("/price-series", response_model=PriceSeriesResponse)
def price_series(history_limit: int = Query(default=260, ge=30, le=2000)) -> dict:
    return read_price_series(history_limit=history_limit)


@router.get("/risk-series", response_model=RiskSeriesResponse)
def risk_series(history_limit: int = Query(default=180, ge=30, le=2000)) -> dict:
    return read_risk_series(history_limit=history_limit)
