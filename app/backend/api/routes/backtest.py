"""Backtest routes."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.schemas.api_schema import BacktestSummaryResponse, TestPredictionsResponse
from backend.dashboard_state.api_file_service import read_backtest_summary, read_test_predictions


router = APIRouter(tags=["backtest"])


@router.get("/backtest/latest", response_model=BacktestSummaryResponse)
def latest_backtest() -> dict:
    return read_backtest_summary()


@router.get("/backtest/test-predictions", response_model=TestPredictionsResponse)
def test_predictions(limit: int = Query(default=250, ge=10, le=2000)) -> TestPredictionsResponse:
    rows = read_test_predictions(limit=limit)
    return TestPredictionsResponse(count=len(rows), predictions=rows)
