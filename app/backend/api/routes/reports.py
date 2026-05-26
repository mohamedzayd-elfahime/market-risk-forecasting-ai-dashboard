"""Report routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.api_schema import ReportResponse
from backend.dashboard_state.api_file_service import read_latest_report


router = APIRouter(tags=["reports"])


@router.get("/report/latest", response_model=ReportResponse)
def latest_report() -> dict:
    return read_latest_report()
