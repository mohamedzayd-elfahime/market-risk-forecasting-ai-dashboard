"""Context routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.api_schema import DashboardContextResponse
from backend.dashboard_state.context_builder import build_dashboard_context


router = APIRouter(tags=["context"])


@router.get("/context/dashboard", response_model=DashboardContextResponse)
def dashboard_context() -> DashboardContextResponse:
    return DashboardContextResponse(context=build_dashboard_context())
