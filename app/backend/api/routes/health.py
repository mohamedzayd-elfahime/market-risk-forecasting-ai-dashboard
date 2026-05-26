"""Health route."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.api_schema import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", message="MASI backend API is running.")
