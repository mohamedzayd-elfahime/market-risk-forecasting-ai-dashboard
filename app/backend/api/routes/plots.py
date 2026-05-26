"""Plot routes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.dashboard_state.api_file_service import latest_plot_path


router = APIRouter(tags=["plots"])


@router.get("/plot/latest")
def latest_plot() -> FileResponse:
    path = latest_plot_path()
    return FileResponse(path, media_type="image/png", filename=path.name)
