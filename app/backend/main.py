"""FastAPI entry point for the MASI backend."""

from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import admin, backtest, chat, context, forecast, health, plots, reports
from backend.core.config import AUTO_START_OLLAMA, LLM_BACKEND, WARM_RAG_ON_STARTUP
from backend.core.paths import APP_ROOT
from backend.llm.local_ollama_client import ensure_ollama_server
from backend.chatbot.rag.retriever import warm_retriever_cache


STATIC_DASHBOARD_PATH = APP_ROOT / "dashboard"


def create_app() -> FastAPI:
    app = FastAPI(
        title="MASI Forecasting API",
        description="Read-only API over the existing MASI forecasting backend outputs.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(forecast.router)
    app.include_router(chat.router)
    app.include_router(chat.router, prefix="/api")
    app.include_router(context.router)
    app.include_router(reports.router)
    app.include_router(plots.router)
    app.include_router(backtest.router)
    app.include_router(admin.router)
    app.mount(
        "/dashboard",
        StaticFiles(directory=STATIC_DASHBOARD_PATH),
        name="dashboard",
    )

    @app.get("/", include_in_schema=False)
    def dashboard_index() -> FileResponse:
        return FileResponse(STATIC_DASHBOARD_PATH / "index.html")

    @app.on_event("startup")
    def startup_warmups() -> None:
        if AUTO_START_OLLAMA and LLM_BACKEND.strip().lower() == "ollama":
            ensure_ollama_server()

        if WARM_RAG_ON_STARTUP:
            thread = threading.Thread(target=warm_retriever_cache, daemon=True)
            thread.start()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
