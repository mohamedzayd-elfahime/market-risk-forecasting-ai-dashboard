"""Route-specific context builder for the MASI chatbot.

Architecture:
1. The intent router classifies the user request.
2. One or more context routes extract only the needed evidence:
   - static_rag: semantic retrieval from the vector/markdown knowledge base.
   - forecast: exact forecast rows from CSV/report files.
   - backtest: exact statistical/economic backtest metrics.
3. The context builder compiles retrieved text and numeric facts for the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.core.config import RAG_RETRIEVER_BACKEND
from backend.chatbot.rag.fast_markdown_retriever import retrieve_fast_markdown_context
from backend.dashboard_state.context_builder import (
    build_economic_backtest_context,
    build_forecast_context,
    build_model_metadata_context,
    build_statistical_backtest_context,
)


MAX_RAG_CONTEXT_CHARS = 4500
MAX_NUMERIC_CONTEXT_CHARS = 3500


@dataclass(frozen=True)
class RoutedContext:
    """Context bundle produced by the three-route chatbot architecture."""

    rag_context: str
    numeric_context: str
    validation_context: str
    routes: tuple[str, ...]
    used_rag: bool
    used_numeric_context: bool


def build_routed_chat_context(
    question: str,
    intent: str,
    rag_k: int = 5,
    rag_min_score: float = 1.5,
) -> RoutedContext:
    """Build context according to the routed chatbot architecture."""

    routes = _routes_for_intent(intent)
    rag_context = ""
    numeric_blocks: list[str] = []

    if "static_rag" in routes:
        rag_context = _retrieve_static_rag_context(
            question,
            k=rag_k,
            min_score=rag_min_score,
        )

    if "forecast" in routes:
        numeric_blocks.append("[Route 2 : Forecast]\n" + build_forecast_context())

    if "backtest" in routes:
        numeric_blocks.append(
            "[Route 3 : Backtest]\n"
            "## Backtesting statistique\n"
            f"{build_statistical_backtest_context()}\n\n"
            "## Backtesting economique\n"
            f"{build_economic_backtest_context()}"
        )

    if "model_metadata" in routes:
        numeric_blocks.append(
            "[Donnees chiffrees : modele et jeu de donnees]\n"
            + build_model_metadata_context()
        )

    numeric_context = _compile_numeric_context(numeric_blocks)
    validation_context = "\n\n".join(
        part for part in (numeric_context, rag_context) if part.strip()
    )

    return RoutedContext(
        rag_context=rag_context,
        numeric_context=numeric_context,
        validation_context=validation_context,
        routes=routes,
        used_rag=bool(rag_context.strip()),
        used_numeric_context=bool(numeric_context.strip()),
    )


def _routes_for_intent(intent: str) -> tuple[str, ...]:
    """Map chatbot intents onto the three requested architecture routes."""

    if intent == "help_request":
        return ("forecast", "backtest")
    if intent == "out_of_scope":
        return ()
    if intent == "forecast_query":
        return ("forecast",)
    if intent in {"backtest_query", "strategy_query"}:
        return ("backtest",)
    if intent in {"data_query", "model_query"}:
        return ("static_rag", "model_metadata")
    return ("static_rag",)


def _retrieve_static_rag_context(query: str, k: int, min_score: float) -> str:
    backend = RAG_RETRIEVER_BACKEND.strip().lower()
    if backend == "markdown":
        context = retrieve_fast_markdown_context(query=query, k=k, min_score=min_score)
    elif backend == "chroma":
        from backend.chatbot.rag.retriever import retrieve_relevant_context

        context = retrieve_relevant_context(query=query, k=k)
    else:
        raise RuntimeError(
            f"Retriever RAG non supporte: {RAG_RETRIEVER_BACKEND}. "
            "Valeurs supportees: markdown, chroma."
        )
    return _truncate_context(context, MAX_RAG_CONTEXT_CHARS)


def _compile_numeric_context(blocks: list[str]) -> str:
    context = "\n\n".join(block.strip() for block in blocks if block.strip())
    return _truncate_context(context, MAX_NUMERIC_CONTEXT_CHARS)


def _truncate_context(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return (
        cleaned[:max_chars].rstrip()
        + "\n\n[Contexte tronque pour garder une generation rapide.]"
    )
