"""
Chat Turn and Chat Service Result structures for the MASI Risk Dashboard Chatbot.

These data structures model the state of a single chatbot interaction
and the final result returned by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.chatbot.response_policy_router import ResponsePolicy


@dataclass
class ChatTurn:
    question: str
    cleaned_history: list[dict[str, str]]
    dashboard_state_context: str
    intent: str
    rag_context: str
    numeric_context: str
    validation_context: str
    prompt: str
    used_rag: bool
    used_dashboard_context: bool
    backend_name: str
    model_name: str
    response_policy: ResponsePolicy | None = None
