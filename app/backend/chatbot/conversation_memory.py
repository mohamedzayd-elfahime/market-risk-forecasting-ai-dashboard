"""
Conversation Memory for the MASI Risk Dashboard Chatbot.

Role:
- Compress the raw conversation history into a short, structured summary.
- Prevent the full history from polluting simple questions.
- Track: last topic, previous intent, last forecast consulted.

The LLM never receives the full raw history — only this compact summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ConversationMemory:
    """Lightweight representation of recent conversation context."""

    last_intent: str = ""
    last_topic: str = ""
    last_forecast_date: str = ""
    recent_questions: list[str] = field(default_factory=list)
    recent_responses: list[str] = field(default_factory=list)
    turn_count: int = 0

    def summary(self) -> str:
        """Return a compact French summary for injection into the prompt."""
        if self.turn_count == 0:
            return ""

        parts: list[str] = []

        if self.turn_count > 0:
            parts.append(f"Tour de conversation : {self.turn_count}.")

        if self.last_topic:
            parts.append(f"Dernier sujet discuté : {self.last_topic}.")

        if self.last_intent:
            parts.append(f"Dernière intention détectée : {self.last_intent}.")

        if self.last_forecast_date:
            parts.append(f"Dernière prévision consultée : {self.last_forecast_date}.")

        if self.recent_questions:
            last_q = self.recent_questions[-1][:100]
            parts.append(f"Dernière question : « {last_q} »")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Topic inference
# ---------------------------------------------------------------------------

_TOPIC_MAP = [
    (["var", "value at risk"], "la VaR"),
    (["expected shortfall", "shortfall", "shorfall"], "l'Expected Shortfall"),
    (["backtest", "kupiec", "christoffersen", "violation"], "le backtesting"),
    (["regime", "hmm"], "les régimes HMM"),
    (["drawdown"], "le drawdown"),
    (["sharpe"], "le Sharpe"),
    (["wealth", "richesse"], "la richesse simulée"),
    (["forecast", "prevision", "prévision"], "les prévisions"),
    (["strategie", "stratégie", "risk targeting", "risk-targeting"], "la stratégie risk-targeting"),
    (["egarch", "gjr", "garch", "lstm", "ridge"], "les modèles"),
    (["masi"], "le MASI"),
    (["dashboard"], "le dashboard"),
]


def _infer_topic(text: str) -> str:
    """Infer the main topic from a piece of text."""
    t = text.lower()
    for keywords, label in _TOPIC_MAP:
        if any(kw in t for kw in keywords):
            return label
    return ""


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_conversation_memory(
    conversation_history: list[dict[str, str]] | None,
    max_turns: int = 4,
) -> ConversationMemory:
    """
    Build a ConversationMemory from raw conversation history.

    Parameters
    ----------
    conversation_history : list[dict] | None
        List of {"role": "user"|"assistant", "content": "..."} dicts.
    max_turns : int
        Maximum number of recent turns to consider.

    Returns
    -------
    ConversationMemory
    """
    memory = ConversationMemory()

    if not conversation_history:
        return memory

    # Keep only valid entries
    valid = [
        m for m in conversation_history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]

    if not valid:
        return memory

    # Limit to recent turns
    recent = valid[-(max_turns * 2):]
    memory.turn_count = sum(1 for m in valid if m["role"] == "user")

    user_msgs = [m["content"] for m in recent if m["role"] == "user"]
    asst_msgs = [m["content"] for m in recent if m["role"] == "assistant"]

    memory.recent_questions = [q[:150] for q in user_msgs[-2:]]
    memory.recent_responses = [r[:150] for r in asst_msgs[-1:]]

    # Infer last topic from combined recent text
    combined = " ".join(m["content"] for m in recent)
    memory.last_topic = _infer_topic(combined)

    return memory


def sanitize_conversation_history(
    conversation_history: list[dict[str, str]] | None,
    max_messages: int = 4,
) -> list[dict[str, str]]:
    """
    Clean and limit the conversation history.

    Keeps only the most recent `max_messages` user/assistant exchanges.
    Removes messages with empty content or invalid roles.

    Parameters
    ----------
    conversation_history : list[dict] | None
        Raw history from the API request.
    max_messages : int
        Maximum number of messages to keep.

    Returns
    -------
    list[dict[str, str]]
        Cleaned, truncated history.
    """
    if not conversation_history:
        return []

    valid = [
        {"role": str(m.get("role", "")).strip(), "content": str(m.get("content", "")).strip()}
        for m in conversation_history
        if isinstance(m, dict)
        and str(m.get("role", "")).strip() in ("user", "assistant")
        and str(m.get("content", "")).strip()
    ]

    return valid[-max_messages:]


def get_conversation_summary(
    conversation_history: list[dict[str, str]] | None,
) -> str:
    """
    Get a compact summary string suitable for prompt injection.

    This is the main function used by chat_service.py.
    """
    memory = build_conversation_memory(conversation_history)
    return memory.summary()


def was_help_offered_recently(
    conversation_history: list[dict[str, str]] | None,
    window: int = 3,
) -> bool:
    """
    Return True if the assistant recently offered guided help.

    Used to detect "ok / vas-y / commence" after a help response.
    """
    if not conversation_history:
        return False

    recent_asst = [
        m["content"]
        for m in conversation_history[-window:]
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]

    joined = " ".join(recent_asst).lower()
    return (
        "prevision 1 jour" in joined
        or "prévision 1 jour" in joined
        or "on peut commencer" in joined
        or "dashboard masi" in joined
    )
