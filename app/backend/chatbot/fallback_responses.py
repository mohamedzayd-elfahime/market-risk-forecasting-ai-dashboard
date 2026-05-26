"""Minimal direct responses for non-generative chatbot cases.

This module is intentionally small. It is not a deterministic answer router:
definitions, forecasts, backtests and model explanations should flow through
the response policy + prompt + LLM pipeline. Direct responses are reserved for
tiny social turns and strict safety/fallback cases.
"""

from __future__ import annotations

import re
import unicodedata


def greeting_response() -> str:
    return (
        "Salut. Je peux t'aider a lire le dashboard MASI, analyser la prevision, "
        "la VaR, l'ES, le regime, le backtest ou la strategie."
    )


def short_ack_response() -> str:
    return "Avec plaisir."


def investment_advice_refusal() -> str:
    return (
        "Je ne peux pas donner de conseil d'investissement personnalise. "
        "Je peux cependant t'aider a interpreter les indicateurs de risque : "
        "la VaR, l'Expected Shortfall, le regime HMM et les resultats du backtest."
    )


def _normalize_question(question: str) -> str:
    normalized = unicodedata.normalize("NFKD", question.lower())
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def _is_short_help_turn(question: str) -> bool:
    q = _normalize_question(question)
    if not q:
        return False
    if len(q.split()) > 8:
        return False

    help_markers = (
        "aide moi",
        "tu peux m aider",
        "comment tu peux m aider",
        "comment tu peux m assister",
        "comment peux tu m aider",
        "comment peux tu m assister",
        "que peux tu faire",
        "tu fais quoi",
        "par ou commencer",
    )
    if any(marker in q for marker in help_markers):
        return True

    has_greeting = any(greeting in q.split() for greeting in ("salut", "bonjour", "bonsoir", "hello", "hi", "hey", "salam"))
    has_help = any(marker in q for marker in ("aider", "assister", "faire", "commencer"))
    return has_greeting and has_help


def get_fallback_response(intent: str, question: str = "") -> str | None:
    """Return a direct response only for tiny social/help turns.

    All substantive questions return None so the controlled LLM pipeline can
    answer with the right context and response policy.
    """
    if intent != "help_request":
        return None

    q = question.lower().strip()
    greetings = {"salut", "bonjour", "bonsoir", "hello", "hi", "hey", "salam"}
    if q in greetings:
        return greeting_response()

    if _is_short_help_turn(question):
        return greeting_response()

    short_acks = ("merci", "ok merci", "d'accord merci", "super", "parfait")
    if any(ack in q for ack in short_acks):
        return short_ack_response()

    return None
