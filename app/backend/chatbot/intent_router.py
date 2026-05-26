"""Semantic intent router for the MASI Risk Dashboard chatbot.

This module intentionally avoids hand-written keyword routing.  Each intent is
represented by a short natural-language route description, and the router picks
the closest route semantically.  The output still matches the rest of the
chatbot architecture:

user request -> intent router -> static RAG / forecast / backtest route
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


_STOP_WORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "cest", "c", "comme", "dans",
    "de", "des", "du", "elle", "en", "est", "et", "il", "je", "la", "le",
    "les", "l", "ma", "me", "moi", "mon", "ne", "pas", "pour", "que",
    "quel", "quelle", "quelles", "quels", "qui", "se", "sur",
    "ta", "te", "tu", "un", "une", "va", "veux",
}


INTENT_DESCRIPTIONS: dict[str, str] = {
    "help_request": (
        "Demande d'aide, salutation, guidage, question sur comment utiliser le "
        "dashboard ou par ou commencer. Exemples naturels: salut, bonjour, oui, "
        "ok, commence, aide moi, guide moi."
    ),
    "definition_query": (
        "Question pedagogique ou conceptuelle: definition, explication d'un "
        "terme, difference entre concepts comme MASI, VaR, Expected Shortfall, "
        "ES, regime HMM, EGARCH, GJR-GARCH, LSTM, risk targeting. Exemples "
        "naturels: c est quoi, qu est ce que, que signifie, veut dire, definis, "
        "definir, explique la difference."
    ),
    "forecast_query": (
        "Question sur les previsions actuelles du dashboard: rendement prevu, "
        "prediction, predictions, predicton, predictin, forecast, forecasting, "
        "forcasting, page de forecasting, horizon un jour ou multi-jours, "
        "VaR actuelle, ES actuelle, volatilite, regime courant, direction "
        "estimee du MASI."
    ),
    "backtest_query": (
        "Question sur le backtest, backtesting, validation statistique, les "
        "resultats de test, les violations VaR, les p-values, Kupiec, "
        "Christoffersen, calibration ou comparaison predit contre realise."
    ),
    "strategy_query": (
        "Question sur la strategie risk-managed ou risk-targeting, exposition, "
        "poids simule, allocation, portefeuille, richesse, Sharpe, drawdown, "
        "budget de risque, regime HMM actuel ou demande de recommandation "
        "financiere, dois acheter MASI, dois vendre MASI, acheter, vendre, "
        "conseil, investissement."
    ),
    "model_query": (
        "Question sur l'architecture des modeles et leur fonctionnement: "
        "EGARCH, GJR-GARCH, LSTM-Ridge, HMM, LLM, chatbot, configuration."
    ),
    "data_query": (
        "Question sur les donnees, source, periode, frequence, preprocessing, "
        "train validation test, variables, sequence length, alpha, metadata."
    ),
    "out_of_scope": (
        "Question hors perimetre du projet: meteo, sport, voyage, "
        "recette, traduction, politique, crypto externe, sujets sans lien avec "
        "le projet."
    ),
}

def classify_user_intent(question: str) -> str:
    """Classify the user request with a semantic route-description matcher."""

    if not isinstance(question, str) or not question.strip():
        return "help_request"
    if _is_short_help_question(question):
        return "help_request"

    query_vector = Counter(_tokenize(question))
    if not query_vector:
        return "out_of_scope"

    scored = [
        (intent, _cosine_similarity(query_vector, vector))
        for intent, vector in _DOCUMENT_VECTORS.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    best_intent, best_score = scored[0]

    # Low confidence means the request does not resemble any MASI route.
    if best_score < 0.08:
        return "out_of_scope"

    return best_intent


def is_help_request(question: str) -> bool:
    return classify_user_intent(question) == "help_request"


def is_definition_query(question: str) -> bool:
    return classify_user_intent(question) == "definition_query"


def is_forecast_query(question: str) -> bool:
    return classify_user_intent(question) == "forecast_query"


def is_out_of_scope(question: str) -> bool:
    return classify_user_intent(question) == "out_of_scope"


def _tokenize(text: str) -> list[str]:
    normalized = _normalize(text)
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 1 and token not in _STOP_WORDS
    ]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("'", " ")
    return text


def _is_short_help_question(question: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z0-9]+", _normalize(question)))
    if len(normalized.split()) > 8:
        return False

    domain_terms = (
        "var",
        "es",
        "expected shortfall",
        "forecast",
        "prevision",
        "prediction",
        "backtest",
        "kupiec",
        "christoffersen",
        "hmm",
        "egarch",
        "risk targeting",
        "risk target",
        "risk managed",
        "masi",
    )
    if any(term in normalized for term in domain_terms):
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
    if any(marker in normalized for marker in help_markers):
        return True

    words = set(normalized.split())
    return bool(words & {"salut", "bonjour", "bonsoir", "hello", "hi", "hey", "salam"}) and bool(
        words & {"aider", "assister", "faire", "commencer"}
    )


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    if numerator == 0:
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


_DOCUMENT_VECTORS = {
    intent: Counter(_tokenize(description))
    for intent, description in INTENT_DESCRIPTIONS.items()
}
