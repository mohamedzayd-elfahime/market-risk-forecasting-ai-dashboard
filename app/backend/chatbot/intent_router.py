"""Embedding-based intent router for the MASI Risk Dashboard chatbot.

Each intent is represented by a small set of annotated route examples. At
runtime, a lightweight local embedding model maps the user question and the
intent examples into the same vector space; the router picks the closest intent
centroid by cosine similarity. The output still matches the rest of the chatbot
architecture:

user request -> intent router -> static RAG / forecast / backtest route
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np


INTENT_SIMILARITY_THRESHOLD = 0.30


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


INTENT_EXAMPLES: dict[str, tuple[str, ...]] = {
    "help_request": (
        INTENT_DESCRIPTIONS["help_request"],
        "bonjour",
        "salut",
        "aide moi a commencer",
        "guide moi dans le dashboard",
        "que peux tu faire",
        "par ou commencer dans le dashboard MASI",
    ),
    "definition_query": (
        INTENT_DESCRIPTIONS["definition_query"],
        "c'est quoi la VaR",
        "explique l'Expected Shortfall",
        "quelle est la difference entre VaR et ES",
        "que signifie regime HMM",
        "definis EGARCH",
        "explique le risk targeting",
    ),
    "forecast_query": (
        INTENT_DESCRIPTIONS["forecast_query"],
        "quelle est la prevision actuelle",
        "resume la prediction a 1 jour",
        "donne le rendement prevu du MASI",
        "quelle est la VaR actuelle",
        "quelle est l'ES actuelle",
        "quel est le regime courant",
        "montre la prevision horizon 10 jours",
    ),
    "backtest_query": (
        INTENT_DESCRIPTIONS["backtest_query"],
        "donne les resultats du backtest",
        "quelle est la p-value Kupiec",
        "explique le test de Christoffersen",
        "combien de violations VaR",
        "le modele est il bien calibre",
        "compare predit et realise",
    ),
    "strategy_query": (
        INTENT_DESCRIPTIONS["strategy_query"],
        "comment marche la strategie risk managed",
        "comment marche le risk-targeting",
        "comment fonctionne le risk-targeting",
        "explique le poids simule",
        "quelle exposition est utilisee",
        "que signifie le drawdown",
        "explique la richesse finale simulee",
        "dois-je acheter le MASI",
        "dois-je vendre le MASI",
    ),
    "model_query": (
        INTENT_DESCRIPTIONS["model_query"],
        "quelle est l'architecture du modele",
        "comment fonctionne le LSTM",
        "comment fonctionne EGARCH",
        "explique le modele HMM",
        "comment le chatbot construit sa reponse",
        "quelle configuration du LLM",
    ),
    "data_query": (
        INTENT_DESCRIPTIONS["data_query"],
        "quelle est la source des donnees",
        "quelle periode couvre le dataset",
        "quelle est la frequence des donnees",
        "comment les donnees sont nettoyees",
        "explique le preprocessing",
        "quelles variables sont utilisees",
    ),
    "out_of_scope": (
        INTENT_DESCRIPTIONS["out_of_scope"],
        "quelle est la meteo a Paris",
        "donne une recette de cuisine",
        "traduis ce texte",
        "parle moi de football",
        "quel film regarder ce soir",
        "actualites politiques",
    ),
}


def classify_user_intent(question: str) -> str:
    """Classify the user request with lightweight embedding centroids."""

    if not isinstance(question, str) or not question.strip():
        return "help_request"
    if _is_short_help_question(question):
        return "help_request"

    try:
        intent = _classify_with_embeddings(question)
        if intent is not None:
            return intent
    except Exception:
        # Keep the chatbot available if the local embedding model is missing.
        pass

    return _classify_with_lexical_similarity(question)


def _classify_with_embeddings(question: str) -> str | None:
    query_embedding = _encode_texts((question.strip(),))[0]
    centroids = _intent_centroids()
    scored = [
        (intent, _vector_cosine_similarity(query_embedding, centroid))
        for intent, centroid in centroids.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    best_intent, best_score = scored[0]
    if best_score < INTENT_SIMILARITY_THRESHOLD:
        return "out_of_scope"
    return best_intent


@lru_cache(maxsize=1)
def _intent_centroids() -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for intent, examples in INTENT_EXAMPLES.items():
        embeddings = _encode_texts(examples)
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[intent] = centroid / norm if norm else centroid
    return centroids


def _encode_texts(texts: tuple[str, ...]) -> np.ndarray:
    from backend.chatbot.rag.retriever import load_embeddings

    embeddings = load_embeddings().embed_documents(list(texts))
    return np.asarray(embeddings, dtype=np.float32)


def _vector_cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def warm_intent_router() -> None:
    """Precompute intent centroids during application warmup."""
    _intent_centroids()


def _classify_with_lexical_similarity(question: str) -> str:
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
