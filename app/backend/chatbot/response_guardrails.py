"""
Response Guardrails for the MASI Risk Dashboard Chatbot.

Role:
- Validate the LLM response BEFORE sending it to the user.
- Detect and correct common failure modes:
    1. Numbers present in answer but absent from context (hallucination).
    2. Financial advice in the response.
    3. VaR / ES confusion (VaR described as an average loss).
    4. "Information indisponible" returned for a help_request.
    5. Forecast summary returned for a definition question.
    6. Invented relative dates (aujourd'hui, demain, semaine prochaine).

If a rule is violated, the validator returns a corrected response
(either from fallback_responses or a safe generic refusal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_PERCENT_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?\s*%")

_FINANCIAL_ADVICE_PATTERNS = (
    "tu dois acheter",
    "tu dois vendre",
    "je recommande d'acheter",
    "je recommande de vendre",
    "achète le masi",
    "vends le masi",
    "je te conseille d'acheter",
    "je te conseille de vendre",
    "il faut acheter",
    "il faut vendre",
)

_VAR_ES_CONFUSION_PATTERNS = (
    "var est la perte moyenne",
    "var mesure la perte moyenne",
    "value at risk est la perte moyenne",
    "var correspond à la perte moyenne",
)

_CERTAINTY_PATTERNS = (
    "va certainement",
    "est certain",
    "garanti",
    "sans risque",
    "perte maximale garantie",
    "va sûrement",
)

_RELATIVE_DATE_PATTERNS = (
    "aujourd'hui",
    "aujourd hui",
    "demain",
    "semaine prochaine",
    "mois prochain",
)

_UNAVAILABLE_MARKERS = (
    "cette information n'est pas disponible dans le contexte actuel",
    "cette information n est pas disponible",
)

_FORECAST_LEAD_PATTERNS = (
    "voici la dernière prévision",
    "voici la derniere prevision",
    "voici les dernières prévisions",
    "voici les dernieres previsions",
)

_HELP_DUMP_PATTERNS = (
    "rendement prevu",
    "rendement pr",
    "var a 5%",
    "var ",
    "expected shortfall",
    "p-value",
    "kupiec",
    "christoffersen",
    "sharpe",
    "violations",
    "richesse finale",
    "drawdown",
)

_EXPECTED_VIOLATION_CONFUSION_PATTERNS = (
    "expected violation rate est une p-value",
    "expected violation rate est la p-value",
    "expected violation rate correspond a la p-value",
    "expected violation rate correspond a une p-value",
    "niveau de violation attendu est une p-value",
    "niveau de violation attendu est la p-value",
    "niveau de violation attendu correspond a la p-value",
    "taux de violation attendu est une p-value",
    "taux de violation attendu est la p-value",
    "taux de violation attendu correspond a la p-value",
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    is_valid: bool
    issues: list[str] = field(default_factory=list)
    corrected_response: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    t = text.lower()
    t = t.replace("é", "e").replace("è", "e").replace("ê", "e")
    t = t.replace("à", "a").replace("ç", "c").replace("û", "u")
    return re.sub(r"\s+", " ", t).strip()


def _extract_percentages(text: str) -> set[str]:
    """Extract all percentage values, normalized as float-strings for comparison."""
    results = set()
    for match in _PERCENT_RE.findall(text):
        # Extract the numeric part (everything before the %)
        num_part = match.rstrip("%").strip().replace(",", ".")
        try:
            val = float(num_part)
            # Standardize to 4 decimal places for comparison to handle rounding/precision
            results.add(f"{val:.4f}")
        except ValueError:
            # Fallback to literal if not a float (unlikely given regex)
            results.add(num_part)
    return results


def _has_unsupported_numbers(response: str, context: str) -> bool:
    """True if the response contains % values absent from the context."""
    context_pcts = _extract_percentages(context)
    response_pcts = _extract_percentages(response)
    # Only flag if there are numbers in response that are NOT in context
    hallucinated = response_pcts - context_pcts
    return len(hallucinated) > 0


def _is_repetitive_response(response: str, max_repeats: int = 3) -> bool:
    parts = re.split(r"(?<=[.!?])\s+|\n+", response.strip())
    counts: dict[str, int] = {}
    for part in parts:
        normalized = _normalise(part)
        if len(normalized) < 28:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
        if counts[normalized] >= max_repeats:
            return True
    return False


def _looks_like_help_context_dump(response: str) -> bool:
    normalized = _normalise(response)
    marker_count = sum(1 for marker in _HELP_DUMP_PATTERNS if marker in normalized)
    return marker_count >= 3 or len(_extract_percentages(response)) >= 4


def _has_expected_violation_confusion(response: str) -> bool:
    normalized = _normalise(response)
    if any(pattern in normalized for pattern in _EXPECTED_VIOLATION_CONFUSION_PATTERNS):
        return True
    expected_terms = (
        "expected violation rate",
        "niveau de violation attendu",
        "taux de violation attendu",
    )
    return (
        any(term in normalized for term in expected_terms)
        and "p-value" in normalized
        and any(link in normalized for link in ("correspond", " est ", "equivaut", "p value de kupiec"))
    )


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_response(response: str, context: str, intent: str, question: str = "") -> GuardrailResult:
    """
    Validate an LLM response before sending to the frontend.

    Parameters
    ----------
    response : str
        The raw LLM response.
    context : str
        All context injected into the prompt (dynamic + RAG).
    intent : str
        The classified intent.
    question : str
        The original user question (used for extra checks).

    Returns
    -------
    GuardrailResult
        .is_valid      — True if the response passes all checks.
        .issues        — list of rule IDs that fired.
        .corrected_response — replacement text if is_valid is False.
    """
    issues: list[str] = []
    r_lower = _normalise(response)
    q_lower = _normalise(question)

    # Rule 1 — financial advice
    if any(pat in r_lower for pat in _FINANCIAL_ADVICE_PATTERNS):
        issues.append("financial_advice")

    # Rule 2 — hallucinated percentages
    if _has_unsupported_numbers(response, context):
        issues.append("hallucinated_number")

    # Rule 3 — excessive certainty
    if any(pat in r_lower for pat in _CERTAINTY_PATTERNS):
        issues.append("excessive_certainty")

    # Rule 4 — VaR/ES confusion
    if any(pat in r_lower for pat in _VAR_ES_CONFUSION_PATTERNS):
        issues.append("var_es_confusion")

    if intent == "backtest_query" and _has_expected_violation_confusion(response):
        issues.append("expected_violation_rate_confusion")

    # Rule 5 — "indisponible" returned for a help request
    if intent == "help_request" and any(m in r_lower for m in _UNAVAILABLE_MARKERS):
        issues.append("help_unavailable_mismatch")

    if intent == "help_request" and _looks_like_help_context_dump(response):
        issues.append("help_context_dump")

    # Rule 6 — forecast summary returned for a definition question
    if intent == "definition_query" and any(pat in r_lower for pat in _FORECAST_LEAD_PATTERNS):
        issues.append("forecast_for_definition")

    # Rule 7 — invented relative dates not present in context
    ctx_lower = _normalise(context)
    invented_dates = [
        pat for pat in _RELATIVE_DATE_PATTERNS
        if pat in r_lower and pat not in ctx_lower and pat not in q_lower
    ]
    if invented_dates:
        issues.append("invented_relative_date")

    if _is_repetitive_response(response):
        issues.append("repetitive_response")

    if not issues:
        return GuardrailResult(is_valid=True)

    # --- Compute corrected response ---
    corrected = _build_correction(response, intent, issues)
    return GuardrailResult(is_valid=False, issues=issues, corrected_response=corrected)


# ---------------------------------------------------------------------------
# Correction logic
# ---------------------------------------------------------------------------

def _build_correction(original_response: str, intent: str, issues: list[str]) -> str:
    """Build a safe replacement response."""
    from backend.chatbot.fallback_responses import greeting_response, investment_advice_refusal

    if "financial_advice" in issues:
        return investment_advice_refusal()

    if "help_context_dump" in issues or (intent == "help_request" and "repetitive_response" in issues):
        return greeting_response()

    if "help_unavailable_mismatch" in issues:
        return original_response.strip()

    if "forecast_for_definition" in issues:
        return (
            "Je vais répondre à ta question de définition sans afficher les prévisions. "
            "Si tu veux consulter la dernière prévision, demande : 'résume la prévision 1 jour'."
        )

    if "var_es_confusion" in issues:
        return (
            "Correction : la VaR n'est pas une perte moyenne. "
            "La VaR est un quantile de perte conditionnel ; "
            "l'Expected Shortfall (ES) mesure la perte moyenne au-delà de ce seuil."
        )

    if "expected_violation_rate_confusion" in issues:
        correction = (
            "Correction : le niveau de violation attendu correspond au taux theorique attendu "
            "de violations VaR pour le niveau alpha, pas a une p-value. "
            "Les p-values sont celles des tests de Kupiec, de Christoffersen et du diagnostic ES."
        )
        return f"{original_response.strip()}\n\n{correction}"

    if "hallucinated_number" in issues:
        return original_response.strip()

    return original_response.strip()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def apply_guardrails(
    response: str,
    context: str,
    intent: str,
    question: str = "",
) -> str:
    """
    Apply guardrails and return either the original response or the corrected one.

    This is the main entry point for chat_service.py.
    """
    result = validate_response(response, context, intent, question)
    if result.is_valid:
        return response
    if result.issues and set(result.issues).issubset({"hallucinated_number", "help_unavailable_mismatch"}):
        return response.strip()
    return result.corrected_response or response
