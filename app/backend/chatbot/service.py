"""Controlled MASI chatbot orchestration.

This module owns the chatbot turn pipeline. The stable public import path
`backend.services.chat_service` remains a thin facade over this implementation.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from backend.chatbot.answer_repair import repair_known_masi_errors
from backend.chatbot.chat_turn import ChatTurn
from backend.chatbot.conversation_memory import get_conversation_summary
from backend.chatbot.fallback_responses import get_fallback_response
from backend.chatbot.intent_router import classify_user_intent as _classify_intent
from backend.chatbot.prompt_builder import build_chat_prompt
from backend.chatbot.response_guardrails import apply_guardrails, validate_response
from backend.chatbot.response_policy_router import build_response_policy
from backend.chatbot.routed_context_builder import build_routed_chat_context
from backend.core.config import LLM_BACKEND, OLLAMA_MODEL
from backend.llm.local_ollama_client import generate_local_answer, generate_local_answer_stream


HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_CLIENT = None
_LOAD_ERROR: str | None = None
_repair_known_masi_errors = repair_known_masi_errors


@dataclass(frozen=True)
class ChatServiceConfig:
    """Runtime configuration for a chatbot turn."""

    rag_k: int = 5
    rag_min_score: float = 1.5
    max_question_chars: int = 2000
    max_history_messages: int = 40
    language: str = "fr"
    llm_backend: str = LLM_BACKEND
    model_name: str = OLLAMA_MODEL


def validate_user_question(question: str, max_chars: int = 2000) -> str:
    """Validate and normalize the user question."""
    if not isinstance(question, str):
        raise TypeError("question must be a string.")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question cannot be empty.")
    if len(cleaned) > max_chars:
        raise ValueError(f"question is too long. Maximum allowed length is {max_chars} characters.")
    return cleaned


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_accents.split())


def _is_prediction_overview_question(question: str) -> bool:
    normalized = _normalize_text(question)
    compact = normalized.replace(" ", "")
    prediction_tokens = ("prediction", "predictions", "prevision", "previsions", "forecast", "forecasts")
    asks_prediction = any(token in normalized or token in compact for token in prediction_tokens)
    asks_what = any(token in normalized for token in ("quel", "quelle", "quelles", "quoi", "sont", "donne", "resume"))
    return asks_prediction and (asks_what or len(normalized.split()) <= 5)


def classify_user_intent(question: str) -> str:
    """Classify the chatbot intent while preserving forecast overview behavior."""
    if _is_prediction_overview_question(question):
        return "forecast_query"
    return _classify_intent(question)


def classify_chat_intent(question: str) -> str:
    """Backward-compatible alias."""
    return classify_user_intent(question)


def sanitize_conversation_history(
    conversation_history: list[dict[str, str]] | None,
    max_messages: int = 8,
) -> list[dict[str, str]]:
    """Keep a small clean memory window."""
    if not conversation_history:
        return []
    cleaned: list[dict[str, str]] = []
    for item in conversation_history[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:4000]})
    return cleaned


def _label_for_dashboard_key(key: str) -> str:
    explicit_labels = {
        "displayed_return": "rendement prevu affiche",
        "displayed_var": "VaR 5% affichee",
        "displayed_es": "Expected Shortfall 5% affiche",
        "displayed_mean": "moyenne EGARCH AR(1) affichee",
        "displayed_egarch_volatility": "volatilite EGARCH affichee",
        "displayed_regime": "regime HMM affiche",
        "selected_horizon": "horizon selectionne",
        "target_date": "date cible",
        "run_date": "date du run",
    }
    if key in explicit_labels:
        return explicit_labels[key]
    return key.replace("_", " ")[:90]


def format_dashboard_state_context(current_dashboard_state: dict[str, Any] | None) -> str:
    """Render exact frontend values for prompt injection."""
    if not isinstance(current_dashboard_state, dict) or not current_dashboard_state:
        return ""

    lines: list[str] = []
    for key in sorted(current_dashboard_state):
        value = current_dashboard_state.get(key)
        if value is None or value == "":
            continue
        rendered = str(value).replace("\n", " ").strip()[:500]
        if rendered:
            lines.append(f"- {_label_for_dashboard_key(key)}: {rendered}")

    if not lines:
        return ""

    return (
        "Ces faits viennent directement de l'interface affichee pour ce tour.\n"
        "Utilise-les comme contexte exact, sans inventer de chiffre et sans les recopier en dump.\n"
        "VaR = Value at Risk; ES = Expected Shortfall; HMM = regime de volatilite, pas direction de marche.\n"
        + "\n".join(lines[:28])
    )


def render_conversation_memory_block(history: list[dict[str, str]]) -> str:
    """Render sanitized memory for the prompt."""
    if not history:
        return ""
    compact_summary = get_conversation_summary(history)
    turns = []
    for item in history:
        role = "Utilisateur" if item["role"] == "user" else "Assistant"
        turns.append(f"{role}: {item['content'].replace(chr(10), ' ')[:900]}")
    rendered = "\n".join(turns)
    return f"{compact_summary}\n\nHistorique recent:\n{rendered}" if compact_summary else f"Historique recent:\n{rendered}"


def build_system_instructions() -> str:
    """Stable high-level chatbot instructions for Hugging Face fallback."""
    return (
        "Tu es un assistant specialise dans le MASI Risk Dashboard. "
        "Tu expliques previsions, VaR, ES, regimes HMM, backtests et risk-targeting. "
        "Tu n'inventes jamais de chiffres et tu ne donnes jamais de conseil d'investissement."
    )


def load_huggingface_client():
    """Load the Hugging Face client lazily."""
    global _CLIENT, _LOAD_ERROR
    if _CLIENT is not None:
        return _CLIENT
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)
    try:
        from huggingface_hub import InferenceClient, get_token
    except Exception as exc:
        _LOAD_ERROR = "Le client Hugging Face n'a pas pu etre initialise."
        raise RuntimeError(_LOAD_ERROR) from exc

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or get_token() or None
    if not hf_token:
        _LOAD_ERROR = "Aucun token Hugging Face n'est disponible."
        raise RuntimeError(_LOAD_ERROR)
    _CLIENT = InferenceClient(token=hf_token)
    return _CLIENT


def _resolve_model_name(backend: str, configured_model_name: str | None) -> str:
    if backend == "ollama":
        return configured_model_name or OLLAMA_MODEL
    if backend == "huggingface":
        return configured_model_name if configured_model_name and configured_model_name != OLLAMA_MODEL else HF_MODEL_ID
    return configured_model_name or ""


def generate_huggingface_answer(prompt: str, model: str = HF_MODEL_ID) -> str:
    """Generate an answer with Hugging Face chat completion."""
    client = load_huggingface_client()
    response = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": build_system_instructions()},
            {"role": "user", "content": prompt},
        ],
        max_tokens=900,
        temperature=0.1,
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        return "Aucune reponse exploitable n'a ete produite par le modele."
    message = getattr(choices[0], "message", None)
    answer = getattr(message, "content", "") if message is not None else ""
    return repair_known_masi_errors(answer.strip())


def generate_llm_answer(prompt: str, backend: str | None = None, model_name: str | None = None) -> str:
    """Generate a complete answer with the configured LLM backend."""
    backend_name = (backend or LLM_BACKEND).strip().lower()
    resolved_model_name = _resolve_model_name(backend_name, model_name)
    if backend_name == "ollama":
        return repair_known_masi_errors(generate_local_answer(prompt, model=resolved_model_name))
    if backend_name == "huggingface":
        return generate_huggingface_answer(prompt, model=resolved_model_name)
    raise RuntimeError(f"Backend LLM non supporte: {backend_name}")


def generate_llm_answer_stream(prompt: str, backend: str | None = None, model_name: str | None = None) -> Iterator[str]:
    """Generate an answer stream. Non-Ollama backends return one complete chunk."""
    backend_name = (backend or LLM_BACKEND).strip().lower()
    resolved_model_name = _resolve_model_name(backend_name, model_name)
    if backend_name != "ollama":
        yield generate_llm_answer(prompt, backend=backend_name, model_name=resolved_model_name)
        return
    yield from generate_local_answer_stream(prompt, model=resolved_model_name)


def validate_chatbot_answer(answer: str, context: str, intent: str, question: str = "") -> tuple[bool, list[str]]:
    """Compatibility validator returning the historical tuple shape."""
    result = validate_response(answer, context, intent, question)
    if result.is_valid:
        return True, []
    mapped = ["unsupported_number" if issue == "hallucinated_number" else issue for issue in result.issues]
    return False, mapped


def _empty_context_fallback_answer(intent: str) -> str:
    if intent == "forecast_query":
        return (
            "Cette information n'est pas disponible dans le contexte actuel. "
            "Je peux analyser la prevision des que l'horizon, la date cible, le rendement prevu, la VaR et l'ES sont fournis."
        )
    if intent == "backtest_query":
        return (
            "Cette information n'est pas disponible dans le contexte actuel. "
            "Pour le backtest, il me faut le taux de violation, Kupiec, Christoffersen et le diagnostic ES s'il existe."
        )
    return "Cette information n'est pas disponible dans le contexte actuel."


def _finalize_answer(answer: str, question: str, intent: str, validation_context: str) -> str:
    repaired = repair_known_masi_errors(answer)
    guarded = apply_guardrails(repaired, validation_context, intent, question)
    is_valid, issues = validate_chatbot_answer(guarded, validation_context, intent, question)
    if is_valid or set(issues).issubset({"unsupported_number", "excessive_certainty", "invented_relative_date"}):
        return guarded
    if "financial_advice" in issues:
        policy = build_response_policy(question, intent)
        return policy.direct_answer or guarded
    if "var_es_confusion" in issues:
        return "La VaR est un seuil conditionnel de perte. L'ES mesure la perte moyenne au-dela de ce seuil."
    return _empty_context_fallback_answer(intent)


def _should_start_guided_forecast(question: str, history: list[dict[str, str]]) -> bool:
    q = _normalize_text(question).strip(" !?.")
    q_tokens = set(q.replace("'", " ").split())
    if q in {"commence", "vas-y", "vas y", "go", "start", "aide moi", "guide moi"}:
        return True
    if any(marker in q for marker in ("aide moi", "guide moi", "aide-moi")) and len(q.split()) <= 6:
        return True
    has_acceptance = bool(q_tokens & {"oui", "ok", "daccord", "d", "accord", "vas", "y"})
    has_help = any(marker in q for marker in ("aide", "guide", "commence", "vas y", "vas-y"))
    if has_acceptance and has_help and len(q.split()) <= 8:
        return True
    if q in {"oui", "ok", "d accord", "d'accord"}:
        recent = " ".join(item["content"] for item in history[-3:] if item["role"] == "assistant")
        normalized_recent = _normalize_text(recent)
        return "dashboard masi" in normalized_recent and "prevision 1 jour" in normalized_recent
    return False


def _first_regex_group(text: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _extract_guided_forecast_values(turn: ChatTurn) -> dict[str, str]:
    source = "\n".join(
        part for part in (turn.dashboard_state_context, turn.numeric_context, turn.validation_context) if part
    )
    return {
        "horizon": _first_regex_group(
            source,
            (
                r"horizon selectionne:\s*([^\n]+)",
                r"horizon principal a\s*([0-9]+)\s*jour",
                r"Pour l'horizon\s*([0-9]+)\s*jour",
            ),
        ),
        "target_date": _first_regex_group(
            source,
            (
                r"date cible:\s*([^\n]+)",
                r"date cible est\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            ),
        ),
        "return": _first_regex_group(
            source,
            (
                r"rendement prevu affiche:\s*([^\n]+)",
                r"rendement prevu est\s*([-+]?\d+(?:[.,]\d+)?%)",
            ),
        ),
        "var": _first_regex_group(
            source,
            (
                r"VaR 5% affichee:\s*([^\n]+)",
                r"VaR 5% est\s*([-+]?\d+(?:[.,]\d+)?%)",
            ),
        ),
        "es": _first_regex_group(
            source,
            (
                r"Expected Shortfall 5% affiche:\s*([^\n]+)",
                r"ES 5% est\s*([-+]?\d+(?:[.,]\d+)?%)",
            ),
        ),
        "regime": _first_regex_group(
            source,
            (
                r"regime HMM affiche:\s*([^\n]+)",
                r"regime courant est\s*([^.\n]+)",
                r"Le regime estime est\s*([^.\n]+)",
            ),
        ),
    }


def _guided_forecast_answer(turn: ChatTurn) -> str:
    values = _extract_guided_forecast_values(turn)
    horizon = values["horizon"] or "1"
    lines = [f"On commence par la prevision a {horizon} jour du dashboard MASI."]

    facts = []
    if values["target_date"]:
        facts.append(f"date cible : {values['target_date']}")
    if values["return"]:
        facts.append(f"rendement prevu : {values['return']}")
    if values["var"]:
        facts.append(f"VaR 5% : {values['var']}")
    if values["es"]:
        facts.append(f"ES 5% : {values['es']}")
    if values["regime"]:
        facts.append(f"regime HMM : {values['regime']}")

    if facts:
        lines.append("Les valeurs cles sont : " + "; ".join(facts) + ".")
    else:
        lines.append(
            "Je n'ai pas encore les valeurs exactes affichees dans le contexte de ce tour, "
            "mais la lecture commence normalement par rendement prevu, VaR, ES et regime HMM."
        )

    lines.append(
        "Lecture rapide : le rendement prevu donne la direction estimee, la VaR donne un seuil "
        "conditionnel de perte, l'ES mesure la perte moyenne au-dela de ce seuil, et le regime HMM "
        "decrit la volatilite plutot que la direction du marche."
    )
    lines.append("Ensuite, on peut verifier si le backtest confirme que ces mesures de risque sont bien calibrees.")
    return "\n\n".join(lines)


def _build_turn(
    question: str,
    conversation_history: list[dict[str, str]] | None,
    current_dashboard_state: dict[str, Any] | None,
    config: ChatServiceConfig,
) -> tuple[ChatTurn, Any]:
    cleaned_question = validate_user_question(question, max_chars=config.max_question_chars)
    cleaned_history = sanitize_conversation_history(conversation_history, max_messages=config.max_history_messages)
    dashboard_state_context = format_dashboard_state_context(current_dashboard_state)
    intent = classify_user_intent(cleaned_question)
    if _should_start_guided_forecast(cleaned_question, cleaned_history):
        intent = "forecast_query"

    routed_context = build_routed_chat_context(
        question=cleaned_question,
        intent=intent,
        rag_k=min(max(config.rag_k, 1), 5),
        rag_min_score=config.rag_min_score,
    )
    numeric_context = "" if dashboard_state_context else routed_context.numeric_context
    policy = build_response_policy(cleaned_question, intent)
    validation_context = "\n\n".join(
        part for part in (dashboard_state_context, routed_context.validation_context) if part.strip()
    )
    prompt = build_chat_prompt(
        question=cleaned_question,
        intent=intent,
        dynamic_context=numeric_context,
        rag_context=routed_context.rag_context,
        conversation_summary=render_conversation_memory_block(cleaned_history),
        dashboard_state_context=dashboard_state_context,
        response_policy=policy,
    )
    backend_name = config.llm_backend.strip().lower()
    turn = ChatTurn(
        question=cleaned_question,
        cleaned_history=cleaned_history,
        dashboard_state_context=dashboard_state_context,
        intent=intent,
        rag_context=routed_context.rag_context,
        numeric_context=numeric_context,
        validation_context=validation_context,
        prompt=prompt,
        used_rag=routed_context.used_rag,
        used_dashboard_context=routed_context.used_numeric_context or bool(dashboard_state_context),
        backend_name=backend_name,
        model_name=_resolve_model_name(backend_name, config.model_name),
        response_policy=policy,
    )
    return turn, routed_context


def _response_payload(turn: ChatTurn, answer: str) -> dict[str, object]:
    return {
        "answer": answer,
        "used_rag": turn.used_rag,
        "used_dashboard_context": turn.used_dashboard_context,
        "llm_backend": turn.backend_name,
        "model_name": turn.model_name,
        "intent": turn.intent,
    }


def ask_masi_chatbot(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
    current_dashboard_state: dict[str, Any] | None = None,
    config: ChatServiceConfig = ChatServiceConfig(),
    return_debug: bool = False,
) -> dict[str, object]:
    """Run one chatbot turn and return the API-compatible response dict."""
    turn, routed_context = _build_turn(question, conversation_history, current_dashboard_state, config)

    guided_start = _should_start_guided_forecast(turn.question, turn.cleaned_history)
    direct = _guided_forecast_answer(turn) if guided_start else get_fallback_response(turn.intent, turn.question)
    if direct is not None:
        response = _response_payload(turn, direct)
    elif not turn.response_policy.allow_llm and turn.response_policy.direct_answer:
        response = _response_payload(turn, turn.response_policy.direct_answer)
    else:
        try:
            raw_answer = generate_llm_answer(turn.prompt, backend=config.llm_backend, model_name=config.model_name)
        except RuntimeError:
            raw_answer = _empty_context_fallback_answer(turn.intent)
        response = _response_payload(turn, _finalize_answer(raw_answer, turn.question, turn.intent, turn.validation_context))

    if return_debug:
        response.update(
            {
                "question": turn.question,
                "rag_context": turn.rag_context,
                "dashboard_context": turn.numeric_context,
                "dashboard_state_context": turn.dashboard_state_context,
                "context_routes": list(routed_context.routes),
                "prompt": turn.prompt,
            }
        )
    return response


def stream_masi_chatbot(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
    current_dashboard_state: dict[str, Any] | None = None,
    config: ChatServiceConfig = ChatServiceConfig(),
) -> Iterator[dict[str, object]]:
    """Collect, repair and emit one final chatbot answer for SSE."""
    turn, routed_context = _build_turn(question, conversation_history, current_dashboard_state, config)

    guided_start = _should_start_guided_forecast(turn.question, turn.cleaned_history)
    direct = _guided_forecast_answer(turn) if guided_start else get_fallback_response(turn.intent, turn.question)
    already_yielded = False
    if direct is not None:
        answer = direct
    elif not turn.response_policy.allow_llm and turn.response_policy.direct_answer:
        answer = turn.response_policy.direct_answer
    else:
        chunks: list[str] = []
        try:
            for chunk in generate_llm_answer_stream(turn.prompt, backend=config.llm_backend, model_name=config.model_name):
                chunks.append(chunk)
            raw_answer = "".join(chunks).strip()
        except RuntimeError:
            raw_answer = _empty_context_fallback_answer(turn.intent)
        answer = _finalize_answer(raw_answer, turn.question, turn.intent, turn.validation_context)

    if not already_yielded:
        yield {"type": "delta", "delta": answer}
    done = _response_payload(turn, answer)
    done.update(
        {
            "type": "done",
            "context_routes": list(routed_context.routes),
            "used_dashboard_state": bool(turn.dashboard_state_context),
        }
    )
    yield done


if __name__ == "__main__":
    for sample in ("bonjour", "c'est quoi la VaR ?", "dois-je acheter le MASI ?"):
        print("#" * 80)
        print(sample)
        print(ask_masi_chatbot(sample)["answer"])
