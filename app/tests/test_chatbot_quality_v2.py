from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.chatbot import service as chat_service
from backend.chatbot.fallback_responses import get_fallback_response, greeting_response, short_ack_response
from backend.chatbot.intent_router import classify_user_intent
from backend.chatbot.prompt_builder import build_chat_prompt
from backend.chatbot.response_guardrails import apply_guardrails, validate_response
from backend.chatbot.response_policy_router import build_response_policy


def test_intent_router_core_cases():
    assert classify_user_intent("bonjour") == "help_request"
    assert classify_user_intent("c'est quoi la VaR ?") == "definition_query"
    assert classify_user_intent("quelle est la prevision 1 jour ?") == "forecast_query"
    assert classify_user_intent("donne la p-value Kupiec") == "backtest_query"
    assert classify_user_intent("comment marche le risk-targeting ?") == "strategy_query"
    assert classify_user_intent("quelle est la meteo a Paris ?") == "out_of_scope"


def test_fallbacks_are_limited_to_tiny_social_turns():
    assert "masi" in greeting_response().lower()
    assert len(short_ack_response()) < 40
    assert get_fallback_response("help_request", "bonjour") is not None
    assert get_fallback_response("help_request", "aide moi") is not None
    assert get_fallback_response("help_request", "salut comment tu peux m assister") is not None
    assert get_fallback_response("definition_query", "c'est quoi la VaR ?") is None
    assert get_fallback_response("out_of_scope", "meteo Paris") is None


def test_response_policy_refuses_investment_advice_before_llm():
    policy = build_response_policy("dois-je acheter le MASI ?", "strategy_query")
    assert policy.allow_llm is False
    assert policy.direct_answer
    assert "conseil" in policy.direct_answer.lower()


def test_response_policy_controls_var_es_without_direct_answer():
    policy = build_response_policy("c'est quoi la VaR et l'ES ?", "definition_query")
    assert policy.allow_llm is True
    assert policy.direct_answer is None
    assert "VaR is maximum loss" in policy.forbidden_claims
    assert "ES is average loss beyond VaR" in policy.must_mention


def test_prompt_injects_policy_constraints():
    policy = build_response_policy("les horizons 10/25 jours sont Monte Carlo ?", "model_query")
    prompt = build_chat_prompt(
        question="les horizons 10/25 jours sont Monte Carlo ?",
        intent="model_query",
        rag_context="Les horizons longs sont des extensions operationnelles.",
        response_policy=policy,
    )
    assert "POLITIQUE DE REPONSE" in prompt
    assert "Monte Carlo" in prompt
    assert "sqrt horizon scaling" in prompt


def test_guardrails_correct_financial_advice():
    corrected = apply_guardrails(
        response="Je te conseille d'acheter le MASI.",
        context="VaR 5% : -1.5%",
        intent="strategy_query",
        question="que faire ?",
    )
    assert "conseil d'investissement" in corrected.lower()
    assert "je te conseille" not in corrected.lower()


def test_guardrails_detect_hallucinated_percentage():
    result = validate_response(
        response="La VaR est de -9.99%.",
        context="VaR 5% : -1.559%",
        intent="forecast_query",
        question="VaR actuelle ?",
    )
    assert not result.is_valid
    assert "hallucinated_number" in result.issues


def test_help_request_does_not_call_llm_or_dump_dashboard(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called for short help requests")

    monkeypatch.setattr(chat_service, "generate_llm_answer", fail_if_called)
    response = chat_service.ask_masi_chatbot("saut comment tu peux m assister")
    answer = response["answer"].lower()
    assert response["intent"] == "help_request"
    assert "masi" in answer
    assert "0.323" not in answer
    assert "kupiec" not in answer
    assert "christoffersen" not in answer


def test_guardrails_replace_repetitive_help_dump():
    repeated = (
        "Je suis ravi de vous aider avec le MASI Risk Dashboard. "
        "Le rendement prevu est de 0.323%. "
        "La VaR a 5% est de -1.45%. "
        "Le ratio de violation ES indique une perte moyenne de -0.0198% sur les tests. "
        "Le ratio de violation ES indique une perte moyenne de -0.0198% sur les tests. "
        "Le ratio de violation ES indique une perte moyenne de -0.0198% sur les tests."
    )
    corrected = apply_guardrails(
        response=repeated,
        context="rendement prevu affiche: 0.323%; VaR 5% affichee: -1.45%",
        intent="help_request",
        question="comment tu peux m assister ?",
    )
    assert "je peux t'aider" in corrected.lower()
    assert "0.323" not in corrected


def test_ask_refuses_investment_advice_without_calling_llm(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called for investment advice")

    monkeypatch.setattr(chat_service, "generate_llm_answer", fail_if_called)
    response = chat_service.ask_masi_chatbot("dois-je acheter le MASI ?")
    assert response["intent"] in {"strategy_query", "forecast_query", "general_query"}
    assert "conseil" in response["answer"].lower()


def test_stream_returns_single_final_delta_for_direct_refusal(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called for investment advice")

    monkeypatch.setattr(chat_service, "generate_llm_answer_stream", fail_if_called)
    events = list(chat_service.stream_masi_chatbot("dois-je acheter le MASI ?"))
    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["delta"] == events[1]["answer"]
