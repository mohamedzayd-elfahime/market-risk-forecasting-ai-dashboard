"""Response policy routing for the MASI chatbot.

This module does not write normal chatbot answers. It describes how the
generative answer must be constrained for a given question and intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResponsePolicy:
    mode: str
    required_context: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    must_mention: list[str] = field(default_factory=list)
    allow_llm: bool = True
    direct_answer: str | None = None


def build_response_policy(question: str, intent: str) -> ResponsePolicy:
    """Build response constraints for the current chatbot turn."""
    q = _normalize(question)

    if _is_investment_advice(q):
        return ResponsePolicy(
            mode="investment_advice_refusal",
            required_context=[],
            forbidden_claims=["buy", "sell", "investment recommendation"],
            must_mention=[],
            allow_llm=False,
            direct_answer=(
                "Je ne peux pas donner de conseil d'investissement personnalise "
                "ni te dire quoi acheter, vendre ou allouer. Je peux en revanche "
                "t'aider a interpreter les previsions, la VaR, l'Expected Shortfall, "
                "le regime HMM et les resultats de backtesting du dashboard."
            ),
        )

    if _is_horizon_question(q):
        return ResponsePolicy(
            mode="horizon_explanation",
            required_context=["canonical_rules", "forecast_context"],
            forbidden_claims=["Monte Carlo", "independent multi-horizon model"],
            must_mention=["operational extension", "sqrt horizon scaling"],
            allow_llm=True,
        )

    if intent == "forecast_query":
        return ResponsePolicy(
            mode="numeric_forecast",
            required_context=["numeric_forecast_context"],
            forbidden_claims=["investment advice", "guaranteed result"],
            must_mention=["horizon", "target date", "forecasted return", "VaR", "ES"],
            allow_llm=True,
        )

    if intent == "backtest_query":
        return ResponsePolicy(
            mode="backtest",
            required_context=["backtest_context"],
            forbidden_claims=[
                "model is guaranteed",
                "future calibration is certain",
                "expected violation rate is a p-value",
                "expected violation rate corresponds to Kupiec p-value",
            ],
            must_mention=[
                "violation rate",
                "expected violation rate is the theoretical expected VaR violation rate, not a p-value",
                "Kupiec",
                "Christoffersen",
                "ES diagnostic if available",
            ],
            allow_llm=True,
        )

    if _is_var_es_question(q):
        return ResponsePolicy(
            mode="risk_measure_definition",
            required_context=["rag_rules"],
            forbidden_claims=["VaR is maximum loss", "ES is maximum loss"],
            must_mention=[
                "VaR is a conditional loss threshold",
                "ES is average loss beyond VaR",
            ],
            allow_llm=True,
        )

    if _is_hmm_question(q) or intent == "strategy_query":
        return ResponsePolicy(
            mode="regime_or_strategy",
            required_context=["canonical_rules"],
            forbidden_claims=[
                "HMM confirms market direction",
                "high volatility means bearish trend",
            ],
            must_mention=[
                "HMM is a volatility regime",
                "direction comes from forecasted return",
            ],
            allow_llm=True,
        )

    return ResponsePolicy(
        mode="general_controlled",
        required_context=[],
        forbidden_claims=[
            "investment advice",
            "guaranteed result",
            "VaR is maximum loss",
            "HMM confirms market direction",
        ],
        must_mention=[],
        allow_llm=True,
    )


def format_response_policy_for_prompt(policy: ResponsePolicy) -> str:
    """Render a policy as prompt constraints, not as answer prose."""
    lines = [
        f"Mode de controle: {policy.mode}",
        f"LLM autorise: {'oui' if policy.allow_llm else 'non'}",
    ]
    if policy.required_context:
        lines.append("Contexte requis: " + ", ".join(policy.required_context))
    if policy.forbidden_claims:
        lines.append("Affirmations interdites: " + "; ".join(policy.forbidden_claims))
    if policy.must_mention:
        lines.append("Points a couvrir si pertinents: " + "; ".join(policy.must_mention))
    lines.append(
        "Ces contraintes guident la reponse. Ne les recopie pas sous forme de liste, "
        "et ne les presente pas comme une politique interne."
    )
    return "\n".join(lines)


def _normalize(text: str) -> str:
    return (
        " ".join(text.lower().split())
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ç", "c")
        .replace("Ã©", "e")
        .replace("Ã¨", "e")
        .replace("Ãª", "e")
        .replace("Ã ", "a")
        .replace("Ã§", "c")
    )


def _is_investment_advice(q: str) -> bool:
    advice_patterns = (
        "acheter",
        "vendre",
        "investir maintenant",
        "quelle position prendre",
        "allocation reelle",
        "dois-je investir",
        "dois je investir",
        "dois-je acheter",
        "dois je acheter",
        "dois-je vendre",
        "dois je vendre",
        "recommandation",
        "recommande",
        "buy",
        "sell",
        "hold",
    )
    if "allocation reelle" in q and any(
        token in q for token in ("est ce que", "est une", "simule", "simulation", "veut dire")
    ):
        return False
    return any(pattern in q for pattern in advice_patterns)


def _is_var_es_question(q: str) -> bool:
    return any(token in q for token in ("var", "value at risk", "expected shortfall", "shortfall", " es "))


def _is_hmm_question(q: str) -> bool:
    return any(token in q for token in ("hmm", "regime", "volatilite", "volatility"))


def _is_horizon_question(q: str) -> bool:
    has_horizon = any(token in q for token in ("10 jours", "25 jours", "10j", "25j", "horizon"))
    asks_method = any(token in q for token in ("monte carlo", "calcule", "methode", "extension", "scaling"))
    return has_horizon and asks_method
