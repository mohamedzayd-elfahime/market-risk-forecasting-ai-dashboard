"""
Prompt Builder for the MASI Risk Dashboard Chatbot.

Role:
- Build the final LLM prompt from structured inputs.
- Tailor the system instructions to the classified intent.
- Inject dynamic context and RAG context only when relevant.
- Never allow the LLM to hallucinate: constraints are embedded in the prompt.

Usage:
    from backend.chatbot.prompt_builder import build_chat_prompt
    prompt = build_chat_prompt(
        question="c'est quoi la VaR ?",
        intent="definition_query",
        dynamic_context="",
        rag_context="...",
        conversation_summary="",
    )
"""

from __future__ import annotations

from backend.chatbot.response_policy_router import (
    ResponsePolicy,
    format_response_policy_for_prompt,
)

# ---------------------------------------------------------------------------
# System prompt base
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
Tu es un assistant spécialisé dans le MASI Risk Dashboard.
Tu expliques les prévisions de rendement, la VaR, l'Expected Shortfall, \
les modèles économétriques (EGARCH, GJR-GARCH), les modèles deep learning (LSTM-Ridge), \
le backtesting statistique et économique, et les stratégies de risk-targeting HMM-LSTM.

Règles absolues :
1. Tu réponds uniquement à la question posée.
2. Tu ne dois jamais inventer de chiffres ou de valeurs numériques.
3. Tu utilises uniquement les informations présentes dans le contexte fourni.
4. Si une valeur précise est demandée mais absente du contexte, réponds : \
"Cette information n'est pas disponible dans le contexte actuel."
5. Tu ne donnes jamais de conseil d'investissement personnalisé.
6. Tu distingues clairement : définition / prévision / interprétation / backtesting / stratégie.
7. Tu ne dis jamais "tu dois acheter", "tu dois vendre", "je recommande d'acheter/vendre".
8. Tu réponds en français, de manière claire, concise et professionnelle.
9. Tu n'inventes jamais de dates, de p-values, de Sharpe ou de drawdown.
10. Si le contexte contient des valeurs, tu les cites exactement. \
Si elles sont absentes, tu dis qu'elles ne sont pas disponibles.\
11. Tu ne recopies jamais le contexte sous forme de dump cle-valeur. Tu synthétises en phrases courtes.
12. Tu ne renommes jamais une metrique technique si tu n'es pas sûr de son nom.
13. VaR signifie toujours Value at Risk. Ne l'appelle jamais "Volatility of Returns".
14. "Rendement prevu" ou "return forecast" n'est pas un rendement historique.
15. La VaR est un seuil conditionnel de perte, pas une perte attendue ni une perte garantie.
16. Ne parle pas de "notre portefeuille"; parle du MASI, du dashboard ou de la simulation.
"""


# ---------------------------------------------------------------------------
# Intent-specific instructions
# ---------------------------------------------------------------------------

_INTENT_INSTRUCTIONS: dict[str, str] = {
    "help_request": """\
L'utilisateur demande de l'aide ou de la guidance.
INSTRUCTION : Ne dis jamais "Cette information n'est pas disponible".
Réponds avec une aide guidée, mais ne répète pas le même menu si la conversation montre que tu l'as déjà proposé.
Si l'utilisateur accepte une guidance ("oui", "guide moi", "aide moi"), continue avec une lecture concrète du dashboard en utilisant l'état exact disponible.
Quand une valeur de rendement, VaR, ES ou volatilite vient du dashboard sous forme decimale, presente-la en pourcentage lisible en gardant la valeur source comme reference si utile.
Propose de commencer par :
- la prévision 1 jour (rendement prévu, VaR, ES, régime HMM),
- le backtesting (Kupiec, Christoffersen),
- la stratégie de risk-targeting.
Sois chaleureux et orienté vers l'action.\
""",

    "definition_query": """\
L'utilisateur pose une question de définition ou d'explication.
INSTRUCTION : Réponds DIRECTEMENT à la question. Évite les introductions générales sur le MASI ou le dashboard sauf si c'est le sujet principal.
Explique les concepts de manière pédagogique.
N'utilise pas les dernières prévisions sauf demande explicite.
Utilise le contexte RAG pour les détails techniques.\
""",

    "forecast_query": """\
L'utilisateur demande des informations sur la prévision actuelle du MASI.
INSTRUCTION : Utilise les valeurs exactes du contexte dynamique ci-dessous.
Mentionne l'horizon temporel (1 jour).
Rappelle que la VaR et l'ES sont des mesures de risque, pas des recommandations.
Ne déduis pas de direction de prix à partir du régime HMM seul.
Si le contexte dynamique est vide, dis que la prévision n'est pas disponible.\
""",

    "backtest_query": """\
L'utilisateur pose une question sur le backtesting du modèle.
INSTRUCTION : CONCENTRE-TOI UNIQUEMENT sur le backtesting statistique (Kupiec, Christoffersen, violations).
N'inclus pas les prévisions actuelles ou la stratégie de risk-targeting sauf demande explicite.
Explique les p-values, les violations de VaR et la calibration de l'ES.
N'invente pas de p-values. Utilise uniquement les valeurs présentes dans le contexte.\
""",

    "strategy_query": """\
L'utilisateur pose une question sur la stratégie de risk-targeting ou le régime HMM.
INSTRUCTION : Explique l'exposition, le régime HMM, le budget de risque et la logique
de réduction du risque basée sur la VaR prédite.
Ne dis jamais "il faut acheter" ou "il faut vendre".
Le poids risk-managed affiché est une simulation historique, pas une recommandation réelle.\
""",

    "model_query": """\
L'utilisateur pose une question sur les modèles utilisés (EGARCH, GJR-GARCH, LSTM-Ridge).
INSTRUCTION : Explique l'architecture ou les différences entre les modèles.
Si l'utilisateur compare les résultats actuels, utilise le contexte dynamique fourni.
Si aucun résultat comparatif n'est disponible dans le contexte, dis-le clairement.\
""",

    "data_query": """\
L'utilisateur pose une question sur les données utilisées dans le dashboard.
INSTRUCTION : Réponds à partir du contexte RAG sur les données (source, période, fréquence,
preprocessing). N'invente pas de dates ou de métadonnées absentes du contexte.\
""",

    "out_of_scope": """\
L'utilisateur pose une question hors du périmètre du dashboard MASI.
INSTRUCTION : Réponds poliment que tu es spécialisé dans le MASI Risk Dashboard.
Propose de revenir sur les prévisions, la VaR, l'ES, le backtest ou la stratégie.\
""",
}

_DEFAULT_INSTRUCTION = """\
INSTRUCTION : Réponds à la question posée en utilisant uniquement le contexte fourni.
N'invente pas de valeurs numériques. Réponds en français, de manière concise.\
"""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_chat_prompt(
    question: str,
    intent: str,
    dynamic_context: str = "",
    rag_context: str = "",
    conversation_summary: str = "",
    dashboard_state_context: str = "",
    response_policy: ResponsePolicy | None = None,
) -> str:
    """
    Build the final LLM prompt.

    Parameters
    ----------
    question : str
        The cleaned user question.
    intent : str
        The classified intent (from intent_router).
    dynamic_context : str
        Formatted routed numeric context for forecast/backtest questions.
        Empty string means "do not inject".
    rag_context : str
        Retrieved RAG passages (from retriever).
        Empty string means "no RAG found".
    conversation_summary : str
        Short summary of recent conversation (from conversation_memory).
        Empty string means first turn.

    Returns
    -------
    str
        A fully assembled prompt ready to send to the LLM.
    """
    intent_instruction = _INTENT_INSTRUCTIONS.get(intent, _DEFAULT_INSTRUCTION)
    policy_block = (
        format_response_policy_for_prompt(response_policy)
        if response_policy is not None
        else "Aucune politique specifique supplementaire."
    )

    # Build context block
    context_parts: list[str] = []

    if dashboard_state_context.strip():
        context_parts.append(
            "[ETAT EXACT DU DASHBOARD - prioritaire]\n"
            f"{dashboard_state_context.strip()}\n"
            "Instruction: utilise ces faits pour repondre naturellement. Ne liste pas toutes les cles. Ne cree pas de nouveaux noms de metriques."
        )

    if dynamic_context.strip() and intent not in ("definition_query", "out_of_scope"):
        context_parts.append(f"[Contexte chiffre — route specialisee]\n{dynamic_context.strip()}")

    if rag_context.strip() and intent not in ("help_request", "out_of_scope"):
        context_parts.append(f"[Contexte documentaire — base RAG]\n{rag_context.strip()}")

    if conversation_summary.strip():
        context_parts.append(f"[Résumé de la conversation]\n{conversation_summary.strip()}")

    context_block = "\n\n".join(context_parts) if context_parts else "Aucun contexte spécifique disponible."

    prompt = f"""SYSTEM:
{_BASE_SYSTEM_PROMPT}

INTENTION DÉTECTÉE: {intent}

{intent_instruction}

POLITIQUE DE REPONSE:
{policy_block}

CONTEXTE:
{context_block}

QUESTION UTILISATEUR: {question.strip()}

RÉPONSE:"""

    return prompt
