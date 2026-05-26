"""Smoke tests for the local Ollama LLM used by the MASI chatbot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from backend.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from backend.llm.local_ollama_client import generate_local_answer
from backend.chatbot.prompt_builder import build_chat_prompt
from backend.chatbot.response_policy_router import build_response_policy


def test_ollama_connection() -> None:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(
            "Ollama n'est pas joignable. Lance `ollama serve` ou l'application Ollama."
        ) from exc

    installed_models = [item.get("name", "") for item in payload.get("models", [])]
    print("[OK] Connexion Ollama")
    print("Modeles installes:", ", ".join(installed_models) or "aucun")

    if OLLAMA_MODEL not in installed_models:
        print(f"[WARN] Modele attendu absent: {OLLAMA_MODEL}")
        print(f"       Installe-le avec: ollama pull {OLLAMA_MODEL}")


def test_simple_generation() -> None:
    answer = generate_local_answer(
        "Reponds en une phrase courte: que mesure la VaR ?",
        model=OLLAMA_MODEL,
    )
    print("\n[OK] Generation simple")
    print(answer)


def test_mini_rag_prompt() -> None:
    rag_context = """
[Passage 1 | Source: test | Section: VaR]
La VaR est un quantile conditionnel du rendement. Elle ne represente pas une perte maximale garantie.

[Passage 2 | Source: test | Section: Expected Shortfall]
L'Expected Shortfall mesure la perte moyenne attendue au-dela du seuil de VaR.
""".strip()

    dashboard_context = """
Derniere prevision fictive:
- Horizon: 1 jour
- Regime HMM: high volatility
- VaR 5%: information non disponible dans ce test
- Expected Shortfall 5%: information non disponible dans ce test
""".strip()

    user_question = "Explique la difference entre VaR et ES dans ce dashboard."
    policy = build_response_policy(user_question, "definition_query")
    prompt = build_chat_prompt(
        question=user_question,
        intent="definition_query",
        dynamic_context=dashboard_context,
        rag_context=rag_context,
        response_policy=policy,
    )

    answer = generate_local_answer(prompt, model=OLLAMA_MODEL)
    print("\n[OK] Generation avec mini contexte RAG fictif")
    print(answer)


if __name__ == "__main__":
    test_ollama_connection()
    test_simple_generation()
    test_mini_rag_prompt()
