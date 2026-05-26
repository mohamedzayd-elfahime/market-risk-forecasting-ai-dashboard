"""Ollama client for local MASI chatbot inference."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from backend.core.config import (
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_GPU,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
)


DEFAULT_TIMEOUT_SECONDS = 120
OLLAMA_STARTUP_TIMEOUT_SECONDS = 8
_OLLAMA_PROCESS: subprocess.Popen | None = None


def _build_generate_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/generate"


def _build_tags_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/tags"


def is_ollama_available(base_url: str = OLLAMA_BASE_URL, timeout: float = 1.0) -> bool:
    try:
        with urlopen(_build_tags_url(base_url), timeout=timeout):
            return True
    except (OSError, TimeoutError, URLError, HTTPError):
        return False


def ensure_ollama_server() -> bool:
    """Start `ollama serve` in the background when it is not already running."""

    global _OLLAMA_PROCESS

    if is_ollama_available():
        return True

    command = ["ollama", "serve"]
    kwargs: dict[str, object] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        _OLLAMA_PROCESS = subprocess.Popen(command, **kwargs)
    except OSError:
        return False

    deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if is_ollama_available(timeout=0.5):
            return True
        if _OLLAMA_PROCESS.poll() is not None:
            return False
        time.sleep(0.3)

    return is_ollama_available(timeout=0.5)


def _extract_ollama_error(raw_body: bytes) -> str:
    try:
        payload = json.loads(raw_body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return raw_body.decode("utf-8", errors="replace").strip()

    error = payload.get("error")
    return str(error).strip() if error else ""


def generate_local_answer(prompt: str, model: str = "qwen2.5:7b-instruct") -> str:
    """Generate a chatbot answer through the local Ollama HTTP API."""

    cleaned_prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not cleaned_prompt:
        raise ValueError("Le prompt envoye au modele local est vide.")

    if not is_ollama_available(timeout=1.0):
        raise RuntimeError(
            "Ollama n'est pas joignable sur http://localhost:11434. "
            "Lance Ollama avec `ollama serve` ou ouvre l'application Ollama."
        )

    payload = {
        "model": model,
        "prompt": cleaned_prompt,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "num_gpu": OLLAMA_NUM_GPU,
        },
    }

    request = Request(
        _build_generate_url(OLLAMA_BASE_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            raw_body = response.read()
    except HTTPError as exc:
        raw_error = exc.read()
        ollama_error = _extract_ollama_error(raw_error)
        if exc.code == 404 or "not found" in ollama_error.lower():
            raise RuntimeError(
                f"Le modele Ollama '{model}' n'est pas installe. "
                f"Execute: ollama pull {model}"
            ) from exc
        raise RuntimeError(
            "L'appel a Ollama a echoue "
            f"(HTTP {exc.code}). Detail: {ollama_error or 'aucun detail fourni'}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            "Le modele Ollama n'a pas repondu avant le timeout. "
            "Verifie que le modele est bien charge et que la machine n'est pas saturee."
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Ollama n'est pas joignable sur http://localhost:11434. "
            "Lance Ollama avec `ollama serve` ou ouvre l'application Ollama."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "Impossible de contacter Ollama. Verifie que le service local est lance."
        ) from exc

    try:
        data = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama a renvoye une reponse JSON invalide.") from exc

    answer = str(data.get("response", "")).strip()
    if not answer:
        raise RuntimeError("Ollama a renvoye une reponse vide.")

    return answer


def generate_local_answer_stream(
    prompt: str,
    model: str = "qwen2.5:7b-instruct",
) -> Iterator[str]:
    """Stream chatbot answer chunks through the local Ollama HTTP API."""

    cleaned_prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not cleaned_prompt:
        raise ValueError("Le prompt envoye au modele local est vide.")

    if not is_ollama_available(timeout=1.0):
        raise RuntimeError(
            "Ollama n'est pas joignable sur http://localhost:11434. "
            "Lance Ollama avec `ollama serve` ou ouvre l'application Ollama."
        )

    payload = {
        "model": model,
        "prompt": cleaned_prompt,
        "stream": True,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "num_gpu": OLLAMA_NUM_GPU,
        },
    }

    request = Request(
        _build_generate_url(OLLAMA_BASE_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Ollama a renvoye un flux JSON invalide.") from exc

                if data.get("error"):
                    raise RuntimeError(str(data["error"]))

                chunk = str(data.get("response", ""))
                if chunk:
                    yield chunk

                if data.get("done"):
                    break
    except HTTPError as exc:
        raw_error = exc.read()
        ollama_error = _extract_ollama_error(raw_error)
        if exc.code == 404 or "not found" in ollama_error.lower():
            raise RuntimeError(
                f"Le modele Ollama '{model}' n'est pas installe. "
                f"Execute: ollama pull {model}"
            ) from exc
        raise RuntimeError(
            "L'appel a Ollama a echoue "
            f"(HTTP {exc.code}). Detail: {ollama_error or 'aucun detail fourni'}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            "Le modele Ollama n'a pas repondu avant le timeout. "
            "Verifie que le modele est bien charge et que la machine n'est pas saturee."
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Ollama n'est pas joignable sur http://localhost:11434. "
            "Lance Ollama avec `ollama serve` ou ouvre l'application Ollama."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "Impossible de contacter Ollama. Verifie que le service local est lance."
        ) from exc
