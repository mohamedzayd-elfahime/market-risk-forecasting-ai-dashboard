"""Chat routes."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.schemas.api_schema import ChatAskRequest, ChatAskResponse
from backend.services.chat_service import ask_masi_chatbot, stream_masi_chatbot


router = APIRouter(prefix="/chat", tags=["chat"])


def _looks_like_service_unavailable(answer: str) -> bool:
    lowered = answer.lower()
    markers = (
        "aucun token hugging face",
        "l'appel au modele hugging face a echoue",
        "le client hugging face n'a pas pu etre initialise",
        "le modele llm local n'a pas pu etre charge",
    )
    return any(marker in lowered for marker in markers)


@router.post("", response_model=ChatAskResponse)
@router.post("/ask", response_model=ChatAskResponse)
def chat_ask(request: ChatAskRequest) -> ChatAskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La question ne peut pas etre vide.",
        )

    try:
        result = ask_masi_chatbot(
            question=question,
            conversation_history=request.conversation_history,
            current_dashboard_state=request.current_dashboard_state,
            return_debug=request.debug,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne est survenue lors du traitement de la requete chatbot.",
        ) from exc

    answer = str(result.get("answer", "")).strip()
    if _looks_like_service_unavailable(answer):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=answer,
        )

    return ChatAskResponse(**result)


@router.post("/ask/stream")
def chat_ask_stream(request: ChatAskRequest) -> StreamingResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La question ne peut pas etre vide.",
        )

    def event_stream():
        try:
            for event in stream_masi_chatbot(
                question=question,
                conversation_history=request.conversation_history,
                current_dashboard_state=request.current_dashboard_state,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except ValueError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n"
        except RuntimeError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n"
        except Exception:
            yield json.dumps(
                {
                    "type": "error",
                    "detail": "Une erreur interne est survenue lors du traitement de la requete chatbot.",
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )
