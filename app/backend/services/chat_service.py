"""Public chatbot service facade.

Keep this module as the stable import path for API routes and external callers.
The implementation lives in `backend.chatbot.service`.
"""

from backend.chatbot.service import (
    ChatServiceConfig,
    HF_MODEL_ID,
    ask_masi_chatbot,
    build_system_instructions,
    classify_chat_intent,
    classify_user_intent,
    format_dashboard_state_context,
    generate_huggingface_answer,
    generate_llm_answer,
    generate_llm_answer_stream,
    load_huggingface_client,
    render_conversation_memory_block,
    repair_known_masi_errors,
    sanitize_conversation_history,
    stream_masi_chatbot,
    validate_chatbot_answer,
    validate_user_question,
)

_repair_known_masi_errors = repair_known_masi_errors

__all__ = [
    "ChatServiceConfig",
    "HF_MODEL_ID",
    "ask_masi_chatbot",
    "build_system_instructions",
    "classify_chat_intent",
    "classify_user_intent",
    "format_dashboard_state_context",
    "generate_huggingface_answer",
    "generate_llm_answer",
    "generate_llm_answer_stream",
    "load_huggingface_client",
    "render_conversation_memory_block",
    "repair_known_masi_errors",
    "_repair_known_masi_errors",
    "sanitize_conversation_history",
    "stream_masi_chatbot",
    "validate_chatbot_answer",
    "validate_user_question",
]
