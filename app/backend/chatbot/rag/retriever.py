"""
Retriever for the MASI Risk Dashboard RAG layer.

Role:
- Load the local ChromaDB vector store.
- Embed the user query with the same embedding model used at indexing time.
- Retrieve the most relevant Markdown chunks.
- Return a clean text context that can be injected into the chatbot prompt.

Important:
- This file does NOT call the LLM.
- This file does NOT read dynamic dashboard outputs.
- This file only retrieves fixed RAG knowledge from backend/chatbot/rag/vector_db/.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
else:
    Document = Any
    Chroma = Any
    HuggingFaceEmbeddings = Any


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
RAG_DIR = CURRENT_FILE.parent
VECTOR_DB_DIR = RAG_DIR / "vector_db"

EMBEDDING_MODEL_NAME = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
COLLECTION_NAME = "masi_risk_rag"
MAX_RETURNED_CHUNK_TOKENS = 200

_EMBEDDINGS = None
_EMBEDDINGS_KEY: tuple[str, str] | None = None
_VECTOR_STORE = None
_VECTOR_STORE_KEY: tuple[str, str, str] | None = None


@dataclass(frozen=True)
class RetrieverConfig:
    vector_db_dir: Path = VECTOR_DB_DIR
    embedding_model_name: str = EMBEDDING_MODEL_NAME
    collection_name: str = COLLECTION_NAME
    default_k: int = 5
    max_k: int = 12
    use_mmr: bool = False
    fetch_k: int = 20
    lambda_mult: float = 0.65


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def get_device() -> str:
    """
    Default to CPU for retrieval embeddings during chatbot inference.

    Reason:
    The generative LLM uses the GPU. Keeping the embedding model on CPU
    avoids CUDA out-of-memory errors on 8GB VRAM GPUs.

    To test retrieval alone on GPU, set:
    RAG_EMBEDDING_DEVICE=cuda
    """

    return os.getenv("RAG_EMBEDDING_DEVICE", "cpu")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def validate_query(query: str) -> str:
    """
    Validate and clean the user query.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string.")

    cleaned = query.strip()

    if not cleaned:
        raise ValueError("query cannot be empty.")

    return cleaned


def validate_k(k: int, max_k: int) -> int:
    """
    Validate the number of retrieved chunks.
    """

    if not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be positive.")

    return min(k, max_k)


def ensure_vector_db_exists(vector_db_dir: Path) -> None:
    """
    Ensure the vector database exists before retrieval.
    """

    if not vector_db_dir.exists():
        raise FileNotFoundError(
            f"Vector database folder not found: {vector_db_dir}. "
            "Run `python -m backend.chatbot.rag.build_index` first."
        )

    if not vector_db_dir.is_dir():
        raise NotADirectoryError(
            f"Vector database path is not a directory: {vector_db_dir}"
        )

    has_files = any(vector_db_dir.iterdir())
    if not has_files:
        raise FileNotFoundError(
            f"Vector database folder is empty: {vector_db_dir}. "
            "Run `python -m backend.chatbot.rag.build_index` first."
        )


# ---------------------------------------------------------------------
# Embeddings and vector store
# ---------------------------------------------------------------------

def load_embeddings(config: RetrieverConfig = RetrieverConfig()) -> HuggingFaceEmbeddings:
    """
    Load the same embedding model used during indexing.
    """

    global _EMBEDDINGS, _EMBEDDINGS_KEY

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception as exc:
        raise RuntimeError(
            "Le backend RAG 'chroma' necessite langchain-huggingface. "
            "Installe les dependances runtime avec `pip install -r requirements.txt` "
            "ou utilise RAG_RETRIEVER_BACKEND=markdown pour depanner."
        ) from exc

    try:
        from huggingface_hub import get_token
    except Exception:
        get_token = None

    device = get_device()
    key = (config.embedding_model_name, device)

    if _EMBEDDINGS is not None and _EMBEDDINGS_KEY == key:
        return _EMBEDDINGS

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or (get_token() if get_token is not None else None)
        or None
    )

    _EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=config.embedding_model_name,
        model_kwargs={
            "device": device,
            "token": hf_token,
            "local_files_only": _env_bool("RAG_LOCAL_FILES_ONLY", True),
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )
    _EMBEDDINGS_KEY = key
    return _EMBEDDINGS


def load_vector_store(config: RetrieverConfig = RetrieverConfig()) -> Chroma:
    """
    Load the local ChromaDB vector store.
    """

    global _VECTOR_STORE, _VECTOR_STORE_KEY

    try:
        from langchain_chroma import Chroma
    except Exception as exc:
        raise RuntimeError(
            "Le backend RAG 'chroma' necessite langchain-chroma. "
            "Installe les dependances runtime avec `pip install -r requirements.txt` "
            "ou utilise RAG_RETRIEVER_BACKEND=markdown pour depanner."
        ) from exc

    ensure_vector_db_exists(config.vector_db_dir)

    cache_key = (
        str(config.vector_db_dir),
        config.collection_name,
        config.embedding_model_name,
    )

    if _VECTOR_STORE is not None and _VECTOR_STORE_KEY == cache_key:
        return _VECTOR_STORE

    embeddings = load_embeddings(config)

    vector_store = Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.vector_db_dir),
    )

    count = vector_store._collection.count()

    if count == 0:
        raise RuntimeError(
            "The ChromaDB collection is empty. "
            "Run `python -m backend.chatbot.rag.build_index` again."
        )

    _VECTOR_STORE = vector_store
    _VECTOR_STORE_KEY = cache_key
    return _VECTOR_STORE


def warm_retriever_cache(config: RetrieverConfig = RetrieverConfig()) -> None:
    """
    Preload embeddings and vector store into the process cache.

    This can be called in a background thread at API startup so that the
    first chatbot request does not pay the full cold-start cost.
    """

    load_vector_store(config)
    try:
        from backend.chatbot.intent_router import warm_intent_router

        warm_intent_router()
    except Exception:
        # RAG warmup remains useful even if intent centroid warmup fails.
        pass


# ---------------------------------------------------------------------
# Metadata formatting
# ---------------------------------------------------------------------

def _safe_meta(doc: Document, key: str) -> Optional[str]:
    value = doc.metadata.get(key)
    if value is None:
        return None
    return str(value).strip()


def format_document_header(doc: Document, rank: int) -> str:
    """
    Format source metadata for a retrieved chunk.
    """

    source = _safe_meta(doc, "source") or "unknown_source"
    header_1 = _safe_meta(doc, "header_1")
    header_2 = _safe_meta(doc, "header_2")
    header_3 = _safe_meta(doc, "header_3")

    sections = [h for h in [header_1, header_2, header_3] if h]

    if sections:
        section_text = " > ".join(sections)
    else:
        section_text = "Section non renseignée"

    return f"[Passage {rank} | Source: {source} | Section: {section_text}]"


def format_retrieved_documents(documents: List[Document]) -> str:
    """
    Convert retrieved documents into a clean text block for the chatbot prompt.
    """

    if not documents:
        return (
            "Aucun passage pertinent n'a été retrouvé dans la base RAG fixe."
        )

    blocks: List[str] = []

    for rank, doc in enumerate(documents, start=1):
        header = format_document_header(doc, rank)
        content = doc.page_content.strip()
        words = content.split()
        if len(words) > MAX_RETURNED_CHUNK_TOKENS:
            content = " ".join(words[:MAX_RETURNED_CHUNK_TOKENS]).strip() + "..."

        blocks.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------

def retrieve_documents(
    query: str,
    k: int = 5,
    config: RetrieverConfig = RetrieverConfig(),
) -> List[Document]:
    """
    Retrieve relevant documents from the vector database.

    Uses MMR by default to reduce redundancy between retrieved chunks.
    """

    cleaned_query = validate_query(query)
    safe_k = validate_k(k, config.max_k)

    vector_store = load_vector_store(config)

    if config.use_mmr:
        docs = vector_store.max_marginal_relevance_search(
            cleaned_query,
            k=safe_k,
            fetch_k=max(config.fetch_k, safe_k),
            lambda_mult=config.lambda_mult,
        )
    else:
        docs = vector_store.similarity_search(
            cleaned_query,
            k=safe_k,
        )

    return docs


def retrieve_relevant_context(
    query: str,
    k: int = 5,
    config: RetrieverConfig = RetrieverConfig(),
) -> str:
    """
    Retrieve relevant fixed RAG context as a formatted text block.

    This is the main function that will later be used by chat_service.py.
    """

    docs = retrieve_documents(query=query, k=k, config=config)
    return format_retrieved_documents(docs)


def inspect_retrieval(
    query: str,
    k: int = 5,
    config: RetrieverConfig = RetrieverConfig(),
) -> None:
    """
    Debug helper: print retrieved chunks with metadata.
    """

    docs = retrieve_documents(query=query, k=k, config=config)

    print(f"Query: {query}")
    print(f"Embedding device: {get_device()}")
    print(f"Vector DB: {config.vector_db_dir}")
    print(f"Retrieved chunks: {len(docs)}")

    for i, doc in enumerate(docs, start=1):
        print("\n" + "=" * 100)
        print(format_document_header(doc, i))
        print("-" * 100)
        print(doc.page_content[:1500])


# ---------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Intent-aware retrieval (new architecture)
# ---------------------------------------------------------------------

# Mapping: intent → extra keyword hints to prepend to the query
_INTENT_QUERY_HINTS: dict[str, str] = {
    "definition_query": "définition glossaire signification expliquer concept",
    "backtest_query": "backtesting Kupiec Christoffersen violation p-value calibration couverture",
    "strategy_query": "risk-targeting HMM régime exposition allocation budget de risque",
    "model_query": "EGARCH GJR-GARCH LSTM-Ridge modèle comparaison architecture",
    "data_query": "données source période fréquence preprocessing séquence",
    "forecast_query": "prévision rendement VaR ES Expected Shortfall horizon",
}

# Intents that should NOT call RAG at all
_NO_RAG_INTENTS = {"help_request", "out_of_scope"}


def retrieve_relevant_context_for_intent(
    question: str,
    intent: str,
    k: int = 5,
    config: RetrieverConfig = RetrieverConfig(),
) -> str:
    """
    Retrieve RAG context filtered by intent.

    Rules
    -----
    - help_request / out_of_scope  → returns empty string (no RAG needed).
    - Other intents               → prepends intent-specific keywords to
                                    the query to bias retrieval toward the
                                    right document types.

    Parameters
    ----------
    question : str
        The user question.
    intent : str
        The classified intent from intent_router.
    k : int
        Maximum number of chunks to return.
    config : RetrieverConfig
        Retriever configuration.

    Returns
    -------
    str
        Formatted RAG context, or empty string if not applicable.
    """
    if intent in _NO_RAG_INTENTS:
        return ""

    hint = _INTENT_QUERY_HINTS.get(intent, "")
    enriched_query = f"{hint} {question}".strip() if hint else question

    try:
        docs = retrieve_documents(query=enriched_query, k=min(k, 5), config=config)
    except Exception:
        return ""

    if not docs:
        return ""

    return format_retrieved_documents(docs)


if __name__ == "__main__":
    test_queries = [
        "Quelle est la différence entre VaR et Expected Shortfall ?",
        "Pourquoi le dashboard affiche les prix alors que le modèle prédit les rendements ?",
        "Est-ce que la VaR est une perte maximale garantie ?",
        "Que signifie le régime high volatility ?",
        "Est-ce que le poids simulé est une recommandation d'investissement ?",
        "Comment interpréter le test de Kupiec ?",
        "Pourquoi le Sharpe peut être meilleur alors que la richesse finale est plus faible ?",
    ]

    for q in test_queries:
        print("\n" + "#" * 120)
        inspect_retrieval(q, k=5)
