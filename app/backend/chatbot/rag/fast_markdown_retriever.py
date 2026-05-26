"""Fast Markdown retriever for demo-friendly chatbot RAG.

This keeps the RAG layer local and deterministic without loading the heavier
embedding model used by the Chroma retriever.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
RAG_DIR = CURRENT_FILE.parent
DOCS_DIR = RAG_DIR / "docs"

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9_]+")
_STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "il",
    "la",
    "le",
    "les",
    "pour",
    "que",
    "qui",
    "sur",
    "un",
    "une",
}

MAX_CHUNK_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 40

_QUERY_EXPANSIONS = {
    "masi": "moroccan all shares index bourse casablanca indice boursier marche actions marocain glossaire definition",
    "quoi": "definition glossaire signifie expliquer concept",
    "definis": "definition glossaire signifie expliquer concept",
    "define": "definition glossaire signifie expliquer concept",
    "baisse": "rendement prevu negatif direction masi regime hmm volatilite tendance",
    "hausse": "rendement prevu positif direction masi regime hmm volatilite tendance",
    "direction": "rendement prevu tendance masi regime hmm volatilite",
    "tendance": "rendement prevu direction masi regime hmm volatilite",
    "regime": "hmm volatilite latent risque pas direction",
    "rigime": "hmm volatilite latent risque pas direction",
    "hmm": "regime volatilite latent risque pas direction",
    "monte": "monte carlo horizons 10 jours 25 jours extension operationnelle",
    "carlo": "monte carlo horizons 10 jours 25 jours extension operationnelle",
    "horizon": "horizons 10 jours 25 jours extension operationnelle scaling racine",
    "poids": "risk managed var budget quantile fenetre allocation simulee",
    "quantile": "risk managed var budget poids fenetre allocation simulee",
    "wealth": "risk managed backtest economique richesse sharpe drawdown",
    "dashboard": "lecture chiffres rendement volatilite var es backtesting",
    "salut": "contrat fonctionnement salutation conversation courte pas dashboard",
    "bonjour": "contrat fonctionnement salutation conversation courte pas dashboard",
    "merci": "contrat fonctionnement conversation courte pas dashboard",
    "ok": "contrat fonctionnement conversation courte pas dashboard",
}


@dataclass(frozen=True)
class MarkdownChunk:
    source: str
    section: str
    doc_type: str
    content: str


@dataclass(frozen=True)
class ScoredMarkdownChunk:
    score: float
    chunk: MarkdownChunk


def _word_tokens_with_spacing(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _split_content_by_token_window(
    content: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    words = _word_tokens_with_spacing(content)
    if len(words) <= max_tokens:
        return [content.strip()] if content.strip() else []

    chunks: list[str] = []
    step = max(1, max_tokens - overlap_tokens)
    for start in range(0, len(words), step):
        window = words[start : start + max_tokens]
        if not window:
            break
        chunks.append(" ".join(window).strip())
        if start + max_tokens >= len(words):
            break
    return chunks


def _tokens(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    }


def _expand_query(query: str) -> str:
    query_tokens = _tokens(query)
    expansions = [
        expansion
        for token, expansion in _QUERY_EXPANSIONS.items()
        if token in query_tokens
    ]
    return " ".join([query, *expansions])


def _split_markdown(source: Path, text: str) -> list[MarkdownChunk]:
    chunks: list[MarkdownChunk] = []
    current_section = source.stem
    buffer: list[str] = []
    doc_type = source.stem.replace("_", " ")

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            for idx, child_content in enumerate(_split_content_by_token_window(content)):
                section = current_section if idx == 0 else f"{current_section} / chunk {idx + 1}"
                chunks.append(
                    MarkdownChunk(
                        source=source.name,
                        section=section,
                        doc_type=doc_type,
                        content=child_content,
                    )
                )
        buffer.clear()

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            current_section = line.lstrip("#").strip() or source.stem
            continue
        if line.strip() == "---":
            flush()
            continue
        buffer.append(line)

    flush()
    return chunks


@lru_cache(maxsize=1)
def load_markdown_chunks() -> tuple[MarkdownChunk, ...]:
    if not DOCS_DIR.exists():
        return ()

    chunks: list[MarkdownChunk] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        chunks.extend(_split_markdown(path, text))

    return tuple(chunks)


def retrieve_fast_markdown_chunks(
    query: str,
    k: int = 5,
    min_score: float = 1.5,
) -> list[ScoredMarkdownChunk]:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query cannot be empty.")

    chunks = load_markdown_chunks()
    if not chunks:
        return []

    expanded_query = _expand_query(cleaned_query)
    query_tokens = _tokens(expanded_query)
    scored: list[tuple[float, MarkdownChunk]] = []
    for chunk in chunks:
        chunk_tokens = _tokens(f"{chunk.section}\n{chunk.content}")
        overlap = query_tokens.intersection(chunk_tokens)
        score = float(len(overlap))
        source_section = f"{chunk.source} {chunk.section}".lower()
        if chunk.source == "chatbot_operating_contract.md":
            score += 0.5
        if chunk.source == "canonical_interpretation_rules.md":
            score += 1.5
        if chunk.source == "glossary_risk_terms.md" and any(
            token in query_tokens for token in {
                "masi", "quoi", "definis", "define", "signification", 
                "difference", "différence", "explique", "explication",
                "var", "shortfall", "es", "garch", "hmm"
            }
        ):
            score += 6.0
        if any(token in query_tokens for token in {"salut", "bonjour", "merci", "ok"}):
            if {"salutation", "conversation", "courte", "dashboard"}.intersection(chunk_tokens):
                score += 5.0
        if any(token in query_tokens for token in {"baisse", "hausse", "direction", "tendance"}):
            if {"regime", "hmm", "volatilite", "direction"}.intersection(chunk_tokens):
                score += 4.0
        if any(token in query_tokens for token in {"poids", "quantile", "wealth"}):
            if {"risk", "managed", "budget", "var", "poids"}.intersection(chunk_tokens):
                score += 3.0
        if any(token in query_tokens for token in {"horizon", "monte", "carlo"}):
            if {"horizons", "jours", "scaling", "extension"}.intersection(chunk_tokens):
                score += 3.0
        if "regles canoniques" in source_section:
            score += 1.0
        if "var" in query_tokens and "var" in chunk_tokens:
            score += 2.0
            if "value-at-risk" in chunk.section.lower() or "var" in chunk.section.lower():
                score += 10.0
        if {"es", "expected", "shortfall"}.intersection(query_tokens) and {
            "es",
            "expected",
            "shortfall",
        }.intersection(chunk_tokens):
            score += 2.0
            if "shortfall" in chunk.section.lower() or "es" in chunk.section.lower():
                score += 10.0
        if score > 0:
            scored.append((score, chunk))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[ScoredMarkdownChunk] = []
    seen_content_prefixes: set[str] = set()
    for score, chunk in scored:
        if score < min_score:
            continue
        prefix = " ".join(chunk.content.split())[:240]
        if prefix in seen_content_prefixes:
            continue
        seen_content_prefixes.add(prefix)
        selected.append(ScoredMarkdownChunk(score=score, chunk=chunk))
        if len(selected) >= max(1, k):
            break
    return selected


def format_markdown_chunks(chunks: list[ScoredMarkdownChunk]) -> str:
    blocks = []
    for rank, item in enumerate(chunks, start=1):
        chunk = item.chunk
        content = chunk.content.strip()
        words = _word_tokens_with_spacing(content)
        if len(words) > MAX_CHUNK_TOKENS:
            content = " ".join(words[:MAX_CHUNK_TOKENS]).strip() + "..."
        blocks.append(
            f"[Passage {rank} | Source: {chunk.source} | Section: {chunk.section} | Type: {chunk.doc_type} | Score: {item.score:.2f}]\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(blocks)


def retrieve_fast_markdown_context(query: str, k: int = 5, min_score: float = 1.5) -> str:
    chunks = retrieve_fast_markdown_chunks(query=query, k=k, min_score=min_score)
    if not chunks:
        return ""
    return format_markdown_chunks(chunks)
