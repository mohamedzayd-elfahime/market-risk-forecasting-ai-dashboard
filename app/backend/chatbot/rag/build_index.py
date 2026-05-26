from __future__ import annotations

import shutil
import os
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CURRENT_FILE = Path(__file__).resolve()
RAG_DIR = CURRENT_FILE.parent
DOCS_DIR = RAG_DIR / "docs"
VECTOR_DB_DIR = RAG_DIR / "vector_db"

EMBEDDING_MODEL_NAME = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
COLLECTION_NAME = "masi_risk_rag"
MAX_CHUNK_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 40


def clean_markdown_text(text: str) -> str:
    """
    Nettoyage léger adapté au Markdown technique.
    On garde les titres, les accents, les formules LaTeX et le vocabulaire financier.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


def split_text_by_token_window(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into precise child chunks capped at about 200 whitespace tokens."""

    words = text.split()
    if len(words) <= max_tokens:
        return [text] if text.strip() else []

    chunks: list[str] = []
    step = max(1, max_tokens - overlap_tokens)
    for start in range(0, len(words), step):
        window = words[start : start + max_tokens]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + max_tokens >= len(words):
            break
    return chunks


def list_markdown_files(docs_dir: Path = DOCS_DIR) -> List[Path]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    markdown_files = sorted(docs_dir.glob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(f"No Markdown files found in: {docs_dir}")

    return markdown_files


def read_markdown_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")

    text = clean_markdown_text(text)

    if not text:
        raise ValueError(f"Markdown file is empty after cleaning: {path}")

    return text


def split_markdown_by_headers(text: str, source_file: str) -> List[Document]:
    """
    Découpage principal par titres Markdown.
    Les headers deviennent des métadonnées utiles pour le retriever.
    """

    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    chunks = splitter.split_text(text)
    documents: List[Document] = []

    for i, chunk in enumerate(chunks):
        content = clean_markdown_text(chunk.page_content)

        if not content:
            continue

        for child_id, child_content in enumerate(split_text_by_token_window(content)):
            metadata = dict(chunk.metadata)
            metadata["source"] = source_file
            metadata["source_type"] = "fixed_rag_markdown_child_chunk"
            metadata["parent_chunk_id"] = i
            metadata["child_chunk_id"] = child_id
            metadata["max_chunk_tokens"] = MAX_CHUNK_TOKENS
            metadata["chunk_overlap_tokens"] = CHUNK_OVERLAP_TOKENS

            documents.append(
                Document(
                    page_content=clean_markdown_text(child_content),
                    metadata=metadata,
                )
            )

    return documents


def refine_large_chunks(
    documents: Iterable[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 180,
) -> List[Document]:
    """
    Certains blocs Markdown peuvent être trop longs.
    On les redécoupe sans perdre les métadonnées.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    refined_documents: List[Document] = []

    for doc in documents:
        if len(doc.page_content) <= chunk_size:
            refined_documents.append(doc)
            continue

        sub_docs = splitter.split_documents([doc])

        for j, sub_doc in enumerate(sub_docs):
            metadata = dict(sub_doc.metadata)
            metadata["subchunk_id"] = j

            refined_documents.append(
                Document(
                    page_content=clean_markdown_text(sub_doc.page_content),
                    metadata=metadata,
                )
            )

    return refined_documents


def load_rag_documents() -> List[Document]:
    all_documents: List[Document] = []

    markdown_files = list_markdown_files(DOCS_DIR)

    for path in markdown_files:
        text = read_markdown_file(path)
        docs = split_markdown_by_headers(text, source_file=path.name)
        all_documents.extend(docs)

    all_documents = refine_large_chunks(all_documents)

    if not all_documents:
        raise RuntimeError("No RAG chunks were created. Check Markdown files.")

    return all_documents


def build_rag_index(rebuild: bool = True) -> Chroma:
    """
    Crée ou recrée la base vectorielle ChromaDB locale.
    """

    documents = load_rag_documents()

    if rebuild and VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": os.getenv("RAG_EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR),
    )

    return vector_store


def main() -> None:
    documents = load_rag_documents()

    print(f"Markdown documents folder: {DOCS_DIR}")
    print(f"Vector DB folder: {VECTOR_DB_DIR}")
    print(f"Number of chunks to index: {len(documents)}")

    build_rag_index(rebuild=True)

    print("RAG index built successfully.")
    print(f"Indexed chunks: {len(documents)}")


if __name__ == "__main__":
    main()
