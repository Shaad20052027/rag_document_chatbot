"""RAG primitives: type-aware chunking, local embeddings, retrieval, grounded generation."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_CHUNK_TOKENS = 450
DEFAULT_OVERLAP_TOKENS = 75

_LOCAL_EMBEDDER = None

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SUPPORTED_EXTENSIONS = ["pdf", "txt", "csv"]


def _tokenize(text: str) -> list[str]:
    """Split text into word and punctuation tokens."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def chunk_text(
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping chunks."""
    tokens = _tokenize(text or "")

    if not tokens:
        return []

    if (
        chunk_tokens <= 0
        or overlap_tokens < 0
        or overlap_tokens >= chunk_tokens
    ):
        raise ValueError(
            "chunk_tokens must be positive and overlap must be smaller than chunk size."
        )

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunks.append(" ".join(tokens[start:end]))

        if end == len(tokens):
            break

        start = end - overlap_tokens

    return chunks


def csv_rows_to_chunks(
    df: pd.DataFrame,
    source: str = "data.csv",
) -> list[tuple[str, int]]:
    """Convert each CSV row into a text chunk."""
    pairs = []
    headers = [str(column) for column in df.columns]

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        rendered = (
            f"Source: {source} | CSV row {row_number}\n"
            + "\n".join(
                f"{header}: {row[header]}" for header in headers
            )
        )

        pairs.append((rendered, row_number))

    return pairs


def embed_texts(
    texts: list[str],
    batch_size: int = 64,
) -> list[list[float]]:
    """Create embeddings locally using Sentence Transformers."""
    global _LOCAL_EMBEDDER

    if _LOCAL_EMBEDDER is None:
        _LOCAL_EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)

    vectors = _LOCAL_EMBEDDER.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return vectors.tolist()


def retrieve(
    collection,
    query: str,
    top_k: int = 5,
    threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """Retrieve relevant document chunks from the vector database."""
    if collection.count() == 0:
        return []

    query_vector = embed_texts([query])[0]

    found = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    results = []

    for text, metadata, distance in zip(
        found["documents"][0],
        found["metadatas"][0],
        found["distances"][0],
    ):
        score = max(-1.0, min(1.0, 1.0 - float(distance)))

        if score >= threshold:
            results.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "score": score,
                }
            )

    return results


def answer_question(
    client,
    question: str,
    retrieved: list[dict[str, Any]],
    model: str = "openai/gpt-oss-20b",
    answer_mode: str = "Document-only",
) -> str:
    """
    Generate an answer using retrieved documents or general AI knowledge.

    Document-only:
        Answer strictly from retrieved evidence.

    Document + AI knowledge:
        Use retrieved evidence when relevant and general knowledge
        when the documents do not contain enough information.
    """

    # Build context from retrieved document chunks.
    context_parts = []

    for index, item in enumerate(retrieved, start=1):
        metadata = item.get("metadata", {})

        if metadata.get("file_type") == "csv":
            locator = f"row {metadata.get('row', '?')}"
        else:
            page = metadata.get("page", -1)

            if page > 0:
                locator = f"page {page}"
            else:
                locator = f"chunk {metadata.get('chunk_index', 0) + 1}"

        source_name = metadata.get("source", "Unknown file")
        label = f"[{source_name} · {locator}]"

        context_parts.append(
            f"Evidence {index} {label}:\n{item.get('text', '')}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Set the system prompt according to the selected answer mode.
    if answer_mode == "Document-only":
        system = """
You are DocuMind, a careful document question-answering assistant.

Your task is to answer the user's question using ONLY the evidence
supplied in the user message.

Rules:
1. Do not use outside knowledge to fill gaps.
2. If the evidence does not support the answer, explicitly say:
   "I couldn't determine this from the indexed documents."
3. Cite claims using the exact source label provided in the evidence.
4. Use citation formats such as [filename.pdf · page 2] or [data.csv · row 4].
5. Never fabricate citations or source information.
6. Be concise and explain uncertainty when appropriate.
"""
    else:
        system = """
You are DocuMind, a helpful assistant that can use both retrieved
documents and general AI knowledge.

Your task is to answer the user's question accurately.

Rules:
1. Use the supplied document evidence whenever it is relevant.
2. If the documents do not contain enough information, you may use
   general AI knowledge to provide a helpful answer.
3. Clearly distinguish document-supported information from general
   knowledge when the distinction matters.
4. Cite document-supported claims using the exact source label provided
   in the evidence.
5. Use citation formats such as [filename.pdf · page 2] or [data.csv · row 4].
6. Do not fabricate citations or source information.
7. If you use general knowledge, do not attach document citations to it.
8. Be concise and explain uncertainty when appropriate.
"""

    # Make the evidence state clear to the model.
    if context:
        user_message = (
            f"Question: {question}\n\n"
            f"Retrieved evidence:\n{context}"
        )
    else:
        user_message = (
            f"Question: {question}\n\n"
            "No relevant evidence was found in the indexed documents."
        )

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )

    return (
        response.choices[0].message.content
        or "I could not generate an answer."
    )