from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

from rag_core import (
    SUPPORTED_EXTENSIONS,
    answer_question,
    chunk_text,
    csv_rows_to_chunks,
    embed_texts,
    retrieve,
)

load_dotenv()

st.set_page_config(
    page_title="DocuMind · RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background: #f7f8fc; }
        [data-testid="stSidebar"] { background: #111827; }
        [data-testid="stSidebar"] * { color: #f3f4f6; }
        .hero {
            padding: 1.25rem 1.5rem;
            border-radius: 18px;
            color: white;
            background: linear-gradient(115deg, #172554 0%, #4338ca 58%, #7c3aed 100%);
            margin-bottom: 1.2rem;
        }
        .hero h1 { margin: 0; font-size: 2rem; color: white; }
        .hero p { margin: .35rem 0 0; color: #e0e7ff; }
        .metric-card {
            background: white;
            padding: 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
        }
        .source-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #6366f1;
            border-radius: 10px;
            padding: .8rem 1rem;
            margin: .5rem 0;
        }
        .small-muted { color: #6b7280; font-size: .86rem; }
        div[data-testid="stChatMessage"] { border-radius: 14px; }

        /* Keep chat and source text readable. */
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] strong,
        div[data-testid="stChatMessage"] h1,
        div[data-testid="stChatMessage"] h2,
        div[data-testid="stChatMessage"] h3,
        .source-card,
        .source-card *,
        .small-muted {
            color: #1f2937 !important;
        }

        /* Keep code text readable. */
        .stMarkdown pre,
        .stMarkdown pre code,
        .stMarkdown pre code span,
        [data-testid="stCode"] pre,
        [data-testid="stCode"] code,
        [data-testid="stCode"] span {
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
        }
        .stMarkdown pre,
        [data-testid="stCode"] pre {
            background-color: #1e293b !important;
        }

        /* Keep banner text white. */
        .hero,
        .hero h1,
        .hero p {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📚 DocuMind</h1>
        <p>Ask questions across your files. Get grounded answers with precise source citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def get_client() -> Groq:
    key = st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY", "")
    if not key:
        raise ValueError(
            "Add your Groq API key in the sidebar or set GROQ_API_KEY in .env."
        )
    return Groq(api_key=key)


def get_collection():
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.PersistentClient(path="./.chroma")
    if "collection" not in st.session_state:
        st.session_state.collection = (
            st.session_state.chroma_client.get_or_create_collection(
                name="documind_chunks",
                metadata={"hnsw:space": "cosine"},
            )
        )
    return st.session_state.collection


def extract_file(uploaded) -> tuple[str, list[dict[str, Any]]]:
    name = uploaded.name
    suffix = Path(name).suffix.lower()
    raw = uploaded.getvalue()
    file_hash = hashlib.sha256(raw).hexdigest()[:16]

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            content = page.extract_text() or ""
            if content.strip():
                pages.append({"text": content, "page": i})
        return file_hash, pages

    if suffix == ".txt":
        return file_hash, [
            {"text": raw.decode("utf-8", errors="replace"), "page": None}
        ]

    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
        return file_hash, [{"dataframe": df}]

    raise ValueError(f"Unsupported file type: {suffix}")


def index_uploads(uploaded_files, client, collection, embedding_model):
    status = st.status("Preparing your documents…", expanded=True)
    added = 0

    for uploaded in uploaded_files:
        file_name = uploaded.name
        try:
            file_hash, extracted = extract_file(uploaded)

            # Stable document ID makes repeat uploads idempotent.
            doc_id = f"{file_hash}"
            if collection.get(where={"doc_id": doc_id}, limit=1).get("ids"):
                status.write(f"Already indexed: **{file_name}**")
                continue

            chunks, metadata = [], []
            suffix = Path(file_name).suffix.lower()

            if suffix == ".csv":
                frame = extracted[0]["dataframe"]
                pairs = csv_rows_to_chunks(frame, file_name)
                for chunk_text_value, row_num in pairs:
                    chunks.append(chunk_text_value)
                    metadata.append(
                        {
                            "source": file_name,
                            "doc_id": doc_id,
                            "chunk_index": len(chunks) - 1,
                            "file_type": "csv",
                            "page": -1,
                            "row": row_num,
                        }
                    )
            else:
                for page_info in extracted:
                    page_chunks = chunk_text(page_info["text"])
                    for piece in page_chunks:
                        chunks.append(piece)
                        metadata.append(
                            {
                                "source": file_name,
                                "doc_id": doc_id,
                                "chunk_index": len(chunks) - 1,
                                "file_type": suffix[1:],
                                "page": page_info["page"] or -1,
                                "row": -1,
                            }
                        )

            if not chunks:
                status.write(f"No extractable text found: **{file_name}**")
                continue

            status.write(
                f"Embedding **{len(chunks)} chunks** from **{file_name}**…"
            )
            vectors = embed_texts(chunks)
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=vectors,
                metadatas=metadata,
            )
            added += len(chunks)
            status.write(f"Indexed: **{file_name}**")

        except Exception as exc:
            status.write(f"⚠️ Could not index **{file_name}**: {exc}")

    status.update(
        label=f"Indexing complete · {added} new chunks",
        state="complete",
        expanded=False,
    )
    st.session_state.last_indexed_count = added


with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    answer_mode = st.radio(
        "Answer mode",
        ["Document-only", "Document + AI knowledge"],
        key="answer_mode",
        help=(
            "Choose whether answers must use your documents or can also use "
            "general AI knowledge."
        ),
    )
    st.caption("The API key is used locally by this app to call Groq.")
    st.text_input(
        "Groq API key",
        type="password",
        key="groq_api_key",
        placeholder="gsk_…",
        help="Or add GROQ_API_KEY to your .env file.",
    )
    st.selectbox(
        "Groq chat model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
        ],
        key="chat_model",
    )
    st.slider("Chunks to retrieve", min_value=2, max_value=10, value=5, key="top_k")
    st.slider(
        "Minimum relevance (cosine similarity)",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        key="threshold",
    )
    st.divider()
    st.markdown("## 📂 Your knowledge base")
    uploaded_files = st.file_uploader(
        "Add PDF, TXT, or CSV files",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="PDF/TXT use token-aware chunks. CSVs are indexed one row per chunk.",
    )

    if st.button(
        "Build / update index",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        try:
            client = get_client()
            index_uploads(
                uploaded_files,
                client,
                get_collection(),
                st.session_state.get("embedding_model", "text-embedding-3-small"),
            )
        except Exception as exc:
            st.error(str(exc))

    if st.button("Clear entire knowledge base", use_container_width=True):
        try:
            db = st.session_state.get("chroma_client") or chromadb.PersistentClient(
                path="./.chroma"
            )
            db.delete_collection("documind_chunks")
            st.session_state.pop("collection", None)
            st.session_state.messages = []
            st.success(
                "Knowledge base cleared. Reload the page if the sidebar count "
                "has not refreshed."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not clear index: {exc}")

    st.caption("Local vector store: `./.chroma`")


collection = get_collection()
count = collection.count()
all_meta = collection.get(include=["metadatas"]).get("metadatas", []) if count else []
sources = sorted({m.get("source", "Unknown") for m in all_meta})

c1, c2, c3 = st.columns(3)
c1.metric("Indexed chunks", count)
c2.metric("Documents", len(sources))
c3.metric("Retrieval threshold", f"{st.session_state.get('threshold', 0.30):.2f}")

with st.expander("How this project works", expanded=False):
    st.markdown(
        """
        1. **Ingest:** extract PDF/TXT text or parse CSV rows.
        2. **Chunk:** token-aware text windows; CSV rows remain atomic.
        3. **Embed:** Local Sentence Transformers embeddings are stored in persistent ChromaDB.
        4. **Retrieve:** cosine-similarity search with a configurable relevance threshold.
        5. **Generate:** the model answers from retrieved context and cites source chunks.
        """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources used"):
                for source in message["sources"]:
                    st.markdown(
                        f"""<div class="source-card"><b>{source['label']}</b>
                        <div class="small-muted">Relevance: {source['score']:.3f}</div>
                        <div>{source['text'][:900]}{'…' if len(source['text']) > 900 else ''}</div></div>""",
                        unsafe_allow_html=True,
                    )

if count == 0:
    st.info(
        "Upload documents in the sidebar, then click **Build / update index** to begin."
    )
elif not st.session_state.messages:
    st.markdown("### Try asking")
    prompts = [
        "Summarize the main ideas in my documents.",
        "What are the key facts and where are they mentioned?",
        "Compare the information across my uploaded files.",
    ]
    cols = st.columns(3)
    for col, prompt in zip(cols, prompts):
        with col:
            if st.button(prompt, use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

question = st.chat_input("Ask a question about your documents…")
if not question and st.session_state.get("pending_prompt"):
    question = st.session_state.pop("pending_prompt")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            client = get_client()
            results = retrieve(
                collection,
                question,
                top_k=st.session_state.get("top_k", 5),
                threshold=st.session_state.get("threshold", 0.30),
            )
            answer_mode = st.session_state.get("answer_mode", "Document-only")

            if not results and answer_mode == "Document-only":
                answer = (
                    "I couldn't find sufficiently relevant information in the indexed "
                    "documents to answer that reliably. Try rephrasing the question "
                    "or adding a relevant file."
                )
                source_cards = []
            else:
                answer = answer_question(
                    client,
                    question,
                    results,
                    model=st.session_state.get("chat_model", "openai/gpt-oss-20b"),
                    answer_mode=answer_mode,
                )
                source_cards = [
                    {
                        "label": (
                            f"{result['metadata'].get('source', 'Unknown')} · "
                            f"{'Page ' + str(result['metadata']['page']) if result['metadata'].get('page', -1) > 0 else ('Row ' + str(result['metadata']['row']) if result['metadata'].get('row', -1) > 0 else 'Chunk ' + str(result['metadata'].get('chunk_index', 0) + 1))}"
                        ),
                        "score": result["score"],
                        "text": result["text"],
                    }
                    for result in results
                ]

            st.markdown(answer)
            if source_cards:
                with st.expander("📎 Sources used", expanded=True):
                    for source in source_cards:
                        st.markdown(
                            f"""<div class="source-card"><b>{source['label']}</b>
                            <div class="small-muted">Relevance: {source['score']:.3f}</div>
                            <div>{source['text'][:900]}{'…' if len(source['text']) > 900 else ''}</div></div>""",
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": source_cards,
                }
            )

        except Exception as exc:
            error_message = f"**Request failed:** {exc}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
