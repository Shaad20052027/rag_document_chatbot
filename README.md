# DocuMind — RAG Chatbot for Documents

A portfolio-ready Retrieval-Augmented Generation (RAG) application that lets users ask questions about PDF, TXT, and CSV files. Answers are grounded in retrieved document chunks and include source citations. The app declines to answer when the retrieved evidence does not meet a configurable relevance threshold.

## Features

- **PDF, TXT, CSV ingestion** with document-aware extraction.
- **Type-aware chunking:** overlapping token-oriented chunks for prose; one complete row per CSV chunk.
- **Embeddings:** Groq `sentence-transformers/all-MiniLM-L6-v2` or `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector search:** persistent ChromaDB collection using cosine distance.
- **Grounded answers:** Groq chat completion with explicit evidence-only instructions and citations.
- **Relevance threshold:** configurable similarity gate to reduce unsupported answers.
- **Source inspection:** see retrieved source chunks and relevance scores.
- **Evaluation harness:** retrieval Hit Rate@K, Mean Reciprocal Rank (MRR@K), and mean similarity.
- **Local-first UI:** Streamlit app; uploaded documents are indexed into the local `.chroma` folder.

## Architecture

```text
PDF / TXT / CSV
      │
      ▼
Extraction ── PDF page text / TXT body / CSV dataframe
      │
      ▼
Type-aware chunking ── overlapping text windows / atomic CSV rows
      │
      ▼
Local Sentence Transformers embeddings
      │
      ▼
ChromaDB (persistent vector index)
      ▲
      │ cosine similarity + relevance threshold
User question ──► Query embedding ──► Retrieved evidence
                                      │
                                      ▼
                            Groq grounded generation
                                      │
                                      ▼
                         Answer + citations + source panels
```

## Requirements

- Groq provides the chat completion endpoint. This app uses local Sentence Transformers for embeddings, so the embedding step does not need a paid API key. The first run downloads the embedding model.

- Python 3.10 or newer
- An Groq API key with access to the selected embedding and chat models
- Internet access for Groq API calls

## Run locally (VS Code / Antigravity)

### 1. Extract the ZIP

Open the `rag_document_chatbot` folder in your editor.

### 2. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Put your key in `.env`:
```env
GROQ_API_KEY=your_real_key_here
```

Do not commit `.env` or expose the key in screenshots, GitHub, or a public deployment. You can also paste a key in the app sidebar for the current session.

### 5. Start the app

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

### 6. Use the application

1. Upload one or more `.pdf`, `.txt`, or `.csv` files in the sidebar.
2. Click **Build / update index**.
3. Ask questions in the chat field.
4. Expand **Sources used** to inspect the exact evidence and similarity score.
5. Adjust the minimum relevance slider to tune how conservative retrieval should be.

The index persists in `.chroma` across app restarts. The clear button removes the full local knowledge base.

## Evaluation

The included `evaluate.py` measures retrieval against a labeled question dataset.

1. Index your evaluation documents using the app.
2. Edit `eval_dataset.json`, replacing the example with your questions and expected file names:
```json
[
  {
    "question": "What is the cancellation period?",
    "expected_sources": ["policy.pdf"]
  }
]
```
3. Run:
```bash
python evaluate.py --dataset eval_dataset.json --top-k 5
```

Metrics:
- **Hit Rate@K:** fraction of queries where an expected source appears in top K.
- **MRR@K:** average reciprocal rank of the first expected source.
- **Mean retrieved cosine similarity:** average similarity of retrieved chunks.

For stronger evaluation, label expected page/row/chunk identifiers and add answer correctness, citation precision, and refusal-quality metrics.

## Project structure

```text
rag_document_chatbot/
├── app.py                 # Streamlit UI, upload flow, indexing, chat, source panels
├── rag_core.py            # Chunking, embeddings, retrieval, grounded generation
├── evaluate.py            # Retrieval evaluation metrics
├── eval_dataset.json      # Editable sample evaluation cases
├── requirements.txt       # Python dependencies
├── .env.example           # API key template
├── .gitignore
└── README.md
```

## Important implementation notes

- Similarity is calculated as `1 - cosine_distance` because Chroma's configured metric is cosine.
- The relevance threshold is a retrieval gate, not a guarantee that every answer is factually perfect. Tune it against representative documents and evaluate.
- CSV rows are indexed as complete records so values from neighboring rows do not get mixed together.
- PDF text extraction works best with text-based PDFs. Scanned documents require OCR, which is not included in this MVP.
- This project uses Groq's API; Groq's free tier has usage/rate limits that can change; check its current pricing and limits. API keys should be treated as secrets.
- The initial app avoids deleting or replacing existing chunks when a file is uploaded again; identical file bytes are recognized by their hash. Clear the knowledge base before replacing a document with a revised version if needed.

## Resume description

**RAG Chatbot for Documents | Python, Streamlit, Groq API, ChromaDB**
- Built a document Q&A application for PDF, TXT, and CSV files using type-aware chunking and persistent vector search.
- Implemented Local embedding generation and cosine-similarity retrieval with a configurable relevance threshold to reduce unsupported responses.
- Generated evidence-grounded answers with file/page/row citations and inspectable retrieved chunks.
- Added retrieval evaluation using Hit Rate@K, MRR@K, and mean cosine similarity.

## Potential next improvements

- Add OCR for scanned PDFs.
- Add per-file deletion and document versioning.
- Add a reranker and hybrid keyword/vector retrieval.
- Add conversation-aware query rewriting.
- Expand evaluation with answer correctness and citation faithfulness.
- Add authentication and a hosted deployment configuration.
