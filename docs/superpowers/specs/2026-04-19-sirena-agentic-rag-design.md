# Sirena Agentic RAG Assistant — Design Spec

**Date:** 2026-04-19  
**Status:** Approved

---

## Overview

Sirena is a personal portfolio assistant embedded in a portfolio website. HR visitors can chat with Sirena to quickly learn about the owner's projects, tech stack, experience, and education. The agent uses agentic RAG (Retrieval-Augmented Generation) via LangGraph to decide whether to retrieve knowledge from a ChromaDB vector store before generating an answer.

---

## Tech Stack

| Component | Choice |
|---|---|
| LLM | GROQ API (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector store | ChromaDB (persistent, local) |
| Agent framework | LangGraph |
| API | FastAPI (streaming SSE) |
| Deployment | Docker (single-stage, Python 3.11-slim) |

---

## Project Structure

```
Sirena-Assitant/
├── data/
│   ├── projects/
│   │   └── *.md              ← project markdown files
│   └── cv.md                 ← personal info / CV
├── src/
│   ├── agents/
│   │   ├── state.py          ← AgentState TypedDict
│   │   └── pipeline.py       ← LangGraph graph builder + compiled app
│   ├── nodes/
│   │   ├── decide_node.py    ← LLM-based routing
│   │   ├── retrieve_node.py  ← ChromaDB retrieval
│   │   ├── grade_node.py     ← relevance grading
│   │   └── generate_node.py  ← streaming answer generation
│   ├── services/
│   │   ├── groq_llm.py       ← GROQ LLM client singleton
│   │   ├── embeddings.py     ← HuggingFace embeddings singleton
│   │   └── vectorstore.py    ← ChromaDB setup & markdown ingestion
│   ├── cores/
│   │   ├── config.py         ← env vars
│   │   ├── logger.py         ← centralized logger
│   │   └── session.py        ← in-memory session store
│   ├── prompts/
│   │   └── system_prompt.md  ← Sirena persona prompt
│   └── api/
│       ├── main.py           ← FastAPI app + lifespan startup
│       └── routes/
│           └── chat.py       ← POST /chat streaming endpoint
├── Dockerfile
├── requirements.txt
└── .env
```

---

## Agent State

```python
class AgentState(TypedDict):
    session_id: str
    question: str
    chat_history: List[dict]   # [{"role": "user/assistant", "content": "..."}]
    documents: List[Document]  # retrieved chunks from ChromaDB
    answer: str
    needs_retrieval: bool
    docs_relevant: bool
```

---

## LangGraph Pipeline

```
         ┌──────────┐
         │  decide  │  ← LLM decides: does this question need retrieval?
         └────┬─────┘
      YES     │     NO
   ┌──────────┘     └──────────────────┐
   ▼                                   ▼
┌──────────┐                   ┌───────────────┐
│ retrieve │                   │ generate_     │
│          │                   │ direct        │
└────┬─────┘                   └───────────────┘
     ▼
┌──────────┐
│  grade   │  ← LLM checks: are retrieved docs actually relevant?
└────┬─────┘
  YES│    NO
     │     └──────────────────┐
     ▼                        ▼
┌──────────┐         ┌───────────────┐
│ generate │         │ generate_     │
│ (w/ RAG) │         │ direct        │
└──────────┘         └───────────────┘
```

### Node Responsibilities

- **decide_node:** LLM call (small/fast) to classify the question — does it require portfolio knowledge or can it be answered directly (e.g. greetings)?
- **retrieve_node:** Queries ChromaDB using HuggingFace embeddings to fetch top-k relevant document chunks.
- **grade_node:** LLM call to verify retrieved chunks are actually relevant to the question. Returns `docs_relevant: bool`.
- **generate_node:** Streams answer from GROQ using retrieved context + chat history in the prompt.
- **generate_direct_node:** Streams answer from GROQ using only chat history (no retrieval context).

---

## Session Memory

- Stored in `cores/session.py` as an in-memory Python dict: `{ session_id: List[dict] }`
- Each `POST /chat` request loads history by `session_id`, passes it into `AgentState`
- After generation, the new user/assistant turn is appended to session history
- Sessions are not persisted across container restarts (acceptable for portfolio use)

---

## API

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send message, receive streaming response |
| `DELETE` | `/chat/{session_id}` | Clear session history |
| `GET` | `/health` | Health check |

### POST /chat

**Request:**
```json
{
  "session_id": "abc-123",
  "message": "What projects have you worked on with React?"
}
```

**Response:** `StreamingResponse` (`text/event-stream`)

**Stream format (SSE):**
```
data: {"token": "I"}
data: {"token": " have"}
...
data: {"done": true}
```

### Error Handling

- Missing/new `session_id` → auto-create session (no error)
- Empty `message` → `422` (Pydantic validation)
- GROQ API failure → `500` with friendly message, no internal details leaked
- ChromaDB unavailable at startup → log error and exit (fail fast)

---

## ChromaDB & Ingestion

- **Trigger:** FastAPI `lifespan` startup event
- **Process:** Read all `.md` files from `data/`, chunk with `RecursiveCharacterTextSplitter`, embed with HuggingFace, upsert into ChromaDB collection
- **Persistence:** ChromaDB data stored at `CHROMA_PERSIST_DIR` (mounted as Docker volume) — survives container restarts without re-ingestion
- **Re-ingestion:** Only runs if collection is empty; otherwise skips (idempotent startup)

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker volume for ChromaDB:**
```bash
docker run -v sirena_chroma:/app/chroma_db --env-file .env -p 8000:8000 sirena
```

---

## Environment Variables

```
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
CHROMA_PERSIST_DIR=./chroma_db
HF_MODEL=paraphrase-multilingual-MiniLM-L12-v2
LOG_LEVEL=INFO
```

---

## Logging

All modules import from `src.cores.logger`. Logger writes structured output to stdout (suitable for Docker log collection). Key events logged:

- App startup / ChromaDB ingestion status
- Per-request: session_id, routing decision, retrieval count, grading result
- Errors: GROQ failures, ChromaDB errors
