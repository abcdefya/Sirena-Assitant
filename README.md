# Sirena Assistant

An agentic RAG (Retrieval-Augmented Generation) chatbot built with LangGraph, FastAPI, ChromaDB, and Groq. Sirena answers questions by intelligently deciding when to retrieve documents and when to respond from context alone.

## Architecture

```
POST /chat
    │
    ▼
[decide_node] ── needs retrieval? ──► [retrieve_node] ──► [grade_node]
    │                                                           │
    └──────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
                                                      [generate_node]
                                                      (streaming SSE)
```

- **decide** — classifies whether the question requires document retrieval
- **retrieve** — fetches relevant chunks from ChromaDB
- **grade** — filters retrieved docs for relevance
- **generate** — streams the final answer via Groq LLM

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Agent graph | LangGraph |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector store | ChromaDB |
| Rate limiting | SlowAPI |

## Getting Started

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/)

### Local setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd Sirena-Assitant

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env            # then fill in your values
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `HF_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `sirena-portfolio` | Collection name |
| `DATA_DIR` | `data` | Directory with source documents |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `RATE_LIMIT_PER_MINUTE` | `10` | Requests per minute per IP |
| `RATE_LIMIT_PER_HOUR` | `50` | Requests per hour per IP |

### Run the server

```bash
uvicorn src.api.main:app --reload
```

On startup the server ingests documents from `DATA_DIR` into ChromaDB automatically.

## API

### `POST /chat`

Send a message and receive a streaming response (Server-Sent Events).

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-123", "message": "Tell me about your experience"}'
```

**Response stream (SSE):**

```
data: {"token": "I"}
data: {"token": " have"}
...
data: {"done": true}
```

### `DELETE /chat/{session_id}`

Clear the conversation history for a session.

```bash
curl -X DELETE http://localhost:8000/chat/user-123
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### `POST /admin/ingest`

Re-trigger document ingestion manually.

```bash
curl -X POST http://localhost:8000/admin/ingest
```

## Docker

```bash
# Build
docker build -t sirena-assistant .

# Run
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  sirena-assistant
```

The image uses a multi-stage build: build tools and the HuggingFace model are baked in at build time so startup is instant.

## Project Structure

```
src/
├── api/
│   ├── main.py          # FastAPI app, lifespan, middleware
│   ├── limiter.py       # SlowAPI rate limiter
│   └── routes/
│       └── chat.py      # /chat endpoint
├── agents/
│   ├── pipeline.py      # LangGraph graph definition
│   └── state.py         # AgentState TypedDict
├── nodes/
│   ├── decide_node.py   # Retrieval decision
│   ├── retrieve_node.py # ChromaDB retrieval
│   ├── grade_node.py    # Relevance grading
│   └── generate_node.py # Streaming generation
├── services/
│   ├── groq_llm.py      # Groq LLM client
│   ├── embeddings.py    # HuggingFace embeddings
│   └── vectorstore.py   # ChromaDB ingestion & search
└── cores/
    ├── config.py        # Environment config
    ├── logger.py        # Structured logging
    └── session.py       # In-memory chat history
```

## Tests

```bash
pytest
```
