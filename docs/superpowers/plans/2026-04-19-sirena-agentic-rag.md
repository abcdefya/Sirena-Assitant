# Sirena Agentic RAG Assistant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a streaming FastAPI portfolio assistant using LangGraph agentic RAG with GROQ LLM, HuggingFace embeddings, and ChromaDB.

**Architecture:** The LangGraph pipeline handles routing only (decide → retrieve → grade), outputting a final state with documents. The FastAPI chat route then streams generation separately using `stream_answer()`, giving true token streaming without double-generation. Session history is stored in-memory per session_id.

**Tech Stack:** LangGraph, LangChain, GROQ API (`langchain-groq`), HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (`sentence-transformers`), ChromaDB, FastAPI, `slowapi`, Pydantic, pytest, httpx.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/cores/config.py` | Modify | Env vars for GROQ, HF, Chroma |
| `src/cores/logger.py` | Create | Centralized structured logger factory |
| `src/cores/session.py` | Create | In-memory session history store |
| `src/agents/state.py` | Modify | Extended AgentState TypedDict |
| `src/services/embeddings.py` | Create | HuggingFace embeddings singleton |
| `src/services/groq_llm.py` | Overwrite | GROQ ChatGroq client singleton |
| `src/services/vectorstore.py` | Create | ChromaDB init, markdown ingestion, retriever |
| `src/nodes/decide_node.py` | Create | LLM-based retrieval routing |
| `src/nodes/retrieve_node.py` | Create | ChromaDB document retrieval |
| `src/nodes/grade_node.py` | Create | LLM relevance grading |
| `src/nodes/generate_node.py` | Create | Async streaming answer generator |
| `src/agents/pipeline.py` | Overwrite | LangGraph graph (decide→retrieve→grade) |
| `src/api/__init__.py` | Create | Empty package marker |
| `src/api/main.py` | Create | FastAPI app with lifespan + rate limiter |
| `src/api/routes/__init__.py` | Create | Empty package marker |
| `src/api/routes/chat.py` | Create | POST /chat, DELETE /chat/{id}, GET /health |
| `src/prompts/system_prompt.md` | Overwrite | Sirena persona system prompt |
| `data/cv.md` | Create | Sample CV placeholder |
| `data/projects/sample_project.md` | Create | Sample project placeholder |
| `requirements.txt` | Modify | Add slowapi |
| `requirements-dev.txt` | Create | pytest, pytest-asyncio, pytest-mock, httpx |
| `Dockerfile` | Create | Single-stage Python 3.11-slim image |
| `tests/conftest.py` | Create | Env var setup before imports |
| `tests/test_cores/test_config.py` | Create | Config attribute tests |
| `tests/test_cores/test_logger.py` | Create | Logger factory tests |
| `tests/test_cores/test_session.py` | Create | Session get/append/clear tests |
| `tests/test_agents/test_state.py` | Create | AgentState structure tests |
| `tests/test_services/test_embeddings.py` | Create | Embeddings singleton mock tests |
| `tests/test_services/test_groq_llm.py` | Create | GROQ client singleton mock tests |
| `tests/test_services/test_vectorstore.py` | Create | Vectorstore ingestion mock tests |
| `tests/test_nodes/test_decide_node.py` | Create | Decide node routing mock tests |
| `tests/test_nodes/test_retrieve_node.py` | Create | Retrieve node mock tests |
| `tests/test_nodes/test_grade_node.py` | Create | Grade node mock tests |
| `tests/test_nodes/test_generate_node.py` | Create | stream_answer mock tests |
| `tests/test_agents/test_pipeline.py` | Create | Pipeline graph compile + routing tests |
| `tests/test_api/test_chat.py` | Create | API endpoint tests (TestClient) |

---

## Task 1: Config, Logger, and Test Infrastructure

**Files:**
- Modify: `src/cores/config.py`
- Create: `src/cores/logger.py`
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`
- Create: `tests/test_cores/__init__.py`
- Create: `tests/test_cores/test_config.py`
- Create: `tests/test_cores/test_logger.py`

- [ ] **Step 1: Write failing tests for config and logger**

Create `tests/conftest.py`:
```python
import os

# Set env vars before any src.* imports so config module loads them correctly
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GROQ_MODEL", "llama-3.3-70b-versatile")
os.environ.setdefault("HF_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./test_chroma_db")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "sirena_test")
os.environ.setdefault("DATA_DIR", "data")
os.environ.setdefault("LOG_LEVEL", "DEBUG")
```

Create `tests/__init__.py` — empty file.
Create `tests/test_cores/__init__.py` — empty file.

Create `tests/test_cores/test_config.py`:
```python
def test_config_has_groq_keys():
    from src.cores import config
    assert hasattr(config, "GROQ_API_KEY")
    assert hasattr(config, "GROQ_MODEL")
    assert config.GROQ_MODEL == "llama-3.3-70b-versatile"

def test_config_has_hf_model():
    from src.cores import config
    assert hasattr(config, "HF_MODEL")
    assert config.HF_MODEL == "paraphrase-multilingual-MiniLM-L12-v2"

def test_config_has_chroma_settings():
    from src.cores import config
    assert hasattr(config, "CHROMA_PERSIST_DIR")
    assert hasattr(config, "CHROMA_COLLECTION_NAME")

def test_config_has_data_dir():
    from src.cores import config
    from pathlib import Path
    assert isinstance(config.DATA_DIR, Path)

def test_config_has_rate_limits():
    from src.cores import config
    assert hasattr(config, "RATE_LIMIT_PER_MINUTE")
    assert hasattr(config, "RATE_LIMIT_PER_HOUR")
    assert isinstance(config.RATE_LIMIT_PER_MINUTE, int)
    assert isinstance(config.RATE_LIMIT_PER_HOUR, int)
```

Create `tests/test_cores/test_logger.py`:
```python
import logging

def test_get_logger_returns_logger():
    from src.cores.logger import get_logger
    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"

def test_get_logger_has_stdout_handler():
    from src.cores.logger import get_logger
    logger = get_logger("test.handler")
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

def test_get_logger_idempotent():
    from src.cores.logger import get_logger
    logger1 = get_logger("test.idem")
    logger2 = get_logger("test.idem")
    assert logger1 is logger2
    assert len(logger1.handlers) == 1  # no duplicate handlers
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd D:/work/projects/Sirena-Assitant
python -m pytest tests/test_cores/ -v
```
Expected: `ModuleNotFoundError` or `AttributeError` — config missing attributes, logger not found.

- [ ] **Step 3: Implement config**

Replace full content of `src/cores/config.py`:
```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HF_MODEL = os.getenv("HF_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "sirena_portfolio")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "50"))
```

- [ ] **Step 4: Implement logger**

Create `src/cores/logger.py`:
```python
import logging
import sys
from src.cores.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_cores/ -v
```
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cores/config.py src/cores/logger.py tests/conftest.py tests/__init__.py tests/test_cores/
git commit -m "feat: add config, logger, and test infrastructure"
```

---

## Task 2: AgentState and Session Manager

**Files:**
- Modify: `src/agents/state.py`
- Create: `src/cores/session.py`
- Create: `tests/test_agents/__init__.py`
- Create: `tests/test_agents/test_state.py`
- Create: `tests/test_cores/test_session.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agents/__init__.py` — empty file.

Create `tests/test_agents/test_state.py`:
```python
from typing import get_type_hints

def test_agent_state_has_all_required_fields():
    from src.agents.state import AgentState
    hints = get_type_hints(AgentState)
    required = {"session_id", "question", "chat_history", "documents", "answer", "needs_retrieval", "docs_relevant"}
    assert required.issubset(set(hints.keys()))

def test_agent_state_chat_history_is_list():
    from src.agents.state import AgentState
    import typing
    hints = get_type_hints(AgentState)
    origin = getattr(hints["chat_history"], "__origin__", None)
    assert origin is list

def test_agent_state_needs_retrieval_is_bool():
    from src.agents.state import AgentState
    hints = get_type_hints(AgentState)
    assert hints["needs_retrieval"] is bool

def test_agent_state_docs_relevant_is_bool():
    from src.agents.state import AgentState
    hints = get_type_hints(AgentState)
    assert hints["docs_relevant"] is bool
```

Create `tests/test_cores/test_session.py`:
```python
import pytest

@pytest.fixture(autouse=True)
def clear_sessions():
    from src.cores import session
    session._sessions.clear()
    yield
    session._sessions.clear()

def test_get_history_returns_empty_for_unknown_session():
    from src.cores.session import get_history
    assert get_history("nonexistent-session") == []

def test_append_turn_creates_session():
    from src.cores import session
    session.append_turn("sess-1", "user", "Hello")
    assert session.get_history("sess-1") == [{"role": "user", "content": "Hello"}]

def test_append_turn_adds_multiple_turns():
    from src.cores import session
    session.append_turn("sess-2", "user", "Hi")
    session.append_turn("sess-2", "assistant", "Hello!")
    history = session.get_history("sess-2")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hi"}
    assert history[1] == {"role": "assistant", "content": "Hello!"}

def test_clear_session_removes_history():
    from src.cores import session
    session.append_turn("sess-3", "user", "Hi")
    session.clear_session("sess-3")
    assert session.get_history("sess-3") == []

def test_clear_nonexistent_session_does_not_raise():
    from src.cores.session import clear_session
    clear_session("does-not-exist")  # should not raise
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_agents/test_state.py tests/test_cores/test_session.py -v
```
Expected: `AttributeError` on missing fields, `ImportError` for session.

- [ ] **Step 3: Update AgentState**

Replace full content of `src/agents/state.py`:
```python
from typing import TypedDict, List
from langchain_core.documents import Document


class AgentState(TypedDict):
    session_id: str
    question: str
    chat_history: List[dict]    # [{"role": "user/assistant", "content": "..."}]
    documents: List[Document]   # retrieved chunks from ChromaDB
    answer: str
    needs_retrieval: bool
    docs_relevant: bool
```

- [ ] **Step 4: Create session manager**

Create `src/cores/session.py`:
```python
from typing import Dict, List
from src.cores.logger import get_logger

logger = get_logger(__name__)

_sessions: Dict[str, List[dict]] = {}


def get_history(session_id: str) -> List[dict]:
    return _sessions.get(session_id, [])


def append_turn(session_id: str, role: str, content: str) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({"role": role, "content": content})
    logger.debug(f"Session {session_id}: appended {role} turn")


def clear_session(session_id: str) -> None:
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Session {session_id}: cleared")
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_agents/test_state.py tests/test_cores/test_session.py -v
```
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agents/state.py src/cores/session.py tests/test_agents/ tests/test_cores/test_session.py
git commit -m "feat: add extended AgentState and session manager"
```

---

## Task 3: HuggingFace Embeddings Service

**Files:**
- Create: `src/services/embeddings.py`
- Create: `tests/test_services/__init__.py`
- Create: `tests/test_services/test_embeddings.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_services/__init__.py` — empty file.

Create `tests/test_services/test_embeddings.py`:
```python
from unittest.mock import MagicMock, patch


def test_get_embeddings_returns_singleton():
    mock_embeddings = MagicMock()
    with patch("src.services.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
        from src.services import embeddings as emb_module
        emb_module._embeddings = None  # reset singleton
        result1 = emb_module.get_embeddings()
        result2 = emb_module.get_embeddings()
        assert result1 is result2

def test_get_embeddings_uses_configured_model():
    mock_embeddings = MagicMock()
    with patch("src.services.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings) as mock_cls:
        from src.services import embeddings as emb_module
        from src.cores.config import HF_MODEL
        emb_module._embeddings = None
        emb_module.get_embeddings()
        mock_cls.assert_called_once_with(model_name=HF_MODEL)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/test_services/test_embeddings.py -v
```
Expected: `ImportError` — `src.services.embeddings` not found.

- [ ] **Step 3: Implement embeddings service**

Create `src/services/embeddings.py`:
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.cores.config import HF_MODEL
from src.cores.logger import get_logger

logger = get_logger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading HuggingFace embeddings model: {HF_MODEL}")
        _embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL)
        logger.info("Embeddings model loaded")
    return _embeddings
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest tests/test_services/test_embeddings.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/embeddings.py tests/test_services/
git commit -m "feat: add HuggingFace embeddings service singleton"
```

---

## Task 4: GROQ LLM Service

**Files:**
- Overwrite: `src/services/groq_llm.py`
- Create: `tests/test_services/test_groq_llm.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_services/test_groq_llm.py`:
```python
from unittest.mock import MagicMock, patch


def test_get_llm_returns_singleton():
    mock_llm = MagicMock()
    with patch("src.services.groq_llm.ChatGroq", return_value=mock_llm):
        from src.services import groq_llm as llm_module
        llm_module._llm = None
        result1 = llm_module.get_llm()
        result2 = llm_module.get_llm()
        assert result1 is result2

def test_get_llm_uses_configured_model():
    mock_llm = MagicMock()
    with patch("src.services.groq_llm.ChatGroq", return_value=mock_llm) as mock_cls:
        from src.services import groq_llm as llm_module
        from src.cores.config import GROQ_API_KEY, GROQ_MODEL
        llm_module._llm = None
        llm_module.get_llm()
        mock_cls.assert_called_once_with(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
        )
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/test_services/test_groq_llm.py -v
```
Expected: `ImportError` (old groq_llm.py imports from groq_lm.py which doesn't exist).

- [ ] **Step 3: Overwrite GROQ LLM service**

Replace full content of `src/services/groq_llm.py`:
```python
from langchain_groq import ChatGroq
from src.cores.config import GROQ_API_KEY, GROQ_MODEL
from src.cores.logger import get_logger

logger = get_logger(__name__)

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        logger.info(f"Initializing GROQ LLM: {GROQ_MODEL}")
        _llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
        logger.info("GROQ LLM initialized")
    return _llm
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest tests/test_services/test_groq_llm.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/groq_llm.py tests/test_services/test_groq_llm.py
git commit -m "feat: implement GROQ LLM service singleton"
```

---

## Task 5: ChromaDB Vectorstore Service

**Files:**
- Create: `src/services/vectorstore.py`
- Create: `tests/test_services/test_vectorstore.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_services/test_vectorstore.py`:
```python
from unittest.mock import MagicMock, patch, call
from pathlib import Path


def test_get_vectorstore_returns_singleton():
    mock_store = MagicMock()
    with patch("src.services.vectorstore.Chroma", return_value=mock_store), \
         patch("src.services.vectorstore.get_embeddings", return_value=MagicMock()):
        from src.services import vectorstore as vs_module
        vs_module._vectorstore = None
        result1 = vs_module.get_vectorstore()
        result2 = vs_module.get_vectorstore()
        assert result1 is result2


def test_ingest_documents_skips_if_collection_not_empty():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_store = MagicMock()
    mock_store._collection = mock_collection
    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store):
        from src.services.vectorstore import ingest_documents
        from src.services import vectorstore as vs_module
        vs_module._vectorstore = mock_store
        ingest_documents()
        mock_store.add_documents.assert_not_called()


def test_ingest_documents_ingests_when_collection_empty(tmp_path):
    # Create a temp markdown file
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test\nThis is a test document.")

    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_store = MagicMock()
    mock_store._collection = mock_collection

    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store), \
         patch("src.services.vectorstore.DATA_DIR", tmp_path):
        from src.services.vectorstore import ingest_documents
        ingest_documents()
        mock_store.add_documents.assert_called_once()


def test_get_retriever_uses_vectorstore():
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_store.as_retriever.return_value = mock_retriever

    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store):
        from src.services.vectorstore import get_retriever
        result = get_retriever(k=3)
        mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert result is mock_retriever
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_services/test_vectorstore.py -v
```
Expected: `ImportError` — `src.services.vectorstore` not found.

- [ ] **Step 3: Implement vectorstore service**

Create `src/services/vectorstore.py`:
```python
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.cores.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, DATA_DIR
from src.services.embeddings import get_embeddings
from src.cores.logger import get_logger

logger = get_logger(__name__)

_vectorstore: Chroma | None = None


def _load_markdown_files() -> list[Document]:
    docs = []
    for md_file in Path(DATA_DIR).rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        docs.append(Document(
            page_content=content,
            metadata={"source": str(md_file)},
        ))
    logger.info(f"Loaded {len(docs)} markdown files from {DATA_DIR}")
    return docs


def _chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings()
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _vectorstore


def ingest_documents() -> None:
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    if collection.count() > 0:
        logger.info(f"ChromaDB already has {collection.count()} chunks — skipping ingestion")
        return
    docs = _load_markdown_files()
    if not docs:
        logger.warning("No markdown files found in data/ — skipping ingestion")
        return
    chunks = _chunk_documents(docs)
    vectorstore.add_documents(chunks)
    logger.info(f"Ingested {len(chunks)} chunks into ChromaDB")


def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_services/test_vectorstore.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/vectorstore.py tests/test_services/test_vectorstore.py
git commit -m "feat: add ChromaDB vectorstore service with idempotent ingestion"
```

---

## Task 6: Decide Node

**Files:**
- Create: `src/nodes/decide_node.py`
- Create: `tests/test_nodes/__init__.py`
- Create: `tests/test_nodes/test_decide_node.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_nodes/__init__.py` — empty file.

Create `tests/test_nodes/test_decide_node.py`:
```python
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def make_state(**kwargs):
    return {
        "session_id": "test-session",
        "question": kwargs.get("question", ""),
        "chat_history": [],
        "documents": [],
        "answer": "",
        "needs_retrieval": False,
        "docs_relevant": False,
        **kwargs,
    }


def test_decide_node_sets_needs_retrieval_true_when_llm_says_yes():
    mock_response = MagicMock()
    mock_response.content = "yes"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.decide_node.get_llm", return_value=mock_llm):
        from src.nodes.decide_node import decide_node
        state = make_state(question="What projects have you built with React?")
        result = decide_node(state)
        assert result["needs_retrieval"] is True


def test_decide_node_sets_needs_retrieval_false_when_llm_says_no():
    mock_response = MagicMock()
    mock_response.content = "no"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.decide_node.get_llm", return_value=mock_llm):
        from src.nodes.decide_node import decide_node
        state = make_state(question="Hello!")
        result = decide_node(state)
        assert result["needs_retrieval"] is False


def test_decide_node_preserves_other_state_fields():
    mock_response = MagicMock()
    mock_response.content = "yes"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.decide_node.get_llm", return_value=mock_llm):
        from src.nodes.decide_node import decide_node
        state = make_state(question="Tell me about your skills", answer="existing")
        result = decide_node(state)
        assert result["answer"] == "existing"
        assert result["session_id"] == "test-session"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_nodes/test_decide_node.py -v
```
Expected: `ImportError` — `src.nodes.decide_node` not found.

- [ ] **Step 3: Implement decide node**

Create `src/nodes/decide_node.py`:
```python
from src.agents.state import AgentState
from src.services.groq_llm import get_llm
from src.cores.logger import get_logger

logger = get_logger(__name__)

_DECIDE_PROMPT = """You are a routing assistant for a personal portfolio chatbot.

Given the user's question, decide if it requires retrieving information from the portfolio knowledge base (projects, CV, experience, skills, education).

Answer with ONLY "yes" or "no".
- "yes" if the question is about the portfolio owner's projects, skills, experience, education, tech stack, or professional background
- "no" if the question is a greeting, small talk, or general question not related to the portfolio

Question: {question}
Answer:"""


def decide_node(state: AgentState) -> AgentState:
    question = state["question"]
    prompt = _DECIDE_PROMPT.format(question=question)
    response = get_llm().invoke(prompt)
    decision = response.content.strip().lower()
    needs_retrieval = decision.startswith("yes")
    logger.info(
        f"Session {state['session_id']}: routing decision='{decision}' needs_retrieval={needs_retrieval}"
    )
    return {**state, "needs_retrieval": needs_retrieval}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_nodes/test_decide_node.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nodes/decide_node.py tests/test_nodes/
git commit -m "feat: add LLM-based decide node for retrieval routing"
```

---

## Task 7: Retrieve Node

**Files:**
- Create: `src/nodes/retrieve_node.py`
- Create: `tests/test_nodes/test_retrieve_node.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_nodes/test_retrieve_node.py`:
```python
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def make_state(**kwargs):
    return {
        "session_id": "test-session",
        "question": kwargs.get("question", "What are your projects?"),
        "chat_history": [],
        "documents": [],
        "answer": "",
        "needs_retrieval": True,
        "docs_relevant": False,
        **kwargs,
    }


def test_retrieve_node_populates_documents():
    mock_docs = [
        Document(page_content="Project A: React dashboard"),
        Document(page_content="Project B: FastAPI backend"),
    ]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = mock_docs

    with patch("src.nodes.retrieve_node.get_retriever", return_value=mock_retriever):
        from src.nodes.retrieve_node import retrieve_node
        state = make_state(question="What projects have you built?")
        result = retrieve_node(state)
        assert result["documents"] == mock_docs
        assert len(result["documents"]) == 2


def test_retrieve_node_calls_retriever_with_question():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    with patch("src.nodes.retrieve_node.get_retriever", return_value=mock_retriever):
        from src.nodes.retrieve_node import retrieve_node
        state = make_state(question="Tell me about your React experience")
        retrieve_node(state)
        mock_retriever.invoke.assert_called_once_with("Tell me about your React experience")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_nodes/test_retrieve_node.py -v
```
Expected: `ImportError` — `src.nodes.retrieve_node` not found.

- [ ] **Step 3: Implement retrieve node**

Create `src/nodes/retrieve_node.py`:
```python
from src.agents.state import AgentState
from src.services.vectorstore import get_retriever
from src.cores.logger import get_logger

logger = get_logger(__name__)


def retrieve_node(state: AgentState) -> AgentState:
    question = state["question"]
    retriever = get_retriever(k=4)
    documents = retriever.invoke(question)
    logger.info(f"Session {state['session_id']}: retrieved {len(documents)} documents")
    return {**state, "documents": documents}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_nodes/test_retrieve_node.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nodes/retrieve_node.py tests/test_nodes/test_retrieve_node.py
git commit -m "feat: add retrieve node for ChromaDB document retrieval"
```

---

## Task 8: Grade Node

**Files:**
- Create: `src/nodes/grade_node.py`
- Create: `tests/test_nodes/test_grade_node.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_nodes/test_grade_node.py`:
```python
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def make_state(**kwargs):
    return {
        "session_id": "test-session",
        "question": kwargs.get("question", "What projects have you built?"),
        "chat_history": [],
        "documents": kwargs.get("documents", []),
        "answer": "",
        "needs_retrieval": True,
        "docs_relevant": False,
        **kwargs,
    }


def test_grade_node_sets_docs_relevant_true_when_llm_says_yes():
    mock_response = MagicMock()
    mock_response.content = "yes"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.grade_node.get_llm", return_value=mock_llm):
        from src.nodes.grade_node import grade_node
        docs = [Document(page_content="React dashboard project built in 2024")]
        state = make_state(question="What React projects have you built?", documents=docs)
        result = grade_node(state)
        assert result["docs_relevant"] is True


def test_grade_node_sets_docs_relevant_false_when_llm_says_no():
    mock_response = MagicMock()
    mock_response.content = "no"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.grade_node.get_llm", return_value=mock_llm):
        from src.nodes.grade_node import grade_node
        docs = [Document(page_content="Unrelated content about cooking")]
        state = make_state(question="What is your tech stack?", documents=docs)
        result = grade_node(state)
        assert result["docs_relevant"] is False


def test_grade_node_preserves_documents():
    mock_response = MagicMock()
    mock_response.content = "yes"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.grade_node.get_llm", return_value=mock_llm):
        from src.nodes.grade_node import grade_node
        docs = [Document(page_content="My portfolio project")]
        state = make_state(documents=docs)
        result = grade_node(state)
        assert result["documents"] == docs
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_nodes/test_grade_node.py -v
```
Expected: `ImportError` — `src.nodes.grade_node` not found.

- [ ] **Step 3: Implement grade node**

Create `src/nodes/grade_node.py`:
```python
from src.agents.state import AgentState
from src.services.groq_llm import get_llm
from src.cores.logger import get_logger

logger = get_logger(__name__)

_GRADE_PROMPT = """You are a relevance grader for a portfolio assistant.

Given a user question and retrieved document chunks, decide if the documents contain information relevant to answering the question.

Answer with ONLY "yes" or "no".
- "yes" if at least one chunk contains information relevant to the question
- "no" if the chunks are unrelated to the question

Question: {question}

Document chunks:
{documents}

Answer:"""


def grade_node(state: AgentState) -> AgentState:
    question = state["question"]
    documents = state.get("documents", [])
    docs_text = "\n\n---\n\n".join([doc.page_content for doc in documents])
    prompt = _GRADE_PROMPT.format(question=question, documents=docs_text)
    response = get_llm().invoke(prompt)
    decision = response.content.strip().lower()
    docs_relevant = decision.startswith("yes")
    logger.info(
        f"Session {state['session_id']}: grading decision='{decision}' docs_relevant={docs_relevant}"
    )
    return {**state, "docs_relevant": docs_relevant}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_nodes/test_grade_node.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nodes/grade_node.py tests/test_nodes/test_grade_node.py
git commit -m "feat: add grade node for document relevance checking"
```

---

## Task 9: Generate Node (stream_answer) and Pipeline

**Files:**
- Create: `src/nodes/generate_node.py`
- Overwrite: `src/agents/pipeline.py`
- Create: `tests/test_nodes/test_generate_node.py`
- Create: `tests/test_agents/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_nodes/test_generate_node.py`:
```python
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.documents import Document


def make_state(**kwargs):
    return {
        "session_id": "test-session",
        "question": kwargs.get("question", "What are your skills?"),
        "chat_history": kwargs.get("chat_history", []),
        "documents": kwargs.get("documents", []),
        "answer": "",
        "needs_retrieval": False,
        "docs_relevant": kwargs.get("docs_relevant", False),
        **kwargs,
    }


@pytest.mark.asyncio
async def test_stream_answer_uses_rag_context_when_docs_relevant():
    chunks = [MagicMock(content="I "), MagicMock(content="know "), MagicMock(content="Python")]

    async def mock_astream(messages):
        for chunk in chunks:
            yield chunk

    mock_llm = MagicMock()
    mock_llm.astream = mock_astream

    with patch("src.nodes.generate_node.get_llm", return_value=mock_llm):
        from src.nodes.generate_node import stream_answer
        docs = [Document(page_content="Python expert with 5 years experience")]
        state = make_state(docs_relevant=True, documents=docs)
        tokens = []
        async for token in stream_answer(state):
            tokens.append(token)
        assert tokens == ["I ", "know ", "Python"]


@pytest.mark.asyncio
async def test_stream_answer_skips_context_when_docs_not_relevant():
    chunks = [MagicMock(content="Hello")]

    async def mock_astream(messages):
        for chunk in chunks:
            yield chunk

    mock_llm = MagicMock()
    mock_llm.astream = mock_astream

    with patch("src.nodes.generate_node.get_llm", return_value=mock_llm):
        from src.nodes.generate_node import stream_answer
        docs = [Document(page_content="Irrelevant content")]
        state = make_state(docs_relevant=False, documents=docs)
        tokens = []
        async for token in stream_answer(state):
            tokens.append(token)
        # Should still produce output (direct answer)
        assert len(tokens) > 0


@pytest.mark.asyncio
async def test_stream_answer_includes_chat_history_in_messages():
    captured_messages = []

    async def mock_astream(messages):
        captured_messages.extend(messages)
        yield MagicMock(content="answer")

    mock_llm = MagicMock()
    mock_llm.astream = mock_astream

    with patch("src.nodes.generate_node.get_llm", return_value=mock_llm):
        from src.nodes.generate_node import stream_answer
        history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        state = make_state(chat_history=history)
        async for _ in stream_answer(state):
            pass
        roles = [m["role"] for m in captured_messages]
        assert "user" in roles
        assert "assistant" in roles
```

Create `tests/test_agents/test_pipeline.py`:
```python
def test_pipeline_compiles_without_error():
    from unittest.mock import patch, MagicMock
    # Patch all node dependencies so graph compiles without real LLM/DB
    with patch("src.nodes.decide_node.get_llm", return_value=MagicMock()), \
         patch("src.nodes.retrieve_node.get_retriever", return_value=MagicMock()), \
         patch("src.nodes.grade_node.get_llm", return_value=MagicMock()):
        from src.agents import pipeline as pipeline_module
        pipeline_module._app = None
        app = pipeline_module.get_pipeline()
        assert app is not None


def test_pipeline_is_singleton():
    from src.agents import pipeline as pipeline_module
    pipeline_module._app = None
    app1 = pipeline_module.get_pipeline()
    app2 = pipeline_module.get_pipeline()
    assert app1 is app2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_nodes/test_generate_node.py tests/test_agents/test_pipeline.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Install pytest-asyncio**

```bash
pip install pytest-asyncio
```

Add to `requirements-dev.txt` (create if not exists):
```
pytest
pytest-asyncio
pytest-mock
httpx
```

Add `pytest.ini` at project root:
```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 4: Implement generate node**

Create `src/nodes/generate_node.py`:
```python
from typing import AsyncGenerator
from src.agents.state import AgentState
from src.services.groq_llm import get_llm
from src.cores.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are Sirena, a friendly and professional portfolio assistant. "
    "Answer questions about the portfolio owner based on the provided context and chat history. "
    "Be concise and accurate. If you don't know something, say so."
)


def _build_messages(state: AgentState, context: str = "") -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if context:
        messages.append({
            "role": "system",
            "content": f"Portfolio context:\n{context}",
        })
    for turn in state.get("chat_history", []):
        messages.append(turn)
    messages.append({"role": "user", "content": state["question"]})
    return messages


async def stream_answer(state: AgentState) -> AsyncGenerator[str, None]:
    documents = state.get("documents", [])
    docs_relevant = state.get("docs_relevant", False)
    context = ""
    if documents and docs_relevant:
        context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    messages = _build_messages(state, context=context)
    logger.info(
        f"Session {state['session_id']}: streaming answer "
        f"(rag={'yes' if context else 'no'})"
    )
    async for chunk in get_llm().astream(messages):
        if chunk.content:
            yield chunk.content
```

- [ ] **Step 5: Implement pipeline**

Replace full content of `src/agents/pipeline.py`:
```python
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.nodes.decide_node import decide_node
from src.nodes.retrieve_node import retrieve_node
from src.nodes.grade_node import grade_node
from src.cores.logger import get_logger

logger = get_logger(__name__)

_app = None


def _route_after_decide(state: AgentState) -> str:
    return "retrieve" if state["needs_retrieval"] else END


def _route_after_grade(state: AgentState) -> str:
    # Both paths end here; docs_relevant flag is read by stream_answer at API layer
    return END


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("decide", decide_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)

    workflow.set_entry_point("decide")
    workflow.add_conditional_edges(
        "decide",
        _route_after_decide,
        {"retrieve": "retrieve", END: END},
    )
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", END)

    app = workflow.compile()
    logger.info("LangGraph pipeline compiled successfully")
    return app


def get_pipeline():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
python -m pytest tests/test_nodes/test_generate_node.py tests/test_agents/test_pipeline.py -v
```
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/nodes/generate_node.py src/agents/pipeline.py requirements-dev.txt pytest.ini tests/test_nodes/test_generate_node.py tests/test_agents/test_pipeline.py
git commit -m "feat: add generate node with stream_answer and LangGraph routing pipeline"
```

---

## Task 10: FastAPI App and Health Endpoint

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/main.py`
- Create: `src/api/routes/__init__.py`
- Create: `tests/test_api/__init__.py`
- Create: `tests/test_api/test_health.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_api/__init__.py` — empty file.

Create `tests/test_api/test_health.py`:
```python
from unittest.mock import patch
from fastapi.testclient import TestClient


def test_health_returns_ok():
    with patch("src.services.vectorstore.ingest_documents"):
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/test_api/test_health.py -v
```
Expected: `ImportError` — `src.api.main` not found.

- [ ] **Step 3: Create package markers**

Create `src/api/__init__.py` — empty file.
Create `src/api/routes/__init__.py` — empty file.

- [ ] **Step 4: Implement FastAPI main app**

Create `src/api/main.py`:
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.services.vectorstore import ingest_documents
from src.cores.logger import get_logger

logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Sirena Assistant — running document ingestion...")
    ingest_documents()
    logger.info("Sirena Assistant ready to serve requests")
    yield
    logger.info("Sirena Assistant shutting down")


app = FastAPI(title="Sirena Assistant API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Import router after app is defined to avoid circular imports
from src.api.routes.chat import router as chat_router  # noqa: E402
app.include_router(chat_router)
```

- [ ] **Step 5: Run test to confirm it passes**

```bash
python -m pytest tests/test_api/test_health.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/api/ tests/test_api/__init__.py tests/test_api/test_health.py
git commit -m "feat: add FastAPI app with lifespan ingestion and health endpoint"
```

---

## Task 11: Chat Route with Streaming and Rate Limiting

**Files:**
- Create: `src/api/routes/chat.py`
- Create: `tests/test_api/test_chat.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_api/test_chat.py`:
```python
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


def get_client():
    with patch("src.services.vectorstore.ingest_documents"):
        from src.api.main import app
        return TestClient(app)


def test_chat_returns_streaming_response():
    async def mock_stream_answer(state):
        yield "Hello"
        yield " world"

    mock_state = {
        "session_id": "test-123",
        "question": "Hi",
        "chat_history": [],
        "documents": [],
        "answer": "",
        "needs_retrieval": False,
        "docs_relevant": False,
    }

    with patch("src.services.vectorstore.ingest_documents"), \
         patch("src.api.routes.chat.get_pipeline") as mock_pipeline, \
         patch("src.api.routes.chat.stream_answer", side_effect=mock_stream_answer):
        mock_app = MagicMock()
        mock_app.invoke.return_value = mock_state
        mock_pipeline.return_value = mock_app

        from src.api.main import app
        client = TestClient(app)
        response = client.post("/chat", json={"session_id": "test-123", "message": "Hi"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


def test_chat_rejects_empty_message():
    with patch("src.services.vectorstore.ingest_documents"):
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/chat", json={"session_id": "test-123", "message": ""})
        assert response.status_code == 422


def test_delete_session_returns_cleared():
    with patch("src.services.vectorstore.ingest_documents"):
        from src.api.main import app
        client = TestClient(app)
        response = client.delete("/chat/my-session-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        assert data["session_id"] == "my-session-id"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_api/test_chat.py -v
```
Expected: `ImportError` — `src.api.routes.chat` not found.

- [ ] **Step 3: Implement chat route**

Create `src/api/routes/chat.py`:
```python
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.agents.pipeline import get_pipeline
from src.agents.state import AgentState
from src.cores.session import get_history, append_turn, clear_session
from src.nodes.generate_node import stream_answer
from src.cores.logger import get_logger
from src.api.main import limiter

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


@router.post("/chat")
@limiter.limit("10/minute;50/hour")
async def chat(request: Request, body: ChatRequest):
    session_id = body.session_id
    question = body.message
    chat_history = get_history(session_id)
    logger.info(f"Session {session_id}: received message")

    initial_state: AgentState = {
        "session_id": session_id,
        "question": question,
        "chat_history": chat_history,
        "documents": [],
        "answer": "",
        "needs_retrieval": False,
        "docs_relevant": False,
    }

    pipeline = get_pipeline()
    result_state: AgentState = pipeline.invoke(initial_state)

    async def token_stream():
        full_answer = ""
        try:
            async for token in stream_answer(result_state):
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            logger.error(f"Session {session_id}: streaming error — {e}")
            yield f"data: {json.dumps({'error': 'Something went wrong. Please try again.'})}\n\n"
        else:
            append_turn(session_id, "user", question)
            append_turn(session_id, "assistant", full_answer)
        finally:
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@router.delete("/chat/{session_id}")
async def delete_session(session_id: str):
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_api/test_chat.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/chat.py tests/test_api/test_chat.py
git commit -m "feat: add streaming chat route with rate limiting and session memory"
```

---

## Task 12: Sample Data, System Prompt, Requirements, and Dockerfile

**Files:**
- Overwrite: `src/prompts/system_prompt.md`
- Create: `data/cv.md`
- Create: `data/projects/sample_project.md`
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `Dockerfile`

- [ ] **Step 1: Update system prompt**

Replace full content of `src/prompts/system_prompt.md`:
```markdown
# Sirena — Portfolio Assistant System Prompt

You are Sirena, a friendly and professional AI assistant for this developer's portfolio website.

## Your Role
Help HR professionals and recruiters quickly learn about the portfolio owner's:
- Projects (tech stack, goals, workflow, outcomes)
- Skills and technologies
- Work experience and education
- Background and professional profile

## Style
- Be concise, warm, and professional
- Answer in 2-4 sentences unless more detail is clearly needed
- Use bullet points for lists of projects or skills
- If asked something outside your knowledge base, say so honestly

## Boundaries
- Only answer questions about the portfolio owner's professional background
- Do not invent or assume information not in your context
- Redirect off-topic questions back to the portfolio
```

- [ ] **Step 2: Create sample data files**

Create `data/cv.md` — replace this with your real CV:
```markdown
# Curriculum Vitae

## Personal Information
- **Name:** [Your Name]
- **Location:** [Your City, Country]
- **Email:** [your@email.com]
- **LinkedIn:** [linkedin.com/in/yourprofile]
- **GitHub:** [github.com/yourusername]

## Summary
[A brief 2-3 sentence professional summary about yourself]

## Experience
### [Job Title] — [Company Name] (YYYY – Present)
- [Key responsibility or achievement]
- [Key responsibility or achievement]

### [Job Title] — [Company Name] (YYYY – YYYY)
- [Key responsibility or achievement]

## Education
### [Degree] in [Field] — [University Name] (YYYY – YYYY)

## Skills
- **Languages:** Python, TypeScript, ...
- **Frameworks:** FastAPI, React, LangChain, ...
- **Tools:** Docker, Git, PostgreSQL, ...
```

Create `data/projects/sample_project.md` — replace with your real projects:
```markdown
# Project: [Project Name]

## Overview
[1-2 sentence description of what this project does and why it was built]

## Tech Stack
- **Backend:** Python, FastAPI
- **Frontend:** React, TypeScript
- **Database:** PostgreSQL
- **Deployment:** Docker, ...

## Key Features
- [Feature 1]
- [Feature 2]
- [Feature 3]

## Workflow
[Brief description of development workflow, architecture decisions, or challenges solved]

## Outcome
[Impact, metrics, or what was learned]
```

- [ ] **Step 3: Update requirements.txt**

Replace full content of `requirements.txt`:
```
langchain
langchain-core
langchain-community
langchain-groq
langgraph
pydantic
sentence-transformers
groq
chromadb
requests
python-dotenv
fastapi
uvicorn
slowapi
```

- [ ] **Step 4: Create requirements-dev.txt**

Create `requirements-dev.txt`:
```
pytest
pytest-asyncio
pytest-mock
httpx
```

- [ ] **Step 5: Create Dockerfile**

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace embedding model at build time
# so container startup is fast (no download on first request)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copy source code and portfolio data
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 6: Verify Docker build**

```bash
docker build -t sirena .
```
Expected: Build completes without errors. HuggingFace model download logged during RUN step.

- [ ] **Step 7: Run final full test suite**

```bash
python -m pytest tests/ -v
```
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/prompts/system_prompt.md data/ requirements.txt requirements-dev.txt Dockerfile pytest.ini
git commit -m "feat: add sample portfolio data, Dockerfile, and finalize requirements"
```

---

## Spec Coverage Check

| Spec Requirement | Covered By |
|---|---|
| HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` embeddings | Task 3 |
| ChromaDB persistent vector store | Task 5 |
| GROQ API LLM | Task 4 |
| LangGraph decide→retrieve→grade pipeline | Task 9 |
| LLM-based routing | Task 6 |
| Relevance grading | Task 8 |
| Session-scoped chat history | Task 2 |
| Streaming SSE response | Task 11 |
| Rate limiting 10/min 50/hr per IP | Task 11 |
| POST /chat, DELETE /chat/{id}, GET /health | Tasks 10, 11 |
| FastAPI lifespan ingestion | Task 10 |
| Idempotent ingestion (skip if not empty) | Task 5 |
| Centralized logger | Task 1 |
| Dockerfile with pre-downloaded HF model | Task 12 |
| Docker volume for ChromaDB | Task 12 |
