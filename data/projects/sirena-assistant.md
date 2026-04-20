# Project: Sirena — Agentic RAG Portfolio Assistant

## Overview
Sirena is a personal AI assistant embedded in my portfolio website that helps HR professionals and recruiters quickly learn about my background, projects, skills, and experience through natural conversation. It uses agentic RAG (Retrieval-Augmented Generation) to decide whether to retrieve relevant knowledge before answering.

## Tech Stack
- **LLM:** GROQ API (llama-3.3-70b-versatile) for fast inference
- **Embeddings:** HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
- **Vector Store:** ChromaDB (persistent, local)
- **Agent Framework:** LangGraph (agentic routing pipeline)
- **Backend:** FastAPI with streaming SSE responses
- **Rate Limiting:** slowapi (per-IP, 10 req/min)
- **Deployment:** Docker (single-stage Python 3.11-slim)

## Architecture & Workflow

The core is a LangGraph routing pipeline with three nodes:

1. **Decide Node** — An LLM call classifies whether the user's question requires retrieving portfolio knowledge (e.g. "What projects have you built?") or can be answered directly (e.g. greetings).
2. **Retrieve Node** — Queries ChromaDB using semantic similarity to fetch the most relevant chunks from portfolio markdown files (CV, project descriptions).
3. **Grade Node** — A second LLM call verifies that the retrieved documents are actually relevant before using them as context.

After routing, the FastAPI layer streams the answer token-by-token using Server-Sent Events (SSE), so the portfolio website shows a live typing effect. Session chat history is maintained in-memory per session ID, so HR visitors can have multi-turn conversations within a session.

## Key Features
- Agentic routing: LLM decides retrieval vs. direct answer — no brittle keyword matching
- Relevance grading: retrieved documents are validated before use, preventing hallucinated answers
- True streaming: tokens stream from the GROQ API directly to the browser
- Session memory: multi-turn conversation within a session
- Rate limiting: protects against abuse and API cost overruns
- Idempotent ingestion: ChromaDB only re-ingests on empty collection, fast restarts

## Outcome
Built a production-ready portfolio assistant demonstrating end-to-end LLM system design: from vector ingestion and agentic routing to streaming API and containerized deployment. The project showcases practical skills in LangGraph, RAG pipelines, FastAPI async patterns, and MLOps (Docker, environment management).

## GitHub
github.com/abcdefya/Sirena-Assitant
