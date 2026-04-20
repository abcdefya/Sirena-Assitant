# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools needed to compile some native extensions (e.g. chroma, hnswlib)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to prevent pip from pulling the CUDA variant
# (~3 GB heavier). Since torch is already satisfied, installing
# sentence-transformers later won't upgrade it to the CUDA build.
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so container startup is instant
ENV HF_HOME=/hf-cache
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# libgomp1 — required at runtime by:
#   • chromadb_rust_bindings (OpenMP parallelism in the Rust extension)
#   • onnxruntime (used by sentence-transformers for CPU inference)
# gcc brings it into the builder stage but it is NOT included in slim by default,
# so native extensions that link against libgomp.so.1 crash on import without it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages (no gcc, no build artefacts)
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
# Copy CLI entry-points (uvicorn, etc.)
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy pre-downloaded HuggingFace model
COPY --from=builder /hf-cache /hf-cache

ENV HF_HOME=/hf-cache

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
