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
