import os

# Set env vars before any src.* imports so config module loads them correctly
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GROQ_MODEL", "llama-3.3-70b-versatile")
os.environ.setdefault("HF_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./test_chroma_db")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "sirena_test")
os.environ.setdefault("DATA_DIR", "data")
os.environ.setdefault("LOG_LEVEL", "DEBUG")
