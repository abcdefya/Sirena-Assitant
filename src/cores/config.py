import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _parse_int(env_var: str, default: int) -> int:
    """Parse an integer from an environment variable with a descriptive error."""
    value = os.getenv(env_var, str(default))
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Environment variable {env_var}={value!r} must be an integer") from None


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HF_MODEL = os.getenv("HF_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "sirena_portfolio")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RATE_LIMIT_PER_MINUTE = _parse_int("RATE_LIMIT_PER_MINUTE", 10)
RATE_LIMIT_PER_HOUR = _parse_int("RATE_LIMIT_PER_HOUR", 50)
