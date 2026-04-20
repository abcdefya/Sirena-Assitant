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
