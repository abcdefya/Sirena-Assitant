from langchain_huggingface import HuggingFaceEmbeddings
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
