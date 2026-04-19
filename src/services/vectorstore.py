from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
