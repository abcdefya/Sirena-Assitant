import hashlib
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.cores.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, DATA_DIR
from src.services.embeddings import get_embeddings
from src.cores.logger import get_logger

logger = get_logger(__name__)

_vectorstore: Chroma | None = None


def _compute_file_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _make_chunk_id(source: str, file_hash: str, chunk_idx: int) -> str:
    return f"{source}::{file_hash}::{chunk_idx}"


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

    # Get all existing IDs and metadata from ChromaDB
    existing = vectorstore.get(include=["metadatas"])
    existing_ids: list[str] = existing["ids"]
    existing_metadatas: list[dict] = existing["metadatas"]

    # Group existing chunk IDs by source path
    existing_by_source: dict[str, set[str]] = {}
    for chunk_id, meta in zip(existing_ids, existing_metadatas):
        source = meta.get("source", "")
        existing_by_source.setdefault(source, set()).add(chunk_id)

    # Discover current markdown files on disk
    current_files: dict[str, str] = {}
    for md_file in Path(DATA_DIR).rglob("*.md"):
        source = str(md_file)
        current_files[source] = md_file.read_text(encoding="utf-8")

    # Remove chunks for files that no longer exist on disk
    current_sources = set(current_files.keys())
    for source, chunk_ids in existing_by_source.items():
        if source not in current_sources:
            vectorstore.delete(ids=list(chunk_ids))
            logger.info(f"Deleted {len(chunk_ids)} chunks for removed file: {source}")

    if not current_files:
        logger.warning("No markdown files found in data/ — skipping ingestion")
        return

    # Add or update chunks for each current file
    total_added = 0
    for source, content in current_files.items():
        file_hash = _compute_file_hash(content)
        source_ids = existing_by_source.get(source, set())

        if source_ids:
            hash_prefix = f"{source}::{file_hash}::"
            if any(chunk_id.startswith(hash_prefix) for chunk_id in source_ids):
                logger.debug(f"File unchanged, skipping: {source}")
                continue
            # File content changed — remove stale chunks before re-indexing
            vectorstore.delete(ids=list(source_ids))
            logger.info(f"Deleted {len(source_ids)} stale chunks for changed file: {source}")

        doc = Document(page_content=content, metadata={"source": source})
        chunks = _chunk_documents([doc])
        ids = [_make_chunk_id(source, file_hash, i) for i in range(len(chunks))]
        vectorstore.add_documents(chunks, ids=ids)
        total_added += len(chunks)
        logger.info(f"Ingested {len(chunks)} chunks for: {source}")

    if total_added:
        logger.info(f"Total: ingested {total_added} new/updated chunks")
    else:
        logger.info("All files up to date — no ingestion needed")


def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
