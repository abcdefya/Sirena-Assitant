from unittest.mock import MagicMock, patch


def test_get_vectorstore_returns_singleton():
    mock_store = MagicMock()
    with patch("src.services.vectorstore.Chroma", return_value=mock_store), \
         patch("src.services.vectorstore.get_embeddings", return_value=MagicMock()):
        from src.services import vectorstore as vs_module
        vs_module._vectorstore = None
        result1 = vs_module.get_vectorstore()
        result2 = vs_module.get_vectorstore()
        assert result1 is result2


def test_ingest_documents_skips_unchanged_file(tmp_path):
    content = "# Test\nThis is a test document."
    md_file = tmp_path / "test.md"
    md_file.write_text(content)

    import hashlib
    file_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
    source = str(md_file)
    existing_chunk_id = f"{source}::{file_hash}::0"

    mock_store = MagicMock()
    mock_store.get.return_value = {
        "ids": [existing_chunk_id],
        "metadatas": [{"source": source}],
    }

    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store), \
         patch("src.services.vectorstore.DATA_DIR", tmp_path):
        from src.services.vectorstore import ingest_documents
        ingest_documents()
        mock_store.add_documents.assert_not_called()
        mock_store.delete.assert_not_called()


def test_ingest_documents_ingests_new_file(tmp_path):
    md_file = tmp_path / "new.md"
    md_file.write_text("# New\nBrand new document.")

    mock_store = MagicMock()
    mock_store.get.return_value = {"ids": [], "metadatas": []}

    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store), \
         patch("src.services.vectorstore.DATA_DIR", tmp_path):
        from src.services.vectorstore import ingest_documents
        ingest_documents()
        mock_store.add_documents.assert_called_once()


def test_ingest_documents_updates_changed_file(tmp_path):
    md_file = tmp_path / "changed.md"
    md_file.write_text("# Updated\nNew content after change.")

    source = str(md_file)
    stale_chunk_id = f"{source}::oldhash::0"

    mock_store = MagicMock()
    mock_store.get.return_value = {
        "ids": [stale_chunk_id],
        "metadatas": [{"source": source}],
    }

    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store), \
         patch("src.services.vectorstore.DATA_DIR", tmp_path):
        from src.services.vectorstore import ingest_documents
        ingest_documents()
        mock_store.delete.assert_called_once_with(ids=[stale_chunk_id])
        mock_store.add_documents.assert_called_once()


def test_ingest_documents_removes_deleted_file(tmp_path):
    # File exists in DB but not on disk
    source = str(tmp_path / "deleted.md")
    stale_id = f"{source}::somehash::0"

    mock_store = MagicMock()
    mock_store.get.return_value = {
        "ids": [stale_id],
        "metadatas": [{"source": source}],
    }

    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store), \
         patch("src.services.vectorstore.DATA_DIR", tmp_path):
        from src.services.vectorstore import ingest_documents
        ingest_documents()
        mock_store.delete.assert_called_once_with(ids=[stale_id])
        mock_store.add_documents.assert_not_called()


def test_get_retriever_uses_vectorstore():
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_store.as_retriever.return_value = mock_retriever

    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store):
        from src.services.vectorstore import get_retriever
        result = get_retriever(k=3)
        mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert result is mock_retriever
