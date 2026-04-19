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


def test_ingest_documents_skips_if_collection_not_empty():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_store = MagicMock()
    mock_store._collection = mock_collection

    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store):
        from src.services import vectorstore as vs_module
        vs_module._vectorstore = mock_store
        from src.services.vectorstore import ingest_documents
        ingest_documents()
        mock_store.add_documents.assert_not_called()


def test_ingest_documents_ingests_when_collection_empty(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test\nThis is a test document.")

    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_store = MagicMock()
    mock_store._collection = mock_collection

    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store), \
         patch("src.services.vectorstore.DATA_DIR", tmp_path):
        from src.services.vectorstore import ingest_documents
        ingest_documents()
        mock_store.add_documents.assert_called_once()


def test_get_retriever_uses_vectorstore():
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_store.as_retriever.return_value = mock_retriever

    with patch("src.services.vectorstore.get_vectorstore", return_value=mock_store):
        from src.services.vectorstore import get_retriever
        result = get_retriever(k=3)
        mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert result is mock_retriever
