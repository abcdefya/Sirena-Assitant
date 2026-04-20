from unittest.mock import MagicMock, patch


def test_get_embeddings_returns_singleton():
    mock_embeddings = MagicMock()
    with patch("src.services.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings):
        from src.services import embeddings as emb_module
        emb_module._embeddings = None  # reset singleton
        result1 = emb_module.get_embeddings()
        result2 = emb_module.get_embeddings()
        assert result1 is result2

def test_get_embeddings_uses_configured_model():
    mock_embeddings = MagicMock()
    with patch("src.services.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings) as mock_cls:
        from src.services import embeddings as emb_module
        from src.cores.config import HF_MODEL
        emb_module._embeddings = None
        emb_module.get_embeddings()
        mock_cls.assert_called_once_with(model_name=HF_MODEL)
