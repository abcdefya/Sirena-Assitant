from unittest.mock import MagicMock, patch


def test_get_llm_returns_singleton():
    mock_llm = MagicMock()
    with patch("src.services.groq_llm.ChatGroq", return_value=mock_llm):
        from src.services import groq_llm as llm_module
        llm_module._llm = None
        result1 = llm_module.get_llm()
        result2 = llm_module.get_llm()
        assert result1 is result2

def test_get_llm_uses_configured_model():
    mock_llm = MagicMock()
    with patch("src.services.groq_llm.ChatGroq", return_value=mock_llm) as mock_cls:
        from src.services import groq_llm as llm_module
        from src.cores.config import GROQ_API_KEY, GROQ_MODEL
        llm_module._llm = None
        llm_module.get_llm()
        mock_cls.assert_called_once_with(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
        )
