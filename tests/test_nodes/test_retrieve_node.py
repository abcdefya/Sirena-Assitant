from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def make_state(**kwargs):
    return {
        "session_id": "test-session",
        "question": kwargs.get("question", "What are your projects?"),
        "chat_history": [],
        "documents": [],
        "answer": "",
        "needs_retrieval": True,
        "docs_relevant": False,
        **kwargs,
    }


def test_retrieve_node_populates_documents():
    mock_docs = [
        Document(page_content="Project A: React dashboard"),
        Document(page_content="Project B: FastAPI backend"),
    ]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = mock_docs

    with patch("src.nodes.retrieve_node.get_retriever", return_value=mock_retriever):
        from src.nodes.retrieve_node import retrieve_node
        state = make_state(question="What projects have you built?")
        result = retrieve_node(state)
        assert result["documents"] == mock_docs
        assert len(result["documents"]) == 2


def test_retrieve_node_calls_retriever_with_question():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    with patch("src.nodes.retrieve_node.get_retriever", return_value=mock_retriever):
        from src.nodes.retrieve_node import retrieve_node
        state = make_state(question="Tell me about your React experience")
        retrieve_node(state)
        mock_retriever.invoke.assert_called_once_with("Tell me about your React experience")
