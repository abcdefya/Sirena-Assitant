from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def make_state(**kwargs):
    return {
        "session_id": "test-session",
        "question": kwargs.get("question", "What projects have you built?"),
        "chat_history": [],
        "documents": kwargs.get("documents", []),
        "answer": "",
        "needs_retrieval": True,
        "docs_relevant": False,
        **kwargs,
    }


def test_grade_node_sets_docs_relevant_true_when_llm_says_yes():
    mock_response = MagicMock()
    mock_response.content = "yes"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.grade_node.get_llm", return_value=mock_llm):
        from src.nodes.grade_node import grade_node
        docs = [Document(page_content="React dashboard project built in 2024")]
        state = make_state(question="What React projects have you built?", documents=docs)
        result = grade_node(state)
        assert result["docs_relevant"] is True


def test_grade_node_sets_docs_relevant_false_when_llm_says_no():
    mock_response = MagicMock()
    mock_response.content = "no"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.grade_node.get_llm", return_value=mock_llm):
        from src.nodes.grade_node import grade_node
        docs = [Document(page_content="Unrelated content about cooking")]
        state = make_state(question="What is your tech stack?", documents=docs)
        result = grade_node(state)
        assert result["docs_relevant"] is False


def test_grade_node_preserves_documents():
    mock_response = MagicMock()
    mock_response.content = "yes"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.grade_node.get_llm", return_value=mock_llm):
        from src.nodes.grade_node import grade_node
        docs = [Document(page_content="My portfolio project")]
        state = make_state(documents=docs)
        result = grade_node(state)
        assert result["documents"] == docs
