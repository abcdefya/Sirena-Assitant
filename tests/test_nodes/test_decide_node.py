from unittest.mock import MagicMock, patch


def make_state(**kwargs):
    return {
        "session_id": "test-session",
        "question": kwargs.get("question", ""),
        "chat_history": [],
        "documents": [],
        "answer": "",
        "needs_retrieval": False,
        "docs_relevant": False,
        **kwargs,
    }


def test_decide_node_sets_needs_retrieval_true_when_llm_says_yes():
    mock_response = MagicMock()
    mock_response.content = "yes"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.decide_node.get_llm", return_value=mock_llm):
        from src.nodes.decide_node import decide_node
        state = make_state(question="What projects have you built with React?")
        result = decide_node(state)
        assert result["needs_retrieval"] is True


def test_decide_node_sets_needs_retrieval_false_when_llm_says_no():
    mock_response = MagicMock()
    mock_response.content = "no"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.decide_node.get_llm", return_value=mock_llm):
        from src.nodes.decide_node import decide_node
        state = make_state(question="Hello!")
        result = decide_node(state)
        assert result["needs_retrieval"] is False


def test_decide_node_preserves_other_state_fields():
    mock_response = MagicMock()
    mock_response.content = "yes"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("src.nodes.decide_node.get_llm", return_value=mock_llm):
        from src.nodes.decide_node import decide_node
        state = make_state(question="Tell me about your skills", answer="existing")
        result = decide_node(state)
        assert result["answer"] == "existing"
        assert result["session_id"] == "test-session"
