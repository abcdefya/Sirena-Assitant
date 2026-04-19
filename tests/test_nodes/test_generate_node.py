import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def make_state(**kwargs):
    return {
        "session_id": "test-session",
        "question": kwargs.get("question", "What are your skills?"),
        "chat_history": kwargs.get("chat_history", []),
        "documents": kwargs.get("documents", []),
        "answer": "",
        "needs_retrieval": False,
        "docs_relevant": kwargs.get("docs_relevant", False),
        **kwargs,
    }


@pytest.mark.asyncio
async def test_stream_answer_yields_tokens_when_docs_relevant():
    chunks = [MagicMock(content="I "), MagicMock(content="know "), MagicMock(content="Python")]

    async def mock_astream(messages):
        for chunk in chunks:
            yield chunk

    mock_llm = MagicMock()
    mock_llm.astream = mock_astream

    with patch("src.nodes.generate_node.get_llm", return_value=mock_llm):
        from src.nodes.generate_node import stream_answer
        docs = [Document(page_content="Python expert with 5 years experience")]
        state = make_state(docs_relevant=True, documents=docs)
        tokens = []
        async for token in stream_answer(state):
            tokens.append(token)
        assert tokens == ["I ", "know ", "Python"]


@pytest.mark.asyncio
async def test_stream_answer_skips_context_when_docs_not_relevant():
    chunks = [MagicMock(content="Hello")]

    async def mock_astream(messages):
        for chunk in chunks:
            yield chunk

    mock_llm = MagicMock()
    mock_llm.astream = mock_astream

    with patch("src.nodes.generate_node.get_llm", return_value=mock_llm):
        from src.nodes.generate_node import stream_answer
        docs = [Document(page_content="Irrelevant content")]
        state = make_state(docs_relevant=False, documents=docs)
        tokens = []
        async for token in stream_answer(state):
            tokens.append(token)
        assert len(tokens) > 0


@pytest.mark.asyncio
async def test_stream_answer_includes_chat_history_in_messages():
    captured_messages = []

    async def mock_astream(messages):
        captured_messages.extend(messages)
        yield MagicMock(content="answer")

    mock_llm = MagicMock()
    mock_llm.astream = mock_astream

    with patch("src.nodes.generate_node.get_llm", return_value=mock_llm):
        from src.nodes.generate_node import stream_answer
        history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        state = make_state(chat_history=history)
        async for _ in stream_answer(state):
            pass
        roles = [m["role"] for m in captured_messages]
        assert "user" in roles
        assert "assistant" in roles
