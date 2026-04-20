from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


def test_chat_returns_streaming_response():
    async def mock_stream_answer(state):
        yield "Hello"
        yield " world"

    mock_state = {
        "session_id": "test-123",
        "question": "Hi",
        "chat_history": [],
        "documents": [],
        "answer": "",
        "needs_retrieval": False,
        "docs_relevant": False,
    }

    with patch("src.services.vectorstore.ingest_documents"), \
         patch("src.api.routes.chat.get_pipeline") as mock_pipeline, \
         patch("src.api.routes.chat.stream_answer", side_effect=mock_stream_answer):
        mock_app_instance = MagicMock()
        mock_app_instance.ainvoke = AsyncMock(return_value=mock_state)
        mock_pipeline.return_value = mock_app_instance

        from src.api.main import app
        client = TestClient(app)
        response = client.post("/chat", json={"session_id": "test-123", "message": "Hi"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


def test_chat_rejects_empty_message():
    with patch("src.services.vectorstore.ingest_documents"):
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/chat", json={"session_id": "test-123", "message": ""})
        assert response.status_code == 422


def test_delete_session_returns_cleared():
    with patch("src.services.vectorstore.ingest_documents"):
        from src.api.main import app
        client = TestClient(app)
        response = client.delete("/chat/my-session-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        assert data["session_id"] == "my-session-id"
