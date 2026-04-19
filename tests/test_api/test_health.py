from unittest.mock import patch
from fastapi.testclient import TestClient


def test_health_returns_ok():
    with patch("src.services.vectorstore.ingest_documents"):
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
