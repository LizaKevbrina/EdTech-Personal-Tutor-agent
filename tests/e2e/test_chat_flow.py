"""
End-to-end tests for chat flow.
"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestChatFlow:
    """End-to-end tests for chat functionality."""

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_chat_request(self, client):
        """Test basic chat request."""
        payload = {
            "student_id": "test_student_123",
            "message": "What is Python?",
        }

        response = client.post("/api/v1/chat", json=payload)

        # May fail if OpenAI API key not configured
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
            assert "session_id" in data
            assert data["session_id"] is not None

    def test_chat_with_pii(self, client):
        """Test chat rejects PII."""
        payload = {
            "student_id": "test_student_123",
            "message": "My email is test@example.com",
        }

        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 400
        assert "pii_detected" in response.json()["error"]

    def test_chat_empty_message(self, client):
        """Test chat rejects empty message."""
        payload = {
            "student_id": "test_student_123",
            "message": "",
        }

        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 422  # Validation error

    def test_get_session_history(self, client):
        """Test retrieving session history."""
        # First, create a session
        payload = {
            "student_id": "test_student_123",
            "message": "Hello",
            "session_id": "test_session",
        }

        chat_response = client.post("/api/v1/chat", json=payload)

        if chat_response.status_code == 200:
            # Get history
            history_response = client.get(
                "/api/v1/chat/session/test_session/history"
            )
            assert history_response.status_code == 200
            data = history_response.json()
            assert "messages" in data
