import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Mock the monitoring module to avoid event loop issues
with patch('server.monitoring.MetricsCollector'):
    from server.rag_api import app

client = TestClient(app)

def test_health():
    """Test the health endpoint returns OK status."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

def test_health_response_format():
    """Test that health endpoint returns proper JSON format."""
    r = client.get("/health")
    assert r.status_code == 200
    response_data = r.json()
    assert "status" in response_data
    assert isinstance(response_data["status"], str)