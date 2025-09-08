import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Mock the monitoring module to avoid event loop issues
with patch('server.monitoring.MetricsCollector'):
    from server.rag_api import app

client = TestClient(app)

def test_search_endpoint_exists():
    """Test that the search endpoint is accessible."""
    # Test with a basic query
    response = client.post("/search", json={"query": "test"})
    # Should not return 404 (endpoint exists)
    assert response.status_code != 404

def test_search_with_empty_query():
    """Test search behavior with empty query."""
    response = client.post("/search", json={"query": ""})
    # Should handle empty queries gracefully
    assert response.status_code in [200, 400]  # Either OK or bad request

def test_search_response_format():
    """Test that search returns proper JSON format."""
    response = client.post("/search", json={"query": "test query"})
    if response.status_code == 200:
        data = response.json()
        # Basic structure validation
        assert isinstance(data, (dict, list))

def test_search_invalid_json():
    """Test search with invalid JSON payload."""
    response = client.post("/search", data="invalid json")
    assert response.status_code == 422  # Unprocessable Entity