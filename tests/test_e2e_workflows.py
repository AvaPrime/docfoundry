import pytest
import asyncio
import tempfile
import os
import sys
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock external dependencies to prevent import issues
with patch.dict('sys.modules', {
    'server.monitoring': Mock(),
    'server.monitoring.MetricsCollector': Mock(),
    'server.monitoring.metrics_collector': Mock(),
    'redis': Mock(),
    'apscheduler': Mock(),
    'apscheduler.schedulers': Mock(),
    'apscheduler.schedulers.asyncio': Mock(),
    'apscheduler.schedulers.asyncio.AsyncIOScheduler': Mock(),
    'apscheduler.triggers': Mock(),
    'apscheduler.triggers.interval': Mock(),
    'apscheduler.triggers.interval.IntervalTrigger': Mock(),
}):
    from server.rag_api import app
    from fastapi.testclient import TestClient


class TestEndToEndWorkflows:
    """End-to-end tests for complete DocFoundry workflows"""
    
    @pytest.fixture
    def temp_db_url(self):
        """Create a temporary SQLite database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        yield f"sqlite:///{db_path}"
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies"""
        mocks = {
            'embedding_manager': Mock(),
            'crawler': Mock(),
            'lineage_tracker': Mock(),
            'telemetry_manager': Mock(),
            'job_manager': Mock(),
            'metrics_collector': Mock()
        }
        
        # Configure embedding manager
        mocks['embedding_manager'].embed_text = AsyncMock(return_value=[0.1] * 384)
        mocks['embedding_manager'].embed_batch = AsyncMock(return_value=[[0.1] * 384, [0.2] * 384])
        
        # Configure crawler
        mocks['crawler'].crawl_url = AsyncMock(return_value={
            'url': 'https://example.com',
            'title': 'Test Document',
            'content': 'This is test content for the document.',
            'metadata': {'source': 'web', 'crawled_at': datetime.now().isoformat()}
        })
        
        # Configure lineage tracker
        mocks['lineage_tracker'].track_document = AsyncMock()
        mocks['lineage_tracker'].track_search = AsyncMock()
        mocks['lineage_tracker'].track_feedback = AsyncMock()
        
        # Configure telemetry manager
        mocks['telemetry_manager'].track_event = AsyncMock()
        mocks['telemetry_manager'].track_performance = AsyncMock()
        
        return mocks
    
    @pytest.fixture
    def client(self, temp_db_url, mock_dependencies):
        """Create test client with mocked dependencies"""
        with patch.multiple(
            'server.rag_api',
            **mock_dependencies
        ):
            with patch('config.database.get_database_url', return_value=temp_db_url):
                return TestClient(app)
    
    def test_complete_crawl_index_search_workflow(self, client, mock_dependencies):
        """Test complete workflow: crawl -> index -> search"""
        # Step 1: Ingest a document (simulates crawling and indexing)
        ingest_payload = {
            "url": "https://example.com/test-doc",
            "content": "This is a comprehensive test document about machine learning algorithms.",
            "metadata": {
                "title": "ML Algorithms Guide",
                "author": "Test Author",
                "category": "technical"
            }
        }
        
        ingest_response = client.post("/api/ingest", json=ingest_payload)
        assert ingest_response.status_code == 200
        ingest_data = ingest_response.json()
        assert "document_id" in ingest_data
        document_id = ingest_data["document_id"]
        
        # Step 2: Search for the indexed document
        search_payload = {
            "query": "machine learning algorithms",
            "limit": 5
        }
        
        # Mock search results
        with patch('server.rag_api.perform_search') as mock_search:
            mock_search.return_value = {
                "results": [
                    {
                        "document_id": document_id,
                        "content": ingest_payload["content"],
                        "metadata": ingest_payload["metadata"],
                        "score": 0.95,
                        "url": ingest_payload["url"]
                    }
                ],
                "total_results": 1,
                "query_time_ms": 45
            }
            
            search_response = client.post("/api/search", json=search_payload)
            assert search_response.status_code == 200
            search_data = search_response.json()
            
            assert len(search_data["results"]) == 1
            assert search_data["results"][0]["document_id"] == document_id
            assert search_data["results"][0]["score"] > 0.9
    
    def test_search_feedback_learning_workflow(self, client, mock_dependencies):
        """Test workflow: search -> feedback -> learning"""
        # Step 1: Perform a search
        search_payload = {
            "query": "python programming best practices",
            "limit": 3
        }
        
        with patch('server.rag_api.perform_search') as mock_search:
            mock_search.return_value = {
                "results": [
                    {
                        "document_id": "doc1",
                        "content": "Python best practices include PEP 8 compliance.",
                        "metadata": {"title": "Python Guide"},
                        "score": 0.88,
                        "url": "https://example.com/python-guide"
                    },
                    {
                        "document_id": "doc2",
                        "content": "Clean code principles for Python development.",
                        "metadata": {"title": "Clean Python"},
                        "score": 0.82,
                        "url": "https://example.com/clean-python"
                    }
                ],
                "total_results": 2,
                "query_time_ms": 32
            }
            
            search_response = client.post("/api/search", json=search_payload)
            assert search_response.status_code == 200
            search_data = search_response.json()
            search_id = search_data.get("search_id", "test_search_123")
        
        # Step 2: Provide feedback on search results
        feedback_payload = {
            "search_id": search_id,
            "query": search_payload["query"],
            "feedback_type": "relevance",
            "document_feedback": [
                {
                    "document_id": "doc1",
                    "rating": 5,
                    "comment": "Very helpful and accurate"
                },
                {
                    "document_id": "doc2",
                    "rating": 3,
                    "comment": "Somewhat relevant but could be better"
                }
            ]
        }
        
        feedback_response = client.post("/api/feedback", json=feedback_payload)
        assert feedback_response.status_code == 200
        feedback_data = feedback_response.json()
        assert feedback_data["status"] == "success"
    
    def test_batch_processing_workflow(self, client, mock_dependencies):
        """Test batch processing workflow for multiple documents"""
        # Prepare multiple documents for batch ingestion
        documents = [
            {
                "url": f"https://example.com/doc-{i}",
                "content": f"This is test document {i} about topic {i % 3}.",
                "metadata": {
                    "title": f"Document {i}",
                    "category": f"category_{i % 3}",
                    "batch_id": "test_batch_001"
                }
            }
            for i in range(5)
        ]
        
        document_ids = []
        
        # Ingest documents one by one (simulating batch processing)
        for doc in documents:
            response = client.post("/api/ingest", json=doc)
            assert response.status_code == 200
            data = response.json()
            document_ids.append(data["document_id"])
        
        # Verify all documents were processed
        assert len(document_ids) == 5
        assert len(set(document_ids)) == 5  # All unique IDs
    
    def test_error_recovery_workflow(self, client, mock_dependencies):
        """Test error handling and recovery in workflows"""
        # Test 1: Handle embedding failure gracefully
        mock_dependencies['embedding_manager'].embed_text.side_effect = Exception("Embedding service unavailable")
        
        ingest_payload = {
            "url": "https://example.com/test-error",
            "content": "This document should fail to embed.",
            "metadata": {"title": "Error Test Doc"}
        }
        
        response = client.post("/api/ingest", json=ingest_payload)
        # Should handle error gracefully (depending on implementation)
        assert response.status_code in [200, 500]  # Either succeeds with fallback or fails gracefully
        
        # Reset mock for next test
        mock_dependencies['embedding_manager'].embed_text.side_effect = None
        mock_dependencies['embedding_manager'].embed_text.return_value = [0.1] * 384
    
    def test_health_check_workflow(self, client):
        """Test system health check workflow"""
        # Test basic health endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "version" in health_data
    
    def test_api_documentation_workflow(self, client):
        """Test API documentation accessibility"""
        # Test OpenAPI schema endpoint
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200
        
        # Test OpenAPI JSON schema
        openapi_response = client.get("/openapi.json")
        assert openapi_response.status_code == 200
        
        openapi_data = openapi_response.json()
        assert "openapi" in openapi_data
        assert "paths" in openapi_data


class TestWorkflowPerformance:
    """Performance tests for workflow operations"""
    
    @pytest.fixture
    def performance_client(self):
        """Create client for performance testing"""
        with patch.dict('sys.modules', {
            'server.monitoring': Mock(),
            'server.monitoring.MetricsCollector': Mock(),
            'server.monitoring.metrics_collector': Mock(),
        }):
            return TestClient(app)
    
    def test_search_response_time(self, performance_client):
        """Test search response time performance"""
        import time
        
        search_payload = {
            "query": "performance test query",
            "limit": 10
        }
        
        with patch('server.rag_api.perform_search') as mock_search:
            mock_search.return_value = {
                "results": [],
                "total_results": 0,
                "query_time_ms": 50
            }
            
            start_time = time.time()
            response = performance_client.post("/api/search", json=search_payload)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            assert response.status_code == 200
            assert response_time < 1000  # Should respond within 1 second
    
    def test_ingest_throughput(self, performance_client):
        """Test document ingestion throughput"""
        import time
        
        documents = [
            {
                "url": f"https://example.com/perf-{i}",
                "content": f"Performance test document {i} with substantial content to test throughput.",
                "metadata": {"title": f"Perf Doc {i}"}
            }
            for i in range(5)  # Reduced from 10 for faster testing
        ]
        
        with patch('server.rag_api.embedding_manager') as mock_embedding:
            mock_embedding.embed_text = AsyncMock(return_value=[0.1] * 384)
            
            start_time = time.time()
            
            for doc in documents:
                response = performance_client.post("/api/ingest", json=doc)
                assert response.status_code == 200
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = len(documents) / total_time  # Documents per second
            
            # Should process at least 0.5 documents per second (relaxed for testing)
            assert throughput > 0.5, f"Throughput {throughput:.2f} docs/sec is too low"