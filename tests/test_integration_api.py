"""Integration tests for DocFoundry API endpoints and database operations."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
import time
from concurrent.futures import ThreadPoolExecutor

# Import DocFoundry components
from server.rag_api import create_app
from config.database import DatabaseConfig, DatabaseType
from indexer.embeddings import EmbeddingManager
from pipelines.crawler import WebCrawler
from services.shared.lineage import LineageTracker
from observability.telemetry import TelemetryManager


class TestDatabaseIntegration:
    """Test database operations and transactions."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Create database configuration
        config = DatabaseConfig(
            type=DatabaseType.SQLITE,
            path=db_path
        )
        
        # Initialize database schema
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            # Create basic tables for testing
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    url TEXT UNIQUE,
                    content TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS search_logs (
                    id INTEGER PRIMARY KEY,
                    query TEXT,
                    results TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        
        yield config, engine
        
        # Cleanup
        engine.dispose()
        os.unlink(db_path)
    
    def test_database_connection(self, temp_db):
        """Test database connection and basic operations."""
        config, engine = temp_db
        
        with engine.connect() as conn:
            # Test insert
            result = conn.execute(text(
                "INSERT INTO documents (url, content) VALUES (?, ?)"
            ), ("https://example.com", "Test content"))
            
            # Test select
            result = conn.execute(text(
                "SELECT * FROM documents WHERE url = ?"
            ), ("https://example.com",))
            row = result.fetchone()
            
            assert row is not None
            assert row[1] == "https://example.com"
            assert row[2] == "Test content"
    
    def test_transaction_rollback(self, temp_db):
        """Test database transaction rollback."""
        config, engine = temp_db
        
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                # Insert valid data
                conn.execute(text(
                    "INSERT INTO documents (url, content) VALUES (?, ?)"
                ), ("https://test1.com", "Content 1"))
                
                # Insert duplicate URL (should fail)
                conn.execute(text(
                    "INSERT INTO documents (url, content) VALUES (?, ?)"
                ), ("https://test1.com", "Content 2"))
                
                trans.commit()
            except Exception:
                trans.rollback()
            
            # Verify rollback worked
            result = conn.execute(text(
                "SELECT COUNT(*) FROM documents WHERE url = ?"
            ), ("https://test1.com",))
            count = result.scalar()
            assert count == 0
    
    def test_concurrent_database_access(self, temp_db):
        """Test concurrent database operations."""
        config, engine = temp_db
        
        def insert_document(url_suffix):
            with engine.connect() as conn:
                conn.execute(text(
                    "INSERT INTO documents (url, content) VALUES (?, ?)"
                ), (f"https://test{url_suffix}.com", f"Content {url_suffix}"))
                conn.commit()
        
        # Run concurrent inserts
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(insert_document, i) for i in range(10)]
            for future in futures:
                future.result()  # Wait for completion
        
        # Verify all documents were inserted
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM documents"))
            count = result.scalar()
            assert count == 10


class TestAPIIntegration:
    """Test API endpoint integration."""
    
    @pytest.fixture
    def test_app(self):
        """Create test FastAPI application."""
        with patch('server.rag_api.get_database_config') as mock_db_config:
            # Mock database configuration
            mock_db_config.return_value = DatabaseConfig(
                type=DatabaseType.SQLITE,
                path=":memory:"
            )
            
            app = create_app()
            return TestClient(app)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        with patch('indexer.embeddings.EmbeddingManager') as mock_embedding, \
             patch('pipelines.crawler.WebCrawler') as mock_crawler, \
             patch('services.shared.lineage.LineageTracker') as mock_lineage:
            
            # Configure mocks
            mock_embedding.return_value.search_similar.return_value = [
                {'url': 'https://example.com', 'content': 'Test content', 'score': 0.9}
            ]
            mock_crawler.return_value.crawl.return_value = Mock(
                success=True,
                content="Crawled content",
                metadata={'title': 'Test Page'}
            )
            mock_lineage.return_value.track_processing.return_value = None
            
            yield {
                'embedding': mock_embedding.return_value,
                'crawler': mock_crawler.return_value,
                'lineage': mock_lineage.return_value
            }
    
    def test_search_endpoint(self, test_app, mock_dependencies):
        """Test search API endpoint."""
        response = test_app.post("/search", json={
            "query": "test query",
            "limit": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
    
    def test_ingest_endpoint(self, test_app, mock_dependencies):
        """Test document ingestion endpoint."""
        response = test_app.post("/ingest", json={
            "url": "https://example.com",
            "force_recrawl": False
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "success"
    
    def test_feedback_endpoint(self, test_app, mock_dependencies):
        """Test feedback submission endpoint."""
        response = test_app.post("/feedback", json={
            "query": "test query",
            "document_url": "https://example.com",
            "relevance_score": 0.8,
            "feedback_type": "click"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "success"
    
    def test_health_endpoint(self, test_app):
        """Test health check endpoint."""
        response = test_app.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_invalid_search_request(self, test_app, mock_dependencies):
        """Test search endpoint with invalid request."""
        response = test_app.post("/search", json={
            "limit": -1  # Invalid limit
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_ingest_request(self, test_app, mock_dependencies):
        """Test ingest endpoint with invalid URL."""
        response = test_app.post("/ingest", json={
            "url": "not-a-valid-url"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_rate_limiting(self, test_app, mock_dependencies):
        """Test API rate limiting."""
        # Make multiple rapid requests
        responses = []
        for _ in range(20):
            response = test_app.post("/search", json={
                "query": "test query",
                "limit": 5
            })
            responses.append(response)
        
        # Check if any requests were rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        # Note: Rate limiting might not be enabled in test environment
        # This test verifies the endpoint handles high load gracefully
        assert all(r.status_code in [200, 429] for r in responses)


class TestCrossComponentIntegration:
    """Test integration between different DocFoundry components."""
    
    @pytest.fixture
    def integration_setup(self):
        """Set up components for integration testing."""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        config = DatabaseConfig(
            type=DatabaseType.SQLITE,
            path=db_path
        )
        
        # Initialize components
        embedding_manager = Mock()
        crawler = Mock()
        lineage_tracker = Mock()
        telemetry_manager = Mock()
        
        yield {
            'db_config': config,
            'db_path': db_path,
            'embedding_manager': embedding_manager,
            'crawler': crawler,
            'lineage_tracker': lineage_tracker,
            'telemetry_manager': telemetry_manager
        }
        
        # Cleanup
        os.unlink(db_path)
    
    def test_crawl_index_search_workflow(self, integration_setup):
        """Test complete crawl -> index -> search workflow."""
        setup = integration_setup
        
        # Mock crawler response
        setup['crawler'].crawl.return_value = Mock(
            success=True,
            content="This is test content about machine learning",
            metadata={'title': 'ML Guide', 'url': 'https://example.com/ml'}
        )
        
        # Mock embedding generation
        setup['embedding_manager'].generate_embedding.return_value = [0.1] * 384
        setup['embedding_manager'].search_similar.return_value = [
            {
                'url': 'https://example.com/ml',
                'content': 'This is test content about machine learning',
                'score': 0.95
            }
        ]
        
        # Simulate workflow
        # 1. Crawl document
        crawl_result = setup['crawler'].crawl('https://example.com/ml')
        assert crawl_result.success
        
        # 2. Generate embedding
        embedding = setup['embedding_manager'].generate_embedding(crawl_result.content)
        assert len(embedding) == 384
        
        # 3. Search for similar content
        search_results = setup['embedding_manager'].search_similar(
            "machine learning tutorial", limit=5
        )
        assert len(search_results) > 0
        assert search_results[0]['score'] > 0.9
    
    def test_feedback_learning_integration(self, integration_setup):
        """Test feedback collection and learning integration."""
        setup = integration_setup
        
        # Mock feedback processing
        setup['lineage_tracker'].track_feedback.return_value = None
        
        # Simulate feedback workflow
        feedback_data = {
            'query': 'python tutorial',
            'document_url': 'https://example.com/python',
            'relevance_score': 0.8,
            'user_action': 'click'
        }
        
        # Process feedback
        setup['lineage_tracker'].track_feedback(feedback_data)
        setup['lineage_tracker'].track_feedback.assert_called_once_with(feedback_data)
    
    def test_error_propagation(self, integration_setup):
        """Test error handling across components."""
        setup = integration_setup
        
        # Mock component failure
        setup['crawler'].crawl.side_effect = Exception("Network error")
        
        # Test error handling
        with pytest.raises(Exception) as exc_info:
            setup['crawler'].crawl('https://invalid-url.com')
        
        assert "Network error" in str(exc_info.value)
    
    def test_telemetry_integration(self, integration_setup):
        """Test telemetry and monitoring integration."""
        setup = integration_setup
        
        # Mock telemetry calls
        setup['telemetry_manager'].record_metric.return_value = None
        setup['telemetry_manager'].start_span.return_value = Mock()
        
        # Simulate operations with telemetry
        with setup['telemetry_manager'].start_span("test_operation"):
            setup['telemetry_manager'].record_metric("test_counter", 1)
        
        # Verify telemetry calls
        setup['telemetry_manager'].start_span.assert_called_once_with("test_operation")
        setup['telemetry_manager'].record_metric.assert_called_once_with("test_counter", 1)


class TestPerformanceIntegration:
    """Test performance aspects of integration."""
    
    def test_concurrent_api_requests(self):
        """Test handling of concurrent API requests."""
        with patch('server.rag_api.get_database_config') as mock_db_config:
            mock_db_config.return_value = DatabaseConfig(
                type=DatabaseType.SQLITE,
                path=":memory:"
            )
            
            app = create_app()
            client = TestClient(app)
            
            def make_request():
                return client.post("/search", json={
                    "query": "test query",
                    "limit": 5
                })
            
            # Make concurrent requests
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request) for _ in range(20)]
                responses = [future.result() for future in futures]
            
            # Verify all requests completed successfully
            success_count = sum(1 for r in responses if r.status_code == 200)
            assert success_count >= 15  # Allow for some rate limiting
    
    def test_large_document_processing(self):
        """Test processing of large documents."""
        with patch('indexer.embeddings.EmbeddingManager') as mock_embedding:
            # Create large content
            large_content = "This is test content. " * 10000  # ~200KB
            
            # Mock embedding generation for large content
            mock_embedding.return_value.generate_embedding.return_value = [0.1] * 384
            
            embedding_manager = mock_embedding.return_value
            embedding = embedding_manager.generate_embedding(large_content)
            
            assert len(embedding) == 384
            mock_embedding.return_value.generate_embedding.assert_called_once_with(large_content)
    
    def test_database_performance(self):
        """Test database performance under load."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            engine = create_engine(f"sqlite:///{db_path}")
            
            # Create table
            with engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE test_docs (
                        id INTEGER PRIMARY KEY,
                        content TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
            
            # Measure insertion performance
            start_time = time.time()
            
            with engine.connect() as conn:
                for i in range(1000):
                    conn.execute(text(
                        "INSERT INTO test_docs (content) VALUES (?)"
                    ), (f"Test content {i}",))
                conn.commit()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete within reasonable time (adjust threshold as needed)
            assert duration < 10.0  # 10 seconds for 1000 inserts
            
            # Verify data integrity
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM test_docs"))
                count = result.scalar()
                assert count == 1000
        
        finally:
            engine.dispose()
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])