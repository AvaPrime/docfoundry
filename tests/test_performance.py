"""Performance tests for DocFoundry system.

This module contains comprehensive performance tests for:
- Search response times
- Indexing throughput
- Concurrent operations
- Memory and CPU usage
- Database performance
- API endpoint performance
"""

import asyncio
import time
import threading
import concurrent.futures
import psutil
import pytest
import sqlite3
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock external dependencies before imports
sys.modules['redis'] = Mock()
sys.modules['apscheduler'] = Mock()
sys.modules['apscheduler.schedulers'] = Mock()
sys.modules['apscheduler.schedulers.asyncio'] = Mock()
sys.modules['apscheduler.schedulers.asyncio.AsyncIOScheduler'] = Mock()
sys.modules['apscheduler.triggers'] = Mock()
sys.modules['apscheduler.triggers.interval'] = Mock()
sys.modules['apscheduler.triggers.interval.IntervalTrigger'] = Mock()
sys.modules['apscheduler.jobstores'] = Mock()
sys.modules['apscheduler.jobstores.redis'] = Mock()
sys.modules['apscheduler.jobstores.redis.RedisJobStore'] = Mock()
sys.modules['apscheduler.executors'] = Mock()
sys.modules['apscheduler.executors.pool'] = Mock()
sys.modules['apscheduler.executors.pool.ThreadPoolExecutor'] = Mock()
sys.modules['apscheduler.executors.asyncio'] = Mock()
sys.modules['apscheduler.executors.asyncio.AsyncIOExecutor'] = Mock()
sys.modules['apscheduler.events'] = Mock()
sys.modules['apscheduler.events.EVENT_JOB_EXECUTED'] = Mock()
sys.modules['apscheduler.events.EVENT_JOB_ERROR'] = Mock()
sys.modules['server.monitoring'] = Mock()

# Import after mocking
from server.rag_api import create_app
from indexer.embeddings import EmbeddingManager
from pipelines.crawler import Crawler
from services.shared.lineage import LineageTracker
from observability.telemetry import TelemetryManager


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.response_times = []
        self.throughput_data = []
        self.memory_usage = []
        self.cpu_usage = []
        self.start_time = None
        self.end_time = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.response_times.clear()
        self.throughput_data.clear()
        self.memory_usage.clear()
        self.cpu_usage.clear()
    
    def record_response_time(self, duration: float):
        """Record a response time measurement."""
        self.response_times.append(duration)
    
    def record_throughput(self, items_processed: int, duration: float):
        """Record throughput measurement."""
        throughput = items_processed / duration if duration > 0 else 0
        self.throughput_data.append(throughput)
    
    def record_system_metrics(self):
        """Record current system metrics."""
        process = psutil.Process()
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(process.cpu_percent())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        if self.response_times:
            stats['response_time'] = {
                'avg': sum(self.response_times) / len(self.response_times),
                'min': min(self.response_times),
                'max': max(self.response_times),
                'p95': sorted(self.response_times)[int(len(self.response_times) * 0.95)],
                'p99': sorted(self.response_times)[int(len(self.response_times) * 0.99)]
            }
        
        if self.throughput_data:
            stats['throughput'] = {
                'avg': sum(self.throughput_data) / len(self.throughput_data),
                'max': max(self.throughput_data)
            }
        
        if self.memory_usage:
            stats['memory_mb'] = {
                'avg': sum(self.memory_usage) / len(self.memory_usage),
                'max': max(self.memory_usage)
            }
        
        if self.cpu_usage:
            stats['cpu_percent'] = {
                'avg': sum(self.cpu_usage) / len(self.cpu_usage),
                'max': max(self.cpu_usage)
            }
        
        return stats


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Initialize database schema
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            url TEXT UNIQUE,
            title TEXT,
            content TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE search_logs (
            id INTEGER PRIMARY KEY,
            query TEXT,
            results_count INTEGER,
            response_time REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def performance_metrics():
    """Create a performance metrics collector."""
    return PerformanceMetrics()


@pytest.fixture
def mock_embedding_manager():
    """Create a mock embedding manager with realistic performance characteristics."""
    manager = Mock(spec=EmbeddingManager)
    
    def mock_embed(text):
        # Simulate embedding computation time
        time.sleep(0.01)  # 10ms per embedding
        return [0.1] * 384  # Mock 384-dimensional embedding
    
    manager.embed_text.side_effect = mock_embed
    manager.embed_batch.side_effect = lambda texts: [mock_embed(t) for t in texts]
    return manager


@pytest.fixture
def mock_crawler():
    """Create a mock crawler with realistic performance characteristics."""
    crawler = Mock(spec=Crawler)
    
    def mock_crawl(url):
        # Simulate crawling time
        time.sleep(0.05)  # 50ms per page
        return {
            'url': url,
            'title': f'Test Page {url}',
            'content': f'Content for {url}' * 100,  # Realistic content size
            'links': [f'{url}/page{i}' for i in range(5)]
        }
    
    crawler.crawl_page.side_effect = mock_crawl
    return crawler


class TestSearchPerformance:
    """Test search operation performance."""
    
    def test_search_response_time_single_query(self, temp_db, mock_embedding_manager, performance_metrics):
        """Test response time for single search queries."""
        # Setup test data
        conn = sqlite3.connect(temp_db)
        for i in range(1000):
            conn.execute(
                'INSERT INTO documents (url, title, content) VALUES (?, ?, ?)',
                (f'http://example.com/doc{i}', f'Document {i}', f'Content for document {i}' * 50)
            )
        conn.commit()
        conn.close()
        
        performance_metrics.start_monitoring()
        
        # Perform search queries
        queries = ['test query', 'document search', 'content analysis', 'information retrieval']
        
        for query in queries * 25:  # 100 total queries
            start_time = time.time()
            
            # Mock search operation
            with patch('sqlite3.connect') as mock_connect:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_cursor.fetchall.return_value = [(i, f'doc{i}', f'title{i}', 0.9) for i in range(10)]
                mock_conn.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn
                
                # Simulate search with embedding
                mock_embedding_manager.embed_text(query)
                
                # Simulate database query time
                time.sleep(0.002)  # 2ms database query
            
            end_time = time.time()
            performance_metrics.record_response_time(end_time - start_time)
            performance_metrics.record_system_metrics()
        
        stats = performance_metrics.get_statistics()
        
        # Performance assertions
        assert stats['response_time']['avg'] < 0.05, f"Average response time too high: {stats['response_time']['avg']}s"
        assert stats['response_time']['p95'] < 0.1, f"95th percentile response time too high: {stats['response_time']['p95']}s"
        assert stats['response_time']['p99'] < 0.2, f"99th percentile response time too high: {stats['response_time']['p99']}s"
    
    def test_search_concurrent_queries(self, temp_db, mock_embedding_manager, performance_metrics):
        """Test performance under concurrent search load."""
        performance_metrics.start_monitoring()
        
        def search_worker(query_id):
            """Worker function for concurrent searches."""
            start_time = time.time()
            
            # Simulate search operation
            mock_embedding_manager.embed_text(f'query {query_id}')
            time.sleep(0.01)  # Simulate processing time
            
            end_time = time.time()
            return end_time - start_time
        
        # Run concurrent searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(search_worker, i) for i in range(100)]
            response_times = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        for rt in response_times:
            performance_metrics.record_response_time(rt)
        
        stats = performance_metrics.get_statistics()
        
        # Concurrent performance assertions
        assert stats['response_time']['avg'] < 0.1, f"Concurrent average response time too high: {stats['response_time']['avg']}s"
        assert len(response_times) == 100, "Not all concurrent queries completed"
    
    def test_search_with_large_result_set(self, temp_db, mock_embedding_manager, performance_metrics):
        """Test search performance with large result sets."""
        performance_metrics.start_monitoring()
        
        # Simulate search with large result set
        start_time = time.time()
        
        mock_embedding_manager.embed_text('broad query')
        
        # Simulate processing large result set
        large_results = [(i, f'doc{i}', f'title{i}', 0.8 - i*0.001) for i in range(1000)]
        
        # Simulate result ranking and filtering
        filtered_results = [r for r in large_results if r[3] > 0.5][:50]
        
        end_time = time.time()
        performance_metrics.record_response_time(end_time - start_time)
        
        stats = performance_metrics.get_statistics()
        
        # Large result set performance assertions
        assert stats['response_time']['avg'] < 0.5, f"Large result set processing too slow: {stats['response_time']['avg']}s"
        assert len(filtered_results) <= 50, "Result set not properly limited"


class TestIndexingPerformance:
    """Test indexing operation performance."""
    
    def test_document_indexing_throughput(self, temp_db, mock_embedding_manager, performance_metrics):
        """Test document indexing throughput."""
        performance_metrics.start_monitoring()
        
        documents = [
            {'url': f'http://example.com/doc{i}', 'title': f'Document {i}', 'content': f'Content {i}' * 100}
            for i in range(100)
        ]
        
        start_time = time.time()
        
        conn = sqlite3.connect(temp_db)
        for doc in documents:
            # Simulate embedding generation
            embedding = mock_embedding_manager.embed_text(doc['content'])
            
            # Insert document
            conn.execute(
                'INSERT OR REPLACE INTO documents (url, title, content, embedding) VALUES (?, ?, ?, ?)',
                (doc['url'], doc['title'], doc['content'], str(embedding))
            )
            
            performance_metrics.record_system_metrics()
        
        conn.commit()
        conn.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        performance_metrics.record_throughput(len(documents), duration)
        stats = performance_metrics.get_statistics()
        
        # Throughput assertions
        assert stats['throughput']['avg'] > 5, f"Indexing throughput too low: {stats['throughput']['avg']} docs/sec"
        assert duration < 30, f"Indexing took too long: {duration}s for {len(documents)} documents"
    
    def test_batch_indexing_performance(self, temp_db, mock_embedding_manager, performance_metrics):
        """Test batch indexing performance."""
        performance_metrics.start_monitoring()
        
        # Create larger batch
        batch_size = 50
        documents = [
            {'url': f'http://example.com/batch{i}', 'title': f'Batch Document {i}', 'content': f'Batch content {i}' * 200}
            for i in range(batch_size)
        ]
        
        start_time = time.time()
        
        # Batch embedding generation
        texts = [doc['content'] for doc in documents]
        embeddings = mock_embedding_manager.embed_batch(texts)
        
        # Batch database insertion
        conn = sqlite3.connect(temp_db)
        batch_data = [
            (doc['url'], doc['title'], doc['content'], str(embedding))
            for doc, embedding in zip(documents, embeddings)
        ]
        
        conn.executemany(
            'INSERT OR REPLACE INTO documents (url, title, content, embedding) VALUES (?, ?, ?, ?)',
            batch_data
        )
        conn.commit()
        conn.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        performance_metrics.record_throughput(batch_size, duration)
        stats = performance_metrics.get_statistics()
        
        # Batch performance assertions
        assert stats['throughput']['avg'] > 10, f"Batch indexing throughput too low: {stats['throughput']['avg']} docs/sec"
        assert duration < 15, f"Batch indexing took too long: {duration}s for {batch_size} documents"
    
    def test_concurrent_indexing(self, temp_db, mock_embedding_manager, performance_metrics):
        """Test concurrent indexing operations."""
        performance_metrics.start_monitoring()
        
        def index_worker(worker_id, doc_count):
            """Worker function for concurrent indexing."""
            start_time = time.time()
            
            conn = sqlite3.connect(temp_db)
            for i in range(doc_count):
                doc_id = f'{worker_id}_{i}'
                content = f'Worker {worker_id} document {i} content' * 50
                
                # Generate embedding
                embedding = mock_embedding_manager.embed_text(content)
                
                # Insert document
                conn.execute(
                    'INSERT OR REPLACE INTO documents (url, title, content, embedding) VALUES (?, ?, ?, ?)',
                    (f'http://example.com/{doc_id}', f'Doc {doc_id}', content, str(embedding))
                )
            
            conn.commit()
            conn.close()
            
            end_time = time.time()
            return end_time - start_time, doc_count
        
        # Run concurrent indexing
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(index_worker, i, 20) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_docs = sum(result[1] for result in results)
        total_time = max(result[0] for result in results)
        
        performance_metrics.record_throughput(total_docs, total_time)
        stats = performance_metrics.get_statistics()
        
        # Concurrent indexing assertions
        assert stats['throughput']['avg'] > 8, f"Concurrent indexing throughput too low: {stats['throughput']['avg']} docs/sec"
        assert total_docs == 100, "Not all documents were indexed"


class TestConcurrentOperations:
    """Test performance under concurrent mixed operations."""
    
    def test_mixed_read_write_operations(self, temp_db, mock_embedding_manager, performance_metrics):
        """Test performance with mixed read/write operations."""
        performance_metrics.start_monitoring()
        
        # Pre-populate database
        conn = sqlite3.connect(temp_db)
        for i in range(500):
            conn.execute(
                'INSERT INTO documents (url, title, content) VALUES (?, ?, ?)',
                (f'http://example.com/initial{i}', f'Initial {i}', f'Initial content {i}' * 50)
            )
        conn.commit()
        conn.close()
        
        def read_worker():
            """Simulate search operations."""
            for _ in range(20):
                start_time = time.time()
                mock_embedding_manager.embed_text('search query')
                time.sleep(0.005)  # Simulate search processing
                end_time = time.time()
                performance_metrics.record_response_time(end_time - start_time)
        
        def write_worker(worker_id):
            """Simulate indexing operations."""
            conn = sqlite3.connect(temp_db)
            for i in range(10):
                content = f'New document {worker_id}_{i}' * 100
                embedding = mock_embedding_manager.embed_text(content)
                conn.execute(
                    'INSERT INTO documents (url, title, content, embedding) VALUES (?, ?, ?, ?)',
                    (f'http://example.com/new_{worker_id}_{i}', f'New {worker_id}_{i}', content, str(embedding))
                )
            conn.commit()
            conn.close()
        
        # Run mixed operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit read and write tasks
            read_futures = [executor.submit(read_worker) for _ in range(4)]
            write_futures = [executor.submit(write_worker, i) for i in range(4)]
            
            # Wait for completion
            for future in concurrent.futures.as_completed(read_futures + write_futures):
                future.result()
        
        stats = performance_metrics.get_statistics()
        
        # Mixed operations assertions
        assert stats['response_time']['avg'] < 0.1, f"Mixed operations response time too high: {stats['response_time']['avg']}s"
        
        # Verify data integrity
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM documents')
        total_docs = cursor.fetchone()[0]
        conn.close()
        
        assert total_docs >= 540, f"Expected at least 540 documents, got {total_docs}"
    
    def test_high_concurrency_stress(self, temp_db, mock_embedding_manager, performance_metrics):
        """Test system behavior under high concurrency stress."""
        performance_metrics.start_monitoring()
        
        def stress_worker(worker_id):
            """High-intensity worker function."""
            operations = 0
            start_time = time.time()
            
            for i in range(50):
                # Alternate between read and write operations
                if i % 2 == 0:
                    # Search operation
                    mock_embedding_manager.embed_text(f'stress query {worker_id}_{i}')
                    time.sleep(0.001)
                else:
                    # Index operation
                    content = f'Stress document {worker_id}_{i}'
                    embedding = mock_embedding_manager.embed_text(content)
                    
                    conn = sqlite3.connect(temp_db)
                    conn.execute(
                        'INSERT INTO documents (url, title, content, embedding) VALUES (?, ?, ?, ?)',
                        (f'http://stress.com/{worker_id}_{i}', f'Stress {worker_id}_{i}', content, str(embedding))
                    )
                    conn.commit()
                    conn.close()
                
                operations += 1
                performance_metrics.record_system_metrics()
            
            end_time = time.time()
            return operations, end_time - start_time
        
        # Run high concurrency stress test
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_operations = sum(result[0] for result in results)
        max_time = max(result[1] for result in results)
        
        performance_metrics.record_throughput(total_operations, max_time)
        stats = performance_metrics.get_statistics()
        
        # Stress test assertions
        assert stats['throughput']['avg'] > 50, f"Stress test throughput too low: {stats['throughput']['avg']} ops/sec"
        assert stats['memory_mb']['max'] < 500, f"Memory usage too high: {stats['memory_mb']['max']} MB"
        assert total_operations == 1000, f"Expected 1000 operations, completed {total_operations}"


class TestSystemResourceUsage:
    """Test system resource usage under various loads."""
    
    def test_memory_usage_during_indexing(self, temp_db, mock_embedding_manager, performance_metrics):
        """Test memory usage during heavy indexing."""
        performance_metrics.start_monitoring()
        
        # Monitor memory during large batch processing
        large_documents = [
            {'content': f'Large document content {i}' * 1000}  # ~25KB per document
            for i in range(100)
        ]
        
        conn = sqlite3.connect(temp_db)
        
        for i, doc in enumerate(large_documents):
            # Generate embedding
            embedding = mock_embedding_manager.embed_text(doc['content'])
            
            # Insert document
            conn.execute(
                'INSERT INTO documents (url, title, content, embedding) VALUES (?, ?, ?, ?)',
                (f'http://large.com/doc{i}', f'Large Doc {i}', doc['content'], str(embedding))
            )
            
            # Record metrics every 10 documents
            if i % 10 == 0:
                performance_metrics.record_system_metrics()
        
        conn.commit()
        conn.close()
        
        stats = performance_metrics.get_statistics()
        
        # Memory usage assertions
        assert stats['memory_mb']['max'] < 1000, f"Memory usage too high: {stats['memory_mb']['max']} MB"
        assert stats['memory_mb']['avg'] < 500, f"Average memory usage too high: {stats['memory_mb']['avg']} MB"
    
    def test_cpu_usage_during_concurrent_operations(self, temp_db, mock_embedding_manager, performance_metrics):
        """Test CPU usage during concurrent operations."""
        performance_metrics.start_monitoring()
        
        def cpu_intensive_worker():
            """CPU-intensive worker function."""
            for i in range(100):
                # Simulate CPU-intensive embedding computation
                content = f'CPU intensive content {i}' * 100
                embedding = mock_embedding_manager.embed_text(content)
                
                # Simulate some processing
                processed = [x * 2 for x in embedding[:100]]
                
                if i % 20 == 0:
                    performance_metrics.record_system_metrics()
        
        # Run CPU-intensive tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_worker) for _ in range(4)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        
        stats = performance_metrics.get_statistics()
        
        # CPU usage assertions (allowing for high CPU during intensive operations)
        assert stats['cpu_percent']['max'] < 200, f"CPU usage too high: {stats['cpu_percent']['max']}%"
        assert stats['cpu_percent']['avg'] < 100, f"Average CPU usage too high: {stats['cpu_percent']['avg']}%"


class TestDatabasePerformance:
    """Test database operation performance."""
    
    def test_database_query_performance(self, temp_db, performance_metrics):
        """Test database query performance with various query types."""
        performance_metrics.start_monitoring()
        
        # Populate database with test data
        conn = sqlite3.connect(temp_db)
        
        # Insert test documents
        for i in range(1000):
            conn.execute(
                'INSERT INTO documents (url, title, content) VALUES (?, ?, ?)',
                (f'http://test.com/doc{i}', f'Test Document {i}', f'Test content {i}' * 100)
            )
        
        # Insert search logs
        for i in range(500):
            conn.execute(
                'INSERT INTO search_logs (query, results_count, response_time) VALUES (?, ?, ?)',
                (f'query {i}', i % 20, 0.01 + (i % 100) * 0.001)
            )
        
        conn.commit()
        
        # Test various query types
        queries = [
            'SELECT COUNT(*) FROM documents',
            'SELECT * FROM documents WHERE title LIKE "%Test%" LIMIT 10',
            'SELECT url, title FROM documents ORDER BY created_at DESC LIMIT 20',
            'SELECT AVG(response_time) FROM search_logs',
            'SELECT query, COUNT(*) FROM search_logs GROUP BY query LIMIT 10'
        ]
        
        for query in queries * 20:  # Run each query 20 times
            start_time = time.time()
            
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            
            end_time = time.time()
            performance_metrics.record_response_time(end_time - start_time)
        
        conn.close()
        
        stats = performance_metrics.get_statistics()
        
        # Database performance assertions
        assert stats['response_time']['avg'] < 0.01, f"Database query time too high: {stats['response_time']['avg']}s"
        assert stats['response_time']['p95'] < 0.05, f"95th percentile query time too high: {stats['response_time']['p95']}s"
    
    def test_database_transaction_performance(self, temp_db, performance_metrics):
        """Test database transaction performance."""
        performance_metrics.start_monitoring()
        
        conn = sqlite3.connect(temp_db)
        
        # Test batch transactions
        start_time = time.time()
        
        conn.execute('BEGIN TRANSACTION')
        
        for i in range(100):
            conn.execute(
                'INSERT INTO documents (url, title, content) VALUES (?, ?, ?)',
                (f'http://batch.com/doc{i}', f'Batch Document {i}', f'Batch content {i}' * 50)
            )
        
        conn.execute('COMMIT')
        
        end_time = time.time()
        transaction_time = end_time - start_time
        
        performance_metrics.record_throughput(100, transaction_time)
        
        conn.close()
        
        stats = performance_metrics.get_statistics()
        
        # Transaction performance assertions
        assert stats['throughput']['avg'] > 200, f"Transaction throughput too low: {stats['throughput']['avg']} ops/sec"
        assert transaction_time < 1.0, f"Batch transaction took too long: {transaction_time}s"


if __name__ == '__main__':
    # Run performance tests with detailed output
    pytest.main([__file__, '-v', '--tb=short'])