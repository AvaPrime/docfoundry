"""Simplified performance tests for DocFoundry core components.

This module contains performance tests that focus on core functionality
without complex server dependencies.
"""

import time
import threading
import concurrent.futures
import psutil
import pytest
import sqlite3
import tempfile
import os
import sys
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.response_times = []
        self.throughput_data = []
        self.memory_usage = []
        self.cpu_usage = []
        self.start_time = None
    
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
        try:
            process = psutil.Process()
            self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
            self.cpu_usage.append(process.cpu_percent())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Handle cases where process info is not accessible
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        if self.response_times:
            sorted_times = sorted(self.response_times)
            stats['response_time'] = {
                'avg': sum(self.response_times) / len(self.response_times),
                'min': min(self.response_times),
                'max': max(self.response_times),
                'p95': sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 0 else 0,
                'p99': sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 0 else 0
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


class MockEmbeddingManager:
    """Mock embedding manager with realistic performance characteristics."""
    
    def embed_text(self, text: str) -> List[float]:
        """Simulate embedding computation."""
        # Simulate processing time based on text length
        processing_time = len(text) * 0.00001  # 10 microseconds per character
        time.sleep(processing_time)
        return [0.1] * 384  # Mock 384-dimensional embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Simulate batch embedding computation."""
        return [self.embed_text(text) for text in texts]


class TestDatabasePerformance:
    """Test database operation performance."""
    
    def test_database_insert_performance(self, temp_db, performance_metrics):
        """Test database insertion performance."""
        performance_metrics.start_monitoring()
        
        conn = sqlite3.connect(temp_db)
        
        # Test individual inserts
        start_time = time.time()
        
        for i in range(100):
            conn.execute(
                'INSERT INTO documents (url, title, content) VALUES (?, ?, ?)',
                (f'http://test.com/doc{i}', f'Document {i}', f'Content for document {i}' * 50)
            )
            
            if i % 10 == 0:
                performance_metrics.record_system_metrics()
        
        conn.commit()
        end_time = time.time()
        
        duration = end_time - start_time
        performance_metrics.record_throughput(100, duration)
        
        conn.close()
        
        stats = performance_metrics.get_statistics()
        
        # Performance assertions
        assert stats['throughput']['avg'] > 50, f"Insert throughput too low: {stats['throughput']['avg']} docs/sec"
        assert duration < 5.0, f"Insert operation took too long: {duration}s"
    
    def test_database_query_performance(self, temp_db, performance_metrics):
        """Test database query performance."""
        performance_metrics.start_monitoring()
        
        # Populate database
        conn = sqlite3.connect(temp_db)
        for i in range(1000):
            conn.execute(
                'INSERT INTO documents (url, title, content) VALUES (?, ?, ?)',
                (f'http://test.com/doc{i}', f'Document {i}', f'Content {i}' * 100)
            )
        conn.commit()
        
        # Test query performance
        queries = [
            'SELECT COUNT(*) FROM documents',
            'SELECT * FROM documents WHERE title LIKE "%Document%" LIMIT 10',
            'SELECT url, title FROM documents ORDER BY created_at DESC LIMIT 20',
        ]
        
        for query in queries * 10:  # Run each query 10 times
            start_time = time.time()
            
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            
            end_time = time.time()
            performance_metrics.record_response_time(end_time - start_time)
        
        conn.close()
        
        stats = performance_metrics.get_statistics()
        
        # Query performance assertions
        assert stats['response_time']['avg'] < 0.1, f"Query time too high: {stats['response_time']['avg']}s"
        assert stats['response_time']['p95'] < 0.2, f"95th percentile query time too high: {stats['response_time']['p95']}s"
    
    def test_database_batch_operations(self, temp_db, performance_metrics):
        """Test batch database operations performance."""
        performance_metrics.start_monitoring()
        
        conn = sqlite3.connect(temp_db)
        
        # Test batch insert
        batch_data = [
            (f'http://batch.com/doc{i}', f'Batch Document {i}', f'Batch content {i}' * 100)
            for i in range(500)
        ]
        
        start_time = time.time()
        
        conn.executemany(
            'INSERT INTO documents (url, title, content) VALUES (?, ?, ?)',
            batch_data
        )
        conn.commit()
        
        end_time = time.time()
        duration = end_time - start_time
        
        performance_metrics.record_throughput(len(batch_data), duration)
        
        conn.close()
        
        stats = performance_metrics.get_statistics()
        
        # Batch performance assertions
        assert stats['throughput']['avg'] > 100, f"Batch throughput too low: {stats['throughput']['avg']} docs/sec"
        assert duration < 10.0, f"Batch operation took too long: {duration}s"


class TestEmbeddingPerformance:
    """Test embedding operation performance."""
    
    def test_single_embedding_performance(self, performance_metrics):
        """Test single text embedding performance."""
        performance_metrics.start_monitoring()
        
        embedding_manager = MockEmbeddingManager()
        
        # Test various text sizes
        texts = [
            'Short text',
            'Medium length text with more content to process',
            'Very long text content that simulates a typical document paragraph with substantial content that would be processed by the embedding model in a real-world scenario'
        ]
        
        for text in texts * 20:  # Process each text 20 times
            start_time = time.time()
            
            embedding = embedding_manager.embed_text(text)
            
            end_time = time.time()
            performance_metrics.record_response_time(end_time - start_time)
            performance_metrics.record_system_metrics()
        
        stats = performance_metrics.get_statistics()
        
        # Embedding performance assertions
        assert stats['response_time']['avg'] < 0.1, f"Embedding time too high: {stats['response_time']['avg']}s"
        assert len(embedding) == 384, "Embedding dimension incorrect"
    
    def test_batch_embedding_performance(self, performance_metrics):
        """Test batch embedding performance."""
        performance_metrics.start_monitoring()
        
        embedding_manager = MockEmbeddingManager()
        
        # Create batch of texts
        batch_texts = [f'Document content {i} with substantial text for embedding' for i in range(50)]
        
        start_time = time.time()
        
        embeddings = embedding_manager.embed_batch(batch_texts)
        
        end_time = time.time()
        duration = end_time - start_time
        
        performance_metrics.record_throughput(len(batch_texts), duration)
        
        stats = performance_metrics.get_statistics()
        
        # Batch embedding assertions
        assert stats['throughput']['avg'] > 10, f"Batch embedding throughput too low: {stats['throughput']['avg']} texts/sec"
        assert len(embeddings) == len(batch_texts), "Batch size mismatch"
        assert all(len(emb) == 384 for emb in embeddings), "Embedding dimensions incorrect"


class TestConcurrentOperations:
    """Test concurrent operation performance."""
    
    def test_concurrent_database_operations(self, temp_db, performance_metrics):
        """Test concurrent database operations."""
        performance_metrics.start_monitoring()
        
        def db_worker(worker_id, operation_count):
            """Worker function for database operations."""
            conn = sqlite3.connect(temp_db)
            operations = 0
            
            for i in range(operation_count):
                if i % 2 == 0:
                    # Insert operation
                    conn.execute(
                        'INSERT INTO documents (url, title, content) VALUES (?, ?, ?)',
                        (f'http://worker{worker_id}.com/doc{i}', f'Worker {worker_id} Doc {i}', f'Content {i}')
                    )
                else:
                    # Query operation
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM documents')
                    cursor.fetchone()
                
                operations += 1
            
            conn.commit()
            conn.close()
            return operations
        
        # Run concurrent database operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(db_worker, i, 20) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_operations = sum(results)
        
        # Concurrent operation assertions
        assert total_operations == 100, f"Expected 100 operations, completed {total_operations}"
    
    def test_concurrent_embedding_operations(self, performance_metrics):
        """Test concurrent embedding operations."""
        performance_metrics.start_monitoring()
        
        embedding_manager = MockEmbeddingManager()
        
        def embedding_worker(worker_id, text_count):
            """Worker function for embedding operations."""
            embeddings = []
            
            for i in range(text_count):
                text = f'Worker {worker_id} processing text {i} with content'
                embedding = embedding_manager.embed_text(text)
                embeddings.append(embedding)
            
            return len(embeddings)
        
        # Run concurrent embedding operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(embedding_worker, i, 25) for i in range(4)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_embeddings = sum(results)
        
        # Concurrent embedding assertions
        assert total_embeddings == 100, f"Expected 100 embeddings, completed {total_embeddings}"


class TestSystemResourceUsage:
    """Test system resource usage under load."""
    
    def test_memory_usage_monitoring(self, performance_metrics):
        """Test memory usage during operations."""
        performance_metrics.start_monitoring()
        
        # Simulate memory-intensive operations
        large_data = []
        
        for i in range(100):
            # Create some data structures
            data = {'id': i, 'content': 'x' * 1000, 'metadata': list(range(100))}
            large_data.append(data)
            
            if i % 10 == 0:
                performance_metrics.record_system_metrics()
        
        # Process the data
        processed = [item['content'][:100] for item in large_data]
        
        stats = performance_metrics.get_statistics()
        
        # Memory usage assertions (allowing reasonable memory usage)
        if stats.get('memory_mb'):
            assert stats['memory_mb']['max'] < 2000, f"Memory usage too high: {stats['memory_mb']['max']} MB"
    
    def test_cpu_usage_monitoring(self, performance_metrics):
        """Test CPU usage during intensive operations."""
        performance_metrics.start_monitoring()
        
        # Simulate CPU-intensive operations
        for i in range(50):
            # Perform some calculations
            result = sum(x * x for x in range(1000))
            
            # String processing
            text = f'Processing iteration {i}' * 100
            processed = text.upper().lower().replace('i', 'I')
            
            if i % 10 == 0:
                performance_metrics.record_system_metrics()
        
        stats = performance_metrics.get_statistics()
        
        # CPU usage assertions (allowing for reasonable CPU usage during processing)
        if stats.get('cpu_percent'):
            # Note: CPU percentage can be high during intensive operations, so we're lenient
            assert stats['cpu_percent']['max'] < 500, f"CPU usage extremely high: {stats['cpu_percent']['max']}%"


class TestSearchPerformance:
    """Test search operation performance simulation."""
    
    def test_simulated_search_performance(self, temp_db, performance_metrics):
        """Test simulated search operations."""
        performance_metrics.start_monitoring()
        
        # Populate database with searchable content
        conn = sqlite3.connect(temp_db)
        for i in range(500):
            conn.execute(
                'INSERT INTO documents (url, title, content) VALUES (?, ?, ?)',
                (f'http://search.com/doc{i}', f'Searchable Document {i}', f'Searchable content {i}' * 50)
            )
        conn.commit()
        
        embedding_manager = MockEmbeddingManager()
        
        # Simulate search queries
        queries = ['searchable', 'document', 'content', 'information']
        
        for query in queries * 25:  # 100 total searches
            start_time = time.time()
            
            # Simulate search process
            # 1. Generate query embedding
            query_embedding = embedding_manager.embed_text(query)
            
            # 2. Search database
            cursor = conn.cursor()
            cursor.execute(
                'SELECT url, title, content FROM documents WHERE title LIKE ? OR content LIKE ? LIMIT 10',
                (f'%{query}%', f'%{query}%')
            )
            results = cursor.fetchall()
            
            # 3. Simulate ranking (simple processing)
            ranked_results = sorted(results, key=lambda x: len(x[2]), reverse=True)[:5]
            
            end_time = time.time()
            performance_metrics.record_response_time(end_time - start_time)
        
        conn.close()
        
        stats = performance_metrics.get_statistics()
        
        # Search performance assertions
        assert stats['response_time']['avg'] < 0.1, f"Search time too high: {stats['response_time']['avg']}s"
        assert stats['response_time']['p95'] < 0.2, f"95th percentile search time too high: {stats['response_time']['p95']}s"


if __name__ == '__main__':
    # Run performance tests with detailed output
    pytest.main([__file__, '-v', '--tb=short'])