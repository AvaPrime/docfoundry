"""Comprehensive test suite for DocFoundry system upgrades.

This module provides systematic testing for all implemented system upgrades
including Gold Set + Metrics Runner, Source Schema Validation, Enhanced Web UI,
and Caching & Performance Optimization.
"""

import pytest
import asyncio
import json
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Import modules to test
sys.path.append(str(Path(__file__).parent.parent))

try:
    from evaluation.gold_set_runner import (
        GoldSetConfig, MetricsSnapshot, GoldSetManager, EnhancedMetricsRunner
    )
    from indexer.enhanced_source_validator import (
        ValidationError, ValidationResult, EnhancedSourceValidator
    )
    from server.caching_optimization import (
        CacheConfig, CacheManager, PerformanceMetrics, CacheKey,
        MemoryCache, DiskCache, EmbeddingCache
    )
except ImportError as e:
    print(f"Import error: {e}")
    # Create mock classes for testing if imports fail
    class GoldSetConfig: pass
    class MetricsSnapshot: pass
    class GoldSetManager: pass
    class EnhancedMetricsRunner: pass
    class ValidationError(Exception): pass
    class ValidationResult: pass
    class EnhancedSourceValidator: pass
    class CacheConfig: pass
    class CacheManager: pass
    class PerformanceMetrics: pass
    class CacheKey: pass
    class MemoryCache: pass
    class DiskCache: pass
    class EmbeddingCache: pass


class TestGoldSetRunner:
    """Test suite for Gold Set + Metrics Runner Enhancement."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def gold_set_config(self, temp_dir):
        """Create test configuration."""
        return GoldSetConfig(
            dataset_path=f"{temp_dir}/test_dataset.json",
            output_dir=temp_dir,
            search_endpoints={
                "semantic": "http://localhost:8001/search/semantic",
                "hybrid": "http://localhost:8001/search"
            },
            metrics=["precision_at_k", "recall_at_k", "ndcg_at_k"],
            k_values=[1, 5, 10],
            performance_thresholds={
                "precision_at_1": 0.8,
                "recall_at_5": 0.7,
                "response_time_ms": 500
            }
        )
    
    @pytest.fixture
    def sample_dataset(self, temp_dir):
        """Create sample evaluation dataset."""
        dataset = {
            "queries": [
                {
                    "id": "q1",
                    "query": "machine learning algorithms",
                    "relevant_docs": ["doc1", "doc2"],
                    "domain": "AI"
                },
                {
                    "id": "q2",
                    "query": "data preprocessing techniques",
                    "relevant_docs": ["doc3", "doc4", "doc5"],
                    "domain": "Data Science"
                }
            ],
            "documents": {
                "doc1": {"title": "ML Algorithms Overview", "content": "..."},
                "doc2": {"title": "Deep Learning Guide", "content": "..."},
                "doc3": {"title": "Data Cleaning Methods", "content": "..."},
                "doc4": {"title": "Feature Engineering", "content": "..."},
                "doc5": {"title": "Data Transformation", "content": "..."}
            }
        }
        
        dataset_path = Path(temp_dir) / "test_dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f)
        
        return dataset_path
    
    def test_gold_set_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = GoldSetConfig(
            dataset_path="test.json",
            search_endpoints={"semantic": "http://localhost:8001/search"}
        )
        assert config.dataset_path == "test.json"
        assert len(config.search_endpoints) == 1
        
        # Test default values
        assert config.batch_size == 10
        assert config.timeout_seconds == 30
        assert "precision_at_k" in config.metrics
    
    def test_metrics_snapshot(self):
        """Test metrics snapshot functionality."""
        snapshot = MetricsSnapshot(
            timestamp=datetime.now(),
            endpoint="semantic",
            domain="AI",
            metrics={
                "precision_at_1": 0.85,
                "recall_at_5": 0.72,
                "response_time_ms": 245
            },
            query_count=50,
            error_count=2
        )
        
        # Test serialization
        snapshot_dict = snapshot.to_dict()
        assert snapshot_dict["endpoint"] == "semantic"
        assert snapshot_dict["metrics"]["precision_at_1"] == 0.85
        assert "timestamp" in snapshot_dict
        
        # Test comparison
        baseline = MetricsSnapshot(
            timestamp=datetime.now() - timedelta(days=1),
            endpoint="semantic",
            domain="AI",
            metrics={
                "precision_at_1": 0.80,
                "recall_at_5": 0.70,
                "response_time_ms": 300
            },
            query_count=50,
            error_count=1
        )
        
        comparison = snapshot.compare_with(baseline)
        assert comparison["precision_at_1"] > 0  # Improvement
        assert comparison["response_time_ms"] < 0  # Improvement (lower is better)
    
    def test_gold_set_manager(self, temp_dir, sample_dataset):
        """Test gold set management functionality."""
        manager = GoldSetManager()
        
        # Test loading dataset
        dataset = manager.load_dataset(str(sample_dataset))
        assert len(dataset.queries) == 2
        assert dataset.queries[0].query == "machine learning algorithms"
        assert len(dataset.queries[0].relevant_docs) == 2
        
        # Test filtering by domain
        ai_queries = manager.filter_by_domain(dataset, "AI")
        assert len(ai_queries) == 1
        assert ai_queries[0].domain == "AI"
        
        # Test sampling
        sample = manager.sample_queries(dataset, 1)
        assert len(sample) == 1
        
        # Test creating new dataset
        new_queries = [
            {
                "id": "q3",
                "query": "neural networks",
                "relevant_docs": ["doc6"],
                "domain": "AI"
            }
        ]
        
        new_dataset_path = Path(temp_dir) / "new_dataset.json"
        manager.create_dataset(new_queries, {}, str(new_dataset_path))
        
        assert new_dataset_path.exists()
        with open(new_dataset_path) as f:
            saved_data = json.load(f)
        assert len(saved_data["queries"]) == 1
    
    @pytest.mark.asyncio
    async def test_enhanced_metrics_runner(self, gold_set_config, sample_dataset):
        """Test enhanced metrics runner functionality."""
        runner = EnhancedMetricsRunner(gold_set_config)
        
        # Mock search API responses
        mock_responses = {
            "machine learning algorithms": [
                {"doc_id": "doc1", "score": 0.95, "title": "ML Algorithms Overview"},
                {"doc_id": "doc3", "score": 0.85, "title": "Data Cleaning Methods"},
                {"doc_id": "doc2", "score": 0.80, "title": "Deep Learning Guide"}
            ],
            "data preprocessing techniques": [
                {"doc_id": "doc3", "score": 0.92, "title": "Data Cleaning Methods"},
                {"doc_id": "doc4", "score": 0.88, "title": "Feature Engineering"},
                {"doc_id": "doc5", "score": 0.85, "title": "Data Transformation"}
            ]
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Configure mock to return appropriate responses
            async def mock_response(url, **kwargs):
                data = kwargs.get('json', {})
                query = data.get('query', '')
                
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.json = AsyncMock(return_value={
                    'results': mock_responses.get(query, [])
                })
                return mock_resp
            
            mock_post.side_effect = mock_response
            
            # Run evaluation
            results = await runner.run_evaluation(str(sample_dataset))
            
            # Verify results structure
            assert "summary" in results
            assert "detailed_results" in results
            assert "performance_analysis" in results
            
            # Check summary metrics
            summary = results["summary"]
            assert "overall_precision_at_1" in summary
            assert "overall_recall_at_5" in summary
            assert "avg_response_time_ms" in summary
            
            # Check detailed results
            detailed = results["detailed_results"]
            assert len(detailed) > 0
            
            # Verify performance analysis
            performance = results["performance_analysis"]
            assert "threshold_compliance" in performance
            assert "recommendations" in performance
    
    def test_report_generation(self, gold_set_config, temp_dir):
        """Test report generation functionality."""
        runner = EnhancedMetricsRunner(gold_set_config)
        
        # Sample results data
        results = {
            "summary": {
                "overall_precision_at_1": 0.85,
                "overall_recall_at_5": 0.72,
                "avg_response_time_ms": 245,
                "total_queries": 100,
                "successful_queries": 98
            },
            "detailed_results": [
                {
                    "query_id": "q1",
                    "query": "test query",
                    "endpoint": "semantic",
                    "precision_at_1": 1.0,
                    "recall_at_5": 0.8,
                    "response_time_ms": 200
                }
            ],
            "performance_analysis": {
                "threshold_compliance": {"precision_at_1": True},
                "recommendations": ["Consider optimizing response time"]
            }
        }
        
        # Test HTML report generation
        html_path = runner.generate_html_report(results, "test_run")
        assert Path(html_path).exists()
        
        with open(html_path) as f:
            html_content = f.read()
        assert "DocFoundry Evaluation Report" in html_content
        assert "0.85" in html_content  # Check precision value
        
        # Test CSV report generation
        csv_path = runner.generate_csv_report(results, "test_run")
        assert Path(csv_path).exists()
        
        with open(csv_path) as f:
            csv_content = f.read()
        assert "query_id,query,endpoint" in csv_content
        assert "q1,test query,semantic" in csv_content


class TestEnhancedSourceValidator:
    """Test suite for Enhanced Source Schema Validation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return EnhancedSourceValidator()
    
    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample files for testing."""
        files = {}
        
        # Valid PDF file (mock)
        pdf_path = Path(temp_dir) / "document.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%mock pdf content")
        files["valid_pdf"] = pdf_path
        
        # Invalid file (too small)
        small_path = Path(temp_dir) / "small.txt"
        small_path.write_text("x")
        files["too_small"] = small_path
        
        # Large text file
        large_path = Path(temp_dir) / "large.txt"
        large_path.write_text("x" * 1000)
        files["large_text"] = large_path
        
        # Configuration file
        config_path = Path(temp_dir) / "config.json"
        config_data = {
            "max_file_size_mb": 10,
            "allowed_extensions": [".pdf", ".txt", ".docx"],
            "require_metadata": True
        }
        config_path.write_text(json.dumps(config_data))
        files["config"] = config_path
        
        return files
    
    def test_validation_error(self):
        """Test validation error functionality."""
        error = ValidationError(
            field="file_size",
            message="File too large",
            severity="error",
            code="FILE_SIZE_EXCEEDED"
        )
        
        assert error.field == "file_size"
        assert error.severity == "error"
        assert error.is_error()
        assert not error.is_warning()
        
        # Test string representation
        error_str = str(error)
        assert "FILE_SIZE_EXCEEDED" in error_str
        assert "File too large" in error_str
    
    def test_validation_result(self):
        """Test validation result functionality."""
        result = ValidationResult()
        
        # Add errors and warnings
        result.add_error("field1", "Error message", "ERROR_CODE")
        result.add_warning("field2", "Warning message", "WARNING_CODE")
        
        assert not result.is_valid()
        assert result.has_errors()
        assert result.has_warnings()
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        
        # Test summary
        summary = result.get_summary()
        assert summary["total_issues"] == 2
        assert summary["error_count"] == 1
        assert summary["warning_count"] == 1
        
        # Test valid result
        valid_result = ValidationResult()
        assert valid_result.is_valid()
        assert not valid_result.has_errors()
    
    def test_file_validation(self, validator, sample_files):
        """Test individual file validation."""
        # Test valid PDF
        result = validator.validate_file(str(sample_files["valid_pdf"]))
        assert result.is_valid() or result.has_warnings()  # May have warnings but should not error
        
        # Test file too small
        result = validator.validate_file(str(sample_files["too_small"]))
        assert result.has_errors()
        error_codes = [error.code for error in result.errors]
        assert "FILE_TOO_SMALL" in error_codes
        
        # Test large text file
        result = validator.validate_file(str(sample_files["large_text"]))
        assert result.is_valid()  # Should be valid
        
        # Test non-existent file
        result = validator.validate_file("/non/existent/file.pdf")
        assert result.has_errors()
        error_codes = [error.code for error in result.errors]
        assert "FILE_NOT_FOUND" in error_codes
    
    def test_configuration_validation(self, validator, sample_files):
        """Test configuration file validation."""
        # Test valid configuration
        result = validator.validate_configuration(str(sample_files["config"]))
        assert result.is_valid()
        
        # Test invalid configuration
        invalid_config = {"invalid_key": "value"}
        invalid_path = sample_files["config"].parent / "invalid_config.json"
        invalid_path.write_text(json.dumps(invalid_config))
        
        result = validator.validate_configuration(str(invalid_path))
        assert result.has_errors() or result.has_warnings()
    
    def test_cross_field_validation(self, validator, temp_dir):
        """Test cross-field validation logic."""
        # Create test data with cross-field dependencies
        test_data = {
            "source_type": "pdf",
            "file_path": str(Path(temp_dir) / "test.pdf"),
            "metadata": {
                "title": "Test Document",
                "author": "Test Author"
            },
            "processing_options": {
                "extract_images": True,
                "ocr_enabled": False  # Conflict: extract_images requires OCR
            }
        }
        
        result = validator._validate_cross_field_constraints(test_data)
        
        # Should detect the logical inconsistency
        if result.has_warnings():
            warning_messages = [w.message for w in result.warnings]
            assert any("OCR" in msg for msg in warning_messages)
    
    def test_performance_validation(self, validator):
        """Test performance constraint validation."""
        # Test with performance settings
        perf_data = {
            "batch_size": 1000,  # Too large
            "timeout_seconds": 1,  # Too small
            "max_concurrent_jobs": 100,  # Too many
            "memory_limit_mb": 50  # Too small
        }
        
        result = validator._validate_performance_settings(perf_data)
        
        # Should have warnings or errors for unrealistic settings
        assert result.has_warnings() or result.has_errors()
        
        # Test with reasonable settings
        good_perf_data = {
            "batch_size": 10,
            "timeout_seconds": 30,
            "max_concurrent_jobs": 4,
            "memory_limit_mb": 512
        }
        
        result = validator._validate_performance_settings(good_perf_data)
        assert result.is_valid()
    
    def test_security_validation(self, validator):
        """Test security constraint validation."""
        # Test with security issues
        security_data = {
            "allowed_paths": ["/tmp", "/var/tmp"],  # Potentially unsafe
            "enable_shell_execution": True,  # Dangerous
            "api_key": "hardcoded_key_123",  # Security issue
            "debug_mode": True  # Should warn in production
        }
        
        result = validator._validate_security_settings(security_data)
        
        # Should detect security issues
        assert result.has_warnings() or result.has_errors()
        
        # Check for specific security warnings
        all_messages = [e.message for e in result.errors + result.warnings]
        assert any("shell" in msg.lower() for msg in all_messages)
    
    def test_directory_validation(self, validator, temp_dir):
        """Test directory validation functionality."""
        # Create test directory structure
        test_dir = Path(temp_dir) / "test_docs"
        test_dir.mkdir()
        
        # Add various files
        (test_dir / "doc1.pdf").write_bytes(b"%PDF-1.4\ntest content" * 100)
        (test_dir / "doc2.txt").write_text("Test document content" * 50)
        (test_dir / "invalid.exe").write_bytes(b"invalid executable")
        (test_dir / "empty.txt").write_text("")  # Empty file
        
        # Create subdirectory
        sub_dir = test_dir / "subdocs"
        sub_dir.mkdir()
        (sub_dir / "doc3.pdf").write_bytes(b"%PDF-1.4\nmore content" * 100)
        
        # Run directory validation
        results = validator.validate_directory(str(test_dir))
        
        # Should have results for each file
        assert len(results) >= 4  # At least 4 files processed
        
        # Check that invalid files were caught
        invalid_results = [r for r in results if not r["result"].is_valid()]
        assert len(invalid_results) > 0  # Should catch invalid.exe and empty.txt
        
        # Generate summary
        summary = validator.generate_validation_summary(results)
        assert "total_files" in summary
        assert "valid_files" in summary
        assert "invalid_files" in summary
        assert summary["total_files"] >= 4


class TestCachingOptimization:
    """Test suite for Caching & Performance Optimization."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_config(self, temp_dir):
        """Create test cache configuration."""
        return CacheConfig(
            redis_url="redis://localhost:6379/15",  # Use test database
            search_results_ttl=300,
            embeddings_ttl=600,
            enable_memory_fallback=True,
            enable_disk_fallback=True,
            disk_cache_path=temp_dir
        )
    
    def test_cache_config(self):
        """Test cache configuration."""
        config = CacheConfig()
        
        # Test default values
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.search_results_ttl == 3600
        assert config.compression_enabled is True
        assert config.enable_memory_fallback is True
    
    def test_performance_metrics(self):
        """Test performance metrics functionality."""
        metrics = PerformanceMetrics()
        
        # Initial state
        assert metrics.hit_rate == 0.0
        assert metrics.miss_rate == 1.0
        assert metrics.total_requests == 0
        
        # Simulate some activity
        metrics.cache_hits = 8
        metrics.cache_misses = 2
        metrics.total_requests = 10
        metrics.avg_response_time = 0.15
        
        assert metrics.hit_rate == 0.8
        assert metrics.miss_rate == 0.2
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        assert metrics_dict["hit_rate"] == 0.8
        assert metrics_dict["total_requests"] == 10
        assert "last_updated" in metrics_dict
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        # Test search results key
        key1 = CacheKey.search_results(
            "machine learning",
            {"source": "docs", "lang": "en"},
            "hybrid"
        )
        key2 = CacheKey.search_results(
            "machine learning",
            {"lang": "en", "source": "docs"},  # Different order
            "hybrid"
        )
        assert key1 == key2  # Should be same despite different order
        
        # Test embedding key
        emb_key1 = CacheKey.embedding("test text", "model1")
        emb_key2 = CacheKey.embedding("test text", "model2")
        assert emb_key1 != emb_key2  # Different models
        
        # Test other key types
        doc_key = CacheKey.document_metadata("doc123")
        assert doc_key.startswith("metadata:")
        
        analytics_key = CacheKey.analytics("search_volume", "daily")
        assert analytics_key.startswith("analytics:")
    
    def test_memory_cache(self):
        """Test in-memory cache functionality."""
        cache = MemoryCache(max_size=3)
        
        # Test basic operations
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        
        # Test LRU eviction
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
        
        # Test TTL
        import time
        cache.set("ttl_key", "ttl_value", ttl=1)
        assert cache.get("ttl_key") == "ttl_value"
        
        # Simulate time passing (we'll test cleanup separately)
        cache.cache["ttl_key"] = ("ttl_value", time.time() - 10)  # Expired
        expired_count = cache.cleanup_expired()
        assert expired_count == 1
        assert cache.get("ttl_key") is None
        
        # Test deletion
        assert cache.delete("key2") is True
        assert cache.get("key2") is None
        assert cache.delete("nonexistent") is False
        
        # Test clear
        cache.clear()
        assert cache.size() == 0
        
        # Test stats
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 3
    
    def test_disk_cache(self, temp_dir):
        """Test disk cache functionality."""
        cache = DiskCache(temp_dir)
        
        # Test basic operations
        test_data = {"message": "hello", "number": 42}
        cache.set("test_key", test_data)
        
        retrieved_data = cache.get("test_key")
        assert retrieved_data == test_data
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
        
        # Test deletion
        assert cache.delete("test_key") is True
        assert cache.get("test_key") is None
        assert cache.delete("nonexistent") is False
        
        # Test TTL (simulate expired file)
        cache.set("ttl_key", "ttl_value", ttl=1)
        
        # Manually modify file to be expired
        file_path = cache._get_file_path("ttl_key")
        if file_path.exists():
            import pickle
            import gzip
            expired_data = {
                'value': "ttl_value",
                'expiry': time.time() - 10,  # Expired
                'created': time.time() - 20
            }
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(expired_data, f)
        
        assert cache.get("ttl_key") is None  # Should be None due to expiry
        
        # Test cleanup
        cache.set("valid_key", "valid_value")
        expired_count = cache.cleanup_expired()
        assert cache.get("valid_key") == "valid_value"  # Should still exist
        
        # Test clear
        cache.clear()
        assert cache.size() == 0
    
    @pytest.mark.asyncio
    async def test_cache_manager(self, cache_config):
        """Test cache manager functionality."""
        # Skip Redis tests if not available
        cache_manager = CacheManager(cache_config)
        
        # Test basic operations
        test_key = "test:manager:key"
        test_value = {"data": "test", "timestamp": time.time()}
        
        # Set value
        success = await cache_manager.aset(test_key, test_value, 60)
        assert success is True
        
        # Get value
        retrieved_value = await cache_manager.aget(test_key)
        assert retrieved_value == test_value
        
        # Test synchronous methods
        sync_key = "sync:key"
        sync_value = "sync_value"
        
        cache_manager.set(sync_key, sync_value)
        assert cache_manager.get(sync_key) == sync_value
        
        # Test deletion
        await cache_manager.adelete(test_key)
        assert await cache_manager.aget(test_key) is None
        
        # Test metrics
        metrics = cache_manager.get_metrics()
        assert metrics.total_requests > 0
        
        # Test cache stats
        stats = cache_manager.get_cache_stats()
        assert "performance" in stats
        assert "memory" in stats
        assert "disk" in stats
    
    def test_embedding_cache(self, cache_config):
        """Test embedding cache functionality."""
        cache_manager = CacheManager(cache_config)
        embedding_cache = EmbeddingCache(cache_manager)
        
        # Test embedding storage and retrieval
        test_text = "This is a test sentence for embedding."
        test_embedding = np.random.rand(384)
        
        # Set embedding
        embedding_cache.set_embedding(test_text, test_embedding)
        
        # Get embedding
        retrieved_embedding = embedding_cache.get_embedding(test_text)
        assert retrieved_embedding is not None
        assert np.allclose(retrieved_embedding, test_embedding)
        
        # Test similarity search
        similar_text = "This is a test sentence for embedding."  # Exact match
        similar_embedding = embedding_cache.get_embedding(similar_text)
        assert similar_embedding is not None
        
        # Test non-existent embedding
        non_existent = embedding_cache.get_embedding("completely different text")
        # May or may not be None depending on similarity threshold
    
    def test_cache_decorator(self, cache_config):
        """Test cache decorator functionality."""
        from server.caching_optimization import cache_result
        
        cache_manager = CacheManager(cache_config)
        
        # Create a test function with caching
        call_count = 0
        
        @cache_result(ttl=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # Attach cache manager to function
        expensive_function._cache_manager = cache_manager
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Different arguments should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2


class TestIntegration:
    """Integration tests for all system upgrades."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end_evaluation(self, temp_dir):
        """Test end-to-end evaluation workflow."""
        # Create test dataset
        dataset = {
            "queries": [
                {
                    "id": "integration_q1",
                    "query": "integration test query",
                    "relevant_docs": ["doc1"],
                    "domain": "Test"
                }
            ],
            "documents": {
                "doc1": {"title": "Test Document", "content": "Test content"}
            }
        }
        
        dataset_path = Path(temp_dir) / "integration_dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f)
        
        # Configure evaluation
        config = GoldSetConfig(
            dataset_path=str(dataset_path),
            output_dir=temp_dir,
            search_endpoints={"test": "http://localhost:8001/search"},
            metrics=["precision_at_k"],
            k_values=[1]
        )
        
        # Mock search response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={
                'results': [{"doc_id": "doc1", "score": 0.95}]
            })
            mock_post.return_value = mock_resp
            
            # Run evaluation
            runner = EnhancedMetricsRunner(config)
            results = await runner.run_evaluation(str(dataset_path))
            
            # Verify results
            assert results["summary"]["overall_precision_at_1"] == 1.0
    
    def test_validation_with_caching(self, temp_dir):
        """Test validation system with caching integration."""
        # Create cache manager
        cache_config = CacheConfig(
            enable_memory_fallback=True,
            enable_disk_fallback=True,
            disk_cache_path=temp_dir
        )
        cache_manager = CacheManager(cache_config)
        
        # Create validator
        validator = EnhancedSourceValidator()
        
        # Create test file
        test_file = Path(temp_dir) / "test_doc.txt"
        test_file.write_text("Test document content" * 100)
        
        # First validation (should be slow)
        start_time = time.time()
        result1 = validator.validate_file(str(test_file))
        first_duration = time.time() - start_time
        
        # Cache the result
        cache_key = f"validation:{test_file.name}"
        cache_manager.set(cache_key, result1.get_summary())
        
        # Second validation (should use cache)
        start_time = time.time()
        cached_result = cache_manager.get(cache_key)
        second_duration = time.time() - start_time
        
        # Verify caching worked
        assert cached_result is not None
        assert second_duration < first_duration  # Should be faster
        assert cached_result["valid_files"] == result1.get_summary()["valid_files"]


if __name__ == "__main__":
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])