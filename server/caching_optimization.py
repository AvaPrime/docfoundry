"""Caching & Performance Optimization for DocFoundry

This module provides comprehensive caching strategies, performance monitoring,
and optimization features for the DocFoundry search system.
"""

import json
import hashlib
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from functools import wraps
from contextlib import asynccontextmanager
import pickle
import gzip
from pathlib import Path

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    aioredis = None

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    
    # Cache TTL settings (in seconds)
    search_results_ttl: int = 3600  # 1 hour
    embeddings_ttl: int = 86400 * 7  # 1 week
    metadata_ttl: int = 86400  # 1 day
    analytics_ttl: int = 300  # 5 minutes
    
    # Cache size limits
    max_search_cache_size: int = 10000
    max_embedding_cache_size: int = 50000
    max_memory_cache_size: int = 1000
    
    # Performance settings
    compression_enabled: bool = True
    compression_threshold: int = 1024  # bytes
    batch_size: int = 100
    
    # Fallback settings
    enable_memory_fallback: bool = True
    enable_disk_fallback: bool = True
    disk_cache_path: str = "./cache"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    
    cache_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    peak_memory_usage: float = 0.0
    error_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': self.total_requests,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'avg_response_time': self.avg_response_time,
            'peak_memory_usage': self.peak_memory_usage,
            'error_count': self.error_count,
            'last_updated': self.last_updated.isoformat()
        }


class CacheKey:
    """Utility class for generating consistent cache keys."""
    
    @staticmethod
    def search_results(query: str, filters: Dict[str, Any], search_mode: str) -> str:
        """Generate cache key for search results."""
        key_data = {
            'query': query.lower().strip(),
            'filters': sorted(filters.items()) if filters else [],
            'mode': search_mode
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"search:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    @staticmethod
    def embedding(text: str, model: str = "default") -> str:
        """Generate cache key for embeddings."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"embedding:{model}:{text_hash}"
    
    @staticmethod
    def document_metadata(doc_id: str) -> str:
        """Generate cache key for document metadata."""
        return f"metadata:{doc_id}"
    
    @staticmethod
    def analytics(metric_type: str, time_window: str) -> str:
        """Generate cache key for analytics data."""
        return f"analytics:{metric_type}:{time_window}"
    
    @staticmethod
    def user_session(session_id: str) -> str:
        """Generate cache key for user session data."""
        return f"session:{session_id}"


class MemoryCache:
    """In-memory LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key][0]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        current_time = time.time()
        expiry_time = current_time + ttl if ttl else float('inf')
        
        if key in self.cache:
            # Update existing key
            self.cache[key] = (value, expiry_time)
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new key
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = (value, expiry_time)
            self.access_order.append(key)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        current_time = time.time()
        expired_keys = []
        
        for key, (_, expiry_time) in self.cache.items():
            if expiry_time < current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0
        }


class DiskCache:
    """Disk-based cache implementation."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            # Check expiry
            if data['expiry'] < time.time():
                file_path.unlink(missing_ok=True)
                return None
            
            return data['value']
        except Exception as e:
            logger.warning(f"Error reading disk cache {key}: {e}")
            file_path.unlink(missing_ok=True)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in disk cache."""
        file_path = self._get_file_path(key)
        expiry_time = time.time() + ttl if ttl else float('inf')
        
        data = {
            'value': value,
            'expiry': expiry_time,
            'created': time.time()
        }
        
        try:
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Error writing disk cache {key}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from disk cache."""
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def clear(self) -> None:
        """Clear all disk cache entries."""
        for file_path in self.cache_dir.glob("*.cache"):
            file_path.unlink(missing_ok=True)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        current_time = time.time()
        expired_count = 0
        
        for file_path in self.cache_dir.glob("*.cache"):
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                if data['expiry'] < current_time:
                    file_path.unlink()
                    expired_count += 1
            except Exception:
                # Remove corrupted files
                file_path.unlink(missing_ok=True)
                expired_count += 1
        
        return expired_count
    
    def size(self) -> int:
        """Get current cache size (number of files)."""
        return len(list(self.cache_dir.glob("*.cache")))


class CacheManager:
    """Main cache manager with multiple backend support."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        
        # Initialize cache backends
        self.redis_client: Optional[redis.Redis] = None
        self.async_redis_client: Optional[aioredis.Redis] = None
        self.memory_cache = MemoryCache(config.max_memory_cache_size) if config.enable_memory_fallback else None
        self.disk_cache = DiskCache(config.disk_cache_path) if config.enable_disk_fallback else None
        
        # Initialize Redis if available
        if REDIS_AVAILABLE:
            self._init_redis()
        
        # Start background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _init_redis(self):
        """Initialize Redis connections."""
        try:
            # Synchronous Redis client
            self.redis_client = redis.from_url(
                self.config.redis_url,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                decode_responses=False,  # We handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            
            # Asynchronous Redis client
            self.async_redis_client = aioredis.from_url(
                self.config.redis_url,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}. Using fallback caches.")
            self.redis_client = None
            self.async_redis_client = None
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Run every 5 minutes
                    await self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            pass
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        data = pickle.dumps(value)
        
        if self.config.compression_enabled and len(data) > self.config.compression_threshold:
            data = gzip.compress(data)
            return b'compressed:' + data
        
        return b'raw:' + data
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if data.startswith(b'compressed:'):
            data = gzip.decompress(data[11:])  # Remove 'compressed:' prefix
        elif data.startswith(b'raw:'):
            data = data[4:]  # Remove 'raw:' prefix
        
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (synchronous)."""
        start_time = time.time()
        
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    data = self.redis_client.get(key)
                    if data is not None:
                        value = self._deserialize_value(data)
                        self._record_hit(time.time() - start_time)
                        return value
                except Exception as e:
                    logger.warning(f"Redis get error for key {key}: {e}")
            
            # Try memory cache
            if self.memory_cache:
                value = self.memory_cache.get(key)
                if value is not None:
                    self._record_hit(time.time() - start_time)
                    return value
            
            # Try disk cache
            if self.disk_cache:
                value = self.disk_cache.get(key)
                if value is not None:
                    self._record_hit(time.time() - start_time)
                    # Promote to memory cache
                    if self.memory_cache:
                        self.memory_cache.set(key, value)
                    return value
            
            self._record_miss(time.time() - start_time)
            return None
        
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self._record_error()
            return None
    
    async def aget(self, key: str) -> Optional[Any]:
        """Get value from cache (asynchronous)."""
        start_time = time.time()
        
        try:
            # Try Redis first
            if self.async_redis_client:
                try:
                    data = await self.async_redis_client.get(key)
                    if data is not None:
                        value = self._deserialize_value(data)
                        self._record_hit(time.time() - start_time)
                        return value
                except Exception as e:
                    logger.warning(f"Async Redis get error for key {key}: {e}")
            
            # Fallback to synchronous methods
            return self.get(key)
        
        except Exception as e:
            logger.error(f"Async cache get error for key {key}: {e}")
            self._record_error()
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache (synchronous)."""
        try:
            serialized_value = self._serialize_value(value)
            success = False
            
            # Set in Redis
            if self.redis_client:
                try:
                    if ttl:
                        self.redis_client.setex(key, ttl, serialized_value)
                    else:
                        self.redis_client.set(key, serialized_value)
                    success = True
                except Exception as e:
                    logger.warning(f"Redis set error for key {key}: {e}")
            
            # Set in memory cache
            if self.memory_cache:
                self.memory_cache.set(key, value, ttl)
                success = True
            
            # Set in disk cache
            if self.disk_cache:
                self.disk_cache.set(key, value, ttl)
                success = True
            
            return success
        
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self._record_error()
            return False
    
    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache (asynchronous)."""
        try:
            serialized_value = self._serialize_value(value)
            success = False
            
            # Set in Redis
            if self.async_redis_client:
                try:
                    if ttl:
                        await self.async_redis_client.setex(key, ttl, serialized_value)
                    else:
                        await self.async_redis_client.set(key, serialized_value)
                    success = True
                except Exception as e:
                    logger.warning(f"Async Redis set error for key {key}: {e}")
            
            # Fallback to synchronous methods for other caches
            if self.memory_cache:
                self.memory_cache.set(key, value, ttl)
                success = True
            
            if self.disk_cache:
                self.disk_cache.set(key, value, ttl)
                success = True
            
            return success
        
        except Exception as e:
            logger.error(f"Async cache set error for key {key}: {e}")
            self._record_error()
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from all caches."""
        success = False
        
        # Delete from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                success = True
            except Exception as e:
                logger.warning(f"Redis delete error for key {key}: {e}")
        
        # Delete from memory cache
        if self.memory_cache:
            if self.memory_cache.delete(key):
                success = True
        
        # Delete from disk cache
        if self.disk_cache:
            if self.disk_cache.delete(key):
                success = True
        
        return success
    
    async def adelete(self, key: str) -> bool:
        """Delete key from all caches (asynchronous)."""
        success = False
        
        # Delete from Redis
        if self.async_redis_client:
            try:
                await self.async_redis_client.delete(key)
                success = True
            except Exception as e:
                logger.warning(f"Async Redis delete error for key {key}: {e}")
        
        # Fallback to synchronous methods
        if self.memory_cache:
            if self.memory_cache.delete(key):
                success = True
        
        if self.disk_cache:
            if self.disk_cache.delete(key):
                success = True
        
        return success
    
    def clear(self) -> None:
        """Clear all caches."""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
        
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self.disk_cache:
            self.disk_cache.clear()
    
    async def _cleanup_expired(self) -> None:
        """Clean up expired entries from all caches."""
        try:
            # Memory cache cleanup
            if self.memory_cache:
                expired_count = self.memory_cache.cleanup_expired()
                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired memory cache entries")
            
            # Disk cache cleanup
            if self.disk_cache:
                expired_count = self.disk_cache.cleanup_expired()
                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired disk cache entries")
        
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def _record_hit(self, response_time: float) -> None:
        """Record cache hit."""
        self.metrics.cache_hits += 1
        self.metrics.total_requests += 1
        self._update_avg_response_time(response_time)
    
    def _record_miss(self, response_time: float) -> None:
        """Record cache miss."""
        self.metrics.cache_misses += 1
        self.metrics.total_requests += 1
        self._update_avg_response_time(response_time)
    
    def _record_error(self) -> None:
        """Record cache error."""
        self.metrics.error_count += 1
    
    def _update_avg_response_time(self, response_time: float) -> None:
        """Update average response time."""
        if self.metrics.total_requests == 1:
            self.metrics.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_response_time = (
                alpha * response_time + (1 - alpha) * self.metrics.avg_response_time
            )
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        self.metrics.last_updated = datetime.now()
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = PerformanceMetrics()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        stats = {
            'performance': self.metrics.to_dict(),
            'redis': {'available': self.redis_client is not None},
            'memory': self.memory_cache.stats() if self.memory_cache else None,
            'disk': {'size': self.disk_cache.size()} if self.disk_cache else None
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats['redis'].update({
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                })
            except Exception as e:
                logger.warning(f"Error getting Redis stats: {e}")
        
        return stats


def cache_result(ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': sorted(kwargs.items())
                }
                key_string = json.dumps(key_data, sort_keys=True, default=str)
                cache_key = f"func:{hashlib.md5(key_string.encode()).hexdigest()}"
            
            # Try to get from cache
            if hasattr(func, '_cache_manager'):
                cache_manager = func._cache_manager
                cached_result = cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            if hasattr(func, '_cache_manager'):
                cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class EmbeddingCache:
    """Specialized cache for embeddings with similarity search."""
    
    def __init__(self, cache_manager: CacheManager, similarity_threshold: float = 0.95):
        self.cache_manager = cache_manager
        self.similarity_threshold = similarity_threshold
        self.embedding_index: Dict[str, np.ndarray] = {}
    
    def get_embedding(self, text: str, model: str = "default") -> Optional[np.ndarray]:
        """Get embedding from cache or find similar."""
        cache_key = CacheKey.embedding(text, model)
        
        # Try exact match first
        cached_embedding = self.cache_manager.get(cache_key)
        if cached_embedding is not None:
            return np.array(cached_embedding)
        
        # Try similarity search if we have embeddings in memory
        if self.embedding_index:
            text_lower = text.lower().strip()
            
            for cached_text, cached_emb in self.embedding_index.items():
                # Simple text similarity check first
                if self._text_similarity(text_lower, cached_text.lower()) > 0.9:
                    return cached_emb
        
        return None
    
    def set_embedding(self, text: str, embedding: np.ndarray, model: str = "default") -> None:
        """Set embedding in cache."""
        cache_key = CacheKey.embedding(text, model)
        
        # Store in cache manager
        self.cache_manager.set(cache_key, embedding.tolist(), self.cache_manager.config.embeddings_ttl)
        
        # Keep in memory index for similarity search (limited size)
        if len(self.embedding_index) < 1000:  # Limit memory usage
            self.embedding_index[text] = embedding
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        if text1 == text2:
            return 1.0
        
        # Simple Jaccard similarity on words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> Optional[CacheManager]:
    """Get global cache manager instance."""
    return _cache_manager


def initialize_cache(config: CacheConfig) -> CacheManager:
    """Initialize global cache manager."""
    global _cache_manager
    _cache_manager = CacheManager(config)
    return _cache_manager


def shutdown_cache() -> None:
    """Shutdown global cache manager."""
    global _cache_manager
    if _cache_manager and _cache_manager._cleanup_task:
        _cache_manager._cleanup_task.cancel()
    _cache_manager = None


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_cache():
        """Test cache functionality."""
        config = CacheConfig(
            redis_url="redis://localhost:6379/0",
            search_results_ttl=300,
            enable_memory_fallback=True,
            enable_disk_fallback=True
        )
        
        cache_manager = initialize_cache(config)
        
        # Test basic operations
        test_key = "test:key"
        test_value = {"message": "Hello, World!", "timestamp": time.time()}
        
        print("Testing cache operations...")
        
        # Set value
        success = await cache_manager.aset(test_key, test_value, 60)
        print(f"Set operation: {'Success' if success else 'Failed'}")
        
        # Get value
        retrieved_value = await cache_manager.aget(test_key)
        print(f"Get operation: {'Success' if retrieved_value == test_value else 'Failed'}")
        
        # Test search result caching
        search_key = CacheKey.search_results(
            "machine learning",
            {"source": "docs", "language": "en"},
            "hybrid"
        )
        
        search_results = [
            {"title": "ML Guide", "score": 0.95, "url": "https://example.com/ml"}
        ]
        
        await cache_manager.aset(search_key, search_results, 300)
        cached_results = await cache_manager.aget(search_key)
        print(f"Search cache: {'Success' if cached_results == search_results else 'Failed'}")
        
        # Test embedding cache
        embedding_cache = EmbeddingCache(cache_manager)
        test_text = "This is a test sentence for embedding."
        test_embedding = np.random.rand(384)  # Simulate embedding
        
        embedding_cache.set_embedding(test_text, test_embedding)
        retrieved_embedding = embedding_cache.get_embedding(test_text)
        
        print(f"Embedding cache: {'Success' if np.allclose(retrieved_embedding, test_embedding) else 'Failed'}")
        
        # Print statistics
        stats = cache_manager.get_cache_stats()
        print(f"\nCache Statistics:")
        print(f"Hit rate: {stats['performance']['hit_rate']:.2%}")
        print(f"Total requests: {stats['performance']['total_requests']}")
        print(f"Average response time: {stats['performance']['avg_response_time']:.4f}s")
        
        shutdown_cache()
    
    # Run test
    asyncio.run(test_cache())