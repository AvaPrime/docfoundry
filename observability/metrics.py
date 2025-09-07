from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time
import threading
from .telemetry import get_meter, record_counter, record_histogram, record_gauge

@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    attributes: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Base class for collecting and aggregating metrics."""
    
    def __init__(self, name: str, retention_period: timedelta = timedelta(hours=24)):
        self.name = name
        self.retention_period = retention_period
        self.data_points: deque[MetricPoint] = deque()
        self._lock = threading.Lock()
    
    def record(self, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric value."""
        with self._lock:
            point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                attributes=attributes or {}
            )
            self.data_points.append(point)
            self._cleanup_old_data()
    
    def _cleanup_old_data(self) -> None:
        """Remove data points older than retention period."""
        cutoff = datetime.utcnow() - self.retention_period
        while self.data_points and self.data_points[0].timestamp < cutoff:
            self.data_points.popleft()
    
    def get_recent_points(self, duration: timedelta = timedelta(minutes=5)) -> List[MetricPoint]:
        """Get data points from the last duration."""
        cutoff = datetime.utcnow() - duration
        with self._lock:
            return [point for point in self.data_points if point.timestamp >= cutoff]
    
    def get_average(self, duration: timedelta = timedelta(minutes=5)) -> float:
        """Get average value over the last duration."""
        points = self.get_recent_points(duration)
        if not points:
            return 0.0
        return sum(point.value for point in points) / len(points)
    
    def get_sum(self, duration: timedelta = timedelta(minutes=5)) -> float:
        """Get sum of values over the last duration."""
        points = self.get_recent_points(duration)
        return sum(point.value for point in points)
    
    def get_count(self, duration: timedelta = timedelta(minutes=5)) -> int:
        """Get count of data points over the last duration."""
        return len(self.get_recent_points(duration))

class SearchMetrics:
    """Metrics collector for search operations."""
    
    def __init__(self):
        self.query_duration = MetricsCollector("search.query_duration")
        self.query_count = MetricsCollector("search.query_count")
        self.result_count = MetricsCollector("search.result_count")
        self.embedding_duration = MetricsCollector("search.embedding_duration")
        self.error_count = MetricsCollector("search.error_count")
        
        # Query type counters
        self.query_types = defaultdict(int)
        self._lock = threading.Lock()
    
    def record_search_query(
        self,
        query_type: str,
        duration: float,
        result_count: int,
        embedding_duration: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Record a search query with all relevant metrics."""
        attributes = {"query_type": query_type}
        
        # Record core metrics
        self.query_duration.record(duration, attributes)
        self.query_count.record(1.0, attributes)
        self.result_count.record(result_count, attributes)
        
        # Record embedding duration if provided
        if embedding_duration is not None:
            self.embedding_duration.record(embedding_duration, attributes)
        
        # Record error if occurred
        if error:
            error_attrs = {**attributes, "error_type": error}
            self.error_count.record(1.0, error_attrs)
        
        # Update query type counter
        with self._lock:
            self.query_types[query_type] += 1
        
        # Send to OpenTelemetry
        record_histogram("docfoundry.search.duration", duration, attributes)
        record_counter("docfoundry.search.queries", 1.0, attributes)
        record_histogram("docfoundry.search.results", result_count, attributes)
        
        if embedding_duration is not None:
            record_histogram("docfoundry.search.embedding_duration", embedding_duration, attributes)
        
        if error:
            record_counter("docfoundry.search.errors", 1.0, error_attrs)
    
    def get_query_stats(self, duration: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """Get aggregated query statistics."""
        return {
            "total_queries": self.query_count.get_sum(duration),
            "avg_duration": self.query_duration.get_average(duration),
            "avg_results": self.result_count.get_average(duration),
            "avg_embedding_duration": self.embedding_duration.get_average(duration),
            "error_rate": self.error_count.get_sum(duration) / max(1, self.query_count.get_sum(duration)),
            "queries_per_minute": self.query_count.get_sum(timedelta(minutes=1))
        }
    
    def get_query_type_distribution(self) -> Dict[str, int]:
        """Get distribution of query types."""
        with self._lock:
            return dict(self.query_types)

class IndexingMetrics:
    """Metrics collector for document indexing operations."""
    
    def __init__(self):
        self.document_count = MetricsCollector("indexing.document_count")
        self.chunk_count = MetricsCollector("indexing.chunk_count")
        self.indexing_duration = MetricsCollector("indexing.duration")
        self.embedding_duration = MetricsCollector("indexing.embedding_duration")
        self.error_count = MetricsCollector("indexing.error_count")
        self.document_size = MetricsCollector("indexing.document_size")
        
        # Document type counters
        self.document_types = defaultdict(int)
        self._lock = threading.Lock()
    
    def record_document_indexed(
        self,
        doc_type: str,
        chunk_count: int,
        document_size: int,
        indexing_duration: float,
        embedding_duration: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Record a document indexing operation."""
        attributes = {"doc_type": doc_type}
        
        # Record core metrics
        self.document_count.record(1.0, attributes)
        self.chunk_count.record(chunk_count, attributes)
        self.document_size.record(document_size, attributes)
        self.indexing_duration.record(indexing_duration, attributes)
        
        # Record embedding duration if provided
        if embedding_duration is not None:
            self.embedding_duration.record(embedding_duration, attributes)
        
        # Record error if occurred
        if error:
            error_attrs = {**attributes, "error_type": error}
            self.error_count.record(1.0, error_attrs)
        
        # Update document type counter
        with self._lock:
            self.document_types[doc_type] += 1
        
        # Send to OpenTelemetry
        record_counter("docfoundry.indexing.documents", 1.0, attributes)
        record_histogram("docfoundry.indexing.chunks", chunk_count, attributes)
        record_histogram("docfoundry.indexing.document_size", document_size, attributes)
        record_histogram("docfoundry.indexing.duration", indexing_duration, attributes)
        
        if embedding_duration is not None:
            record_histogram("docfoundry.indexing.embedding_duration", embedding_duration, attributes)
        
        if error:
            record_counter("docfoundry.indexing.errors", 1.0, error_attrs)
    
    def get_indexing_stats(self, duration: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """Get aggregated indexing statistics."""
        return {
            "total_documents": self.document_count.get_sum(duration),
            "total_chunks": self.chunk_count.get_sum(duration),
            "avg_chunks_per_doc": self.chunk_count.get_average(duration),
            "avg_document_size": self.document_size.get_average(duration),
            "avg_indexing_duration": self.indexing_duration.get_average(duration),
            "avg_embedding_duration": self.embedding_duration.get_average(duration),
            "error_rate": self.error_count.get_sum(duration) / max(1, self.document_count.get_sum(duration)),
            "documents_per_minute": self.document_count.get_sum(timedelta(minutes=1))
        }
    
    def get_document_type_distribution(self) -> Dict[str, int]:
        """Get distribution of document types."""
        with self._lock:
            return dict(self.document_types)

class APIMetrics:
    """Metrics collector for API operations."""
    
    def __init__(self):
        self.request_count = MetricsCollector("api.request_count")
        self.request_duration = MetricsCollector("api.request_duration")
        self.response_size = MetricsCollector("api.response_size")
        self.error_count = MetricsCollector("api.error_count")
        
        # Endpoint and status code counters
        self.endpoints = defaultdict(int)
        self.status_codes = defaultdict(int)
        self._lock = threading.Lock()
    
    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        response_size: Optional[int] = None,
        error: Optional[str] = None
    ) -> None:
        """Record an API request."""
        attributes = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        }
        
        # Record core metrics
        self.request_count.record(1.0, attributes)
        self.request_duration.record(duration, attributes)
        
        if response_size is not None:
            self.response_size.record(response_size, attributes)
        
        # Record error if occurred
        if error or status_code >= 400:
            error_attrs = {**attributes}
            if error:
                error_attrs["error_type"] = error
            self.error_count.record(1.0, error_attrs)
        
        # Update counters
        with self._lock:
            self.endpoints[f"{method} {endpoint}"] += 1
            self.status_codes[status_code] += 1
        
        # Send to OpenTelemetry
        record_counter("docfoundry.api.requests", 1.0, attributes)
        record_histogram("docfoundry.api.duration", duration, attributes)
        
        if response_size is not None:
            record_histogram("docfoundry.api.response_size", response_size, attributes)
        
        if error or status_code >= 400:
            record_counter("docfoundry.api.errors", 1.0, error_attrs)
    
    def get_api_stats(self, duration: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """Get aggregated API statistics."""
        total_requests = self.request_count.get_sum(duration)
        return {
            "total_requests": total_requests,
            "avg_duration": self.request_duration.get_average(duration),
            "avg_response_size": self.response_size.get_average(duration),
            "error_rate": self.error_count.get_sum(duration) / max(1, total_requests),
            "requests_per_minute": self.request_count.get_sum(timedelta(minutes=1))
        }
    
    def get_endpoint_distribution(self) -> Dict[str, int]:
        """Get distribution of API endpoints."""
        with self._lock:
            return dict(self.endpoints)
    
    def get_status_code_distribution(self) -> Dict[int, int]:
        """Get distribution of HTTP status codes."""
        with self._lock:
            return dict(self.status_codes)

class SystemMetrics:
    """Metrics collector for system-level metrics."""
    
    def __init__(self):
        self.database_connections = MetricsCollector("system.db_connections")
        self.memory_usage = MetricsCollector("system.memory_usage")
        self.cpu_usage = MetricsCollector("system.cpu_usage")
        self.disk_usage = MetricsCollector("system.disk_usage")
    
    def record_system_stats(
        self,
        db_connections: Optional[int] = None,
        memory_usage_mb: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        disk_usage_mb: Optional[float] = None
    ) -> None:
        """Record system statistics."""
        if db_connections is not None:
            self.database_connections.record(db_connections)
            record_gauge("docfoundry.system.db_connections", db_connections)
        
        if memory_usage_mb is not None:
            self.memory_usage.record(memory_usage_mb)
            record_gauge("docfoundry.system.memory_usage_mb", memory_usage_mb)
        
        if cpu_percent is not None:
            self.cpu_usage.record(cpu_percent)
            record_gauge("docfoundry.system.cpu_percent", cpu_percent)
        
        if disk_usage_mb is not None:
            self.disk_usage.record(disk_usage_mb)
            record_gauge("docfoundry.system.disk_usage_mb", disk_usage_mb)

# Global metrics instances
search_metrics = SearchMetrics()
indexing_metrics = IndexingMetrics()
api_metrics = APIMetrics()
system_metrics = SystemMetrics()

def get_all_metrics_summary(duration: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
    """Get a summary of all metrics."""
    return {
        "search": search_metrics.get_query_stats(duration),
        "indexing": indexing_metrics.get_indexing_stats(duration),
        "api": api_metrics.get_api_stats(duration),
        "query_types": search_metrics.get_query_type_distribution(),
        "document_types": indexing_metrics.get_document_type_distribution(),
        "endpoints": api_metrics.get_endpoint_distribution(),
        "status_codes": api_metrics.get_status_code_distribution()
    }

# Example usage
if __name__ == "__main__":
    # Example search metrics
    search_metrics.record_search_query(
        query_type="semantic",
        duration=0.15,
        result_count=5,
        embedding_duration=0.05
    )
    
    # Example indexing metrics
    indexing_metrics.record_document_indexed(
        doc_type="markdown",
        chunk_count=10,
        document_size=5000,
        indexing_duration=2.5,
        embedding_duration=1.2
    )
    
    # Example API metrics
    api_metrics.record_api_request(
        endpoint="/search/semantic",
        method="POST",
        status_code=200,
        duration=0.2,
        response_size=1500
    )
    
    # Get summary
    summary = get_all_metrics_summary()
    print("Metrics Summary:")
    for category, stats in summary.items():
        print(f"  {category}: {stats}")