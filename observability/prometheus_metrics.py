"""Prometheus metrics integration for DocFoundry API."""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
import time
from typing import Optional, Dict, Any
import logging
import psutil
import os

logger = logging.getLogger(__name__)

# Create custom registry for DocFoundry metrics
docfoundry_registry = CollectorRegistry()

# Request metrics
request_count = Counter(
    'docfoundry_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=docfoundry_registry
)

request_duration = Histogram(
    'docfoundry_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=docfoundry_registry
)

request_size = Histogram(
    'docfoundry_http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000],
    registry=docfoundry_registry
)

response_size = Histogram(
    'docfoundry_http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000],
    registry=docfoundry_registry
)

# Search metrics
search_requests = Counter(
    'docfoundry_search_requests_total',
    'Total number of search requests',
    ['search_type', 'status'],
    registry=docfoundry_registry
)

search_duration = Histogram(
    'docfoundry_search_duration_seconds',
    'Search request duration in seconds',
    ['search_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=docfoundry_registry
)

search_results_count = Histogram(
    'docfoundry_search_results_count',
    'Number of search results returned',
    ['search_type'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
    registry=docfoundry_registry
)

embedding_duration = Histogram(
    'docfoundry_embedding_duration_seconds',
    'Embedding generation duration in seconds',
    ['model'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=docfoundry_registry
)

# Database metrics
db_connections_active = Gauge(
    'docfoundry_db_connections_active',
    'Number of active database connections',
    registry=docfoundry_registry
)

db_query_duration = Histogram(
    'docfoundry_db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=docfoundry_registry
)

db_query_count = Counter(
    'docfoundry_db_queries_total',
    'Total number of database queries',
    ['query_type', 'status'],
    registry=docfoundry_registry
)

# Indexing metrics
indexing_documents = Counter(
    'docfoundry_indexing_documents_total',
    'Total number of documents indexed',
    ['document_type', 'status'],
    registry=docfoundry_registry
)

indexing_duration = Histogram(
    'docfoundry_indexing_duration_seconds',
    'Document indexing duration in seconds',
    ['document_type'],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
    registry=docfoundry_registry
)

indexing_chunks = Histogram(
    'docfoundry_indexing_chunks_count',
    'Number of chunks created per document',
    ['document_type'],
    buckets=[1, 5, 10, 25, 50, 100, 250],
    registry=docfoundry_registry
)

# System metrics
system_memory_usage = Gauge(
    'docfoundry_system_memory_usage_bytes',
    'System memory usage in bytes',
    registry=docfoundry_registry
)

system_cpu_usage = Gauge(
    'docfoundry_system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=docfoundry_registry
)

system_disk_usage = Gauge(
    'docfoundry_system_disk_usage_bytes',
    'System disk usage in bytes',
    registry=docfoundry_registry
)

# Application info
app_info = Info(
    'docfoundry_app_info',
    'DocFoundry application information',
    registry=docfoundry_registry
)

# Error metrics
error_count = Counter(
    'docfoundry_errors_total',
    'Total number of errors',
    ['error_type', 'component'],
    registry=docfoundry_registry
)

class PrometheusMiddleware:
    """Middleware to collect Prometheus metrics for HTTP requests."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        method = request.method
        path = request.url.path
        
        # Normalize endpoint path (remove IDs, etc.)
        endpoint = self._normalize_endpoint(path)
        
        start_time = time.time()
        
        # Get request size
        request_body_size = 0
        if "content-length" in request.headers:
            try:
                request_body_size = int(request.headers["content-length"])
            except ValueError:
                pass
        
        # Track request size
        request_size.labels(method=method, endpoint=endpoint).observe(request_body_size)
        
        # Process request
        response_body_size = 0
        status_code = 500  # Default to error
        
        async def send_wrapper(message):
            nonlocal response_body_size, status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                response_body_size += len(body)
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            error_count.labels(error_type=type(e).__name__, component="http").inc()
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            request_count.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            
            request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            response_size.labels(
                method=method,
                endpoint=endpoint
            ).observe(response_body_size)
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path to reduce cardinality."""
        # Replace UUIDs and IDs with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace hash-like strings
        path = re.sub(r'/[a-f0-9]{32,}', '/{hash}', path)
        
        return path

def setup_prometheus_metrics(app: FastAPI) -> None:
    """Setup Prometheus metrics collection for FastAPI app."""
    # Add middleware
    app.add_middleware(PrometheusMiddleware)
    
    # Add metrics endpoint
    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        return generate_latest(docfoundry_registry)
    
    # Set application info
    app_info.info({
        'version': os.getenv('APP_VERSION', 'unknown'),
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'build_date': os.getenv('BUILD_DATE', 'unknown')
    })
    
    logger.info("Prometheus metrics configured")

def record_search_metrics(search_type: str, duration: float, result_count: int, 
                         embedding_time: Optional[float] = None, error: Optional[str] = None) -> None:
    """Record search-related metrics."""
    status = "error" if error else "success"
    
    search_requests.labels(search_type=search_type, status=status).inc()
    search_duration.labels(search_type=search_type).observe(duration)
    
    if not error:
        search_results_count.labels(search_type=search_type).observe(result_count)
    
    if embedding_time is not None:
        embedding_duration.labels(model="default").observe(embedding_time)
    
    if error:
        error_count.labels(error_type="search_error", component="search").inc()

def record_indexing_metrics(doc_type: str, duration: float, chunk_count: int, 
                           error: Optional[str] = None) -> None:
    """Record indexing-related metrics."""
    status = "error" if error else "success"
    
    indexing_documents.labels(document_type=doc_type, status=status).inc()
    
    if not error:
        indexing_duration.labels(document_type=doc_type).observe(duration)
        indexing_chunks.labels(document_type=doc_type).observe(chunk_count)
    
    if error:
        error_count.labels(error_type="indexing_error", component="indexing").inc()

def record_db_metrics(query_type: str, duration: float, error: Optional[str] = None) -> None:
    """Record database-related metrics."""
    status = "error" if error else "success"
    
    db_query_count.labels(query_type=query_type, status=status).inc()
    
    if not error:
        db_query_duration.labels(query_type=query_type).observe(duration)
    
    if error:
        error_count.labels(error_type="db_error", component="database").inc()

def update_system_metrics() -> None:
    """Update system-level metrics."""
    try:
        # Memory usage
        memory = psutil.virtual_memory()
        system_memory_usage.set(memory.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        system_cpu_usage.set(cpu_percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        system_disk_usage.set(disk.used)
        
    except Exception as e:
        logger.error(f"Error updating system metrics: {e}")
        error_count.labels(error_type="system_metrics_error", component="monitoring").inc()

def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of current metrics."""
    try:
        return {
            "requests_total": request_count._value.sum(),
            "search_requests_total": search_requests._value.sum(),
            "indexing_documents_total": indexing_documents._value.sum(),
            "errors_total": error_count._value.sum(),
            "db_connections_active": db_connections_active._value._value,
            "memory_usage_bytes": system_memory_usage._value._value,
            "cpu_usage_percent": system_cpu_usage._value._value
        }
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        return {}