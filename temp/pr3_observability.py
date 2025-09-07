# ============================================================================
# PR-3: Observability Foundation
# Files: Metrics endpoints + Prometheus rules + Alertmanager config
# ============================================================================

# FILE: services/api/metrics.py (API metrics)

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import time
from functools import wraps
from fastapi import Request
import asyncio

# API Metrics
REQUEST_COUNT = Counter(
    'docfoundry_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'docfoundry_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

SEARCH_COUNT = Counter(
    'docfoundry_searches_total',
    'Total search requests',
    ['site', 'search_type']
)

SEARCH_RESULTS = Histogram(
    'docfoundry_search_results_count',
    'Number of search results returned',
    ['search_type']
)

SEARCH_LATENCY = Histogram(
    'docfoundry_search_latency_seconds',
    'Search operation latency',
    ['search_type'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

ACTIVE_CONNECTIONS = Gauge(
    'docfoundry_active_connections',
    'Number of active connections'
)

DATABASE_CONNECTIONS = Gauge(
    'docfoundry_db_connections_active',
    'Active database connections'
)

DOCUMENT_COUNT = Gauge(
    'docfoundry_documents_total',
    'Total number of documents in index'
)

CHUNK_COUNT = Gauge(
    'docfoundry_chunks_total',
    'Total number of chunks in index'
)

# Application info
APP_INFO = Info(
    'docfoundry_app_info',
    'Application information'
)

# Set application info
APP_INFO.info({
    'version': '1.0.0',
    'python_version': '3.11',
    'environment': 'production'
})

def track_request_metrics(func):
    """Decorator to track request metrics"""
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        start_time = time.time()
        method = request.method
        endpoint = request.url.path
        
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await func(request, *args, **kwargs)
            status = getattr(response, 'status_code', 200)
            
            # Track request count and duration
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(
                time.time() - start_time
            )
            
            return response
            
        except Exception as e:
            # Track errors
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=500).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(
                time.time() - start_time
            )
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()
    
    return wrapper

async def update_document_metrics(db_session):
    """Update document and chunk count metrics"""
    try:
        from sqlalchemy import text
        
        # Get document count
        doc_result = await db_session.execute(text("SELECT COUNT(*) FROM documents"))
        doc_count = doc_result.scalar()
        DOCUMENT_COUNT.set(doc_count)
        
        # Get chunk count
        chunk_result = await db_session.execute(text("SELECT COUNT(*) FROM chunks"))
        chunk_count = chunk_result.scalar()
        CHUNK_COUNT.set(chunk_count)
        
    except Exception as e:
        import logging
        logging.error(f"Failed to update document metrics: {e}")

async def update_db_connection_metrics(engine):
    """Update database connection metrics"""
    try:
        pool = engine.pool
        DATABASE_CONNECTIONS.set(pool.checkedout())
    except Exception:
        pass

# ============================================================================
# FILE: services/api/app.py (metrics integration)

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from services.api.metrics import (
    track_request_metrics, 
    update_document_metrics,
    SEARCH_COUNT,
    SEARCH_RESULTS,
    SEARCH_LATENCY
)
import time

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}

# Enhanced endpoints with metrics
@app.post("/query")
@limiter.limit(QUERY_RATE_LIMIT)
@track_request_metrics
async def query_documents_with_metrics(
    request: Request,
    query_request: QueryRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """Query documents with metrics tracking"""
    start_time = time.time()
    site = getattr(query_request, 'site', None)
    
    try:
        # Your existing query logic here
        results = await perform_search(query_request, db)
        
        # Track metrics
        SEARCH_COUNT.labels(site=site or 'all', search_type='hybrid').inc()
        SEARCH_RESULTS.labels(search_type='hybrid').observe(len(results))
        SEARCH_LATENCY.labels(search_type='hybrid').observe(time.time() - start_time)
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        SEARCH_LATENCY.labels(search_type='hybrid').observe(time.time() - start_time)
        raise

# Background task to update metrics
@app.on_event("startup")
async def startup_event():
    """Start background metrics collection"""
    async def metrics_collector():
        while True:
            try:
                async with AsyncSessionLocal() as session:
                    await update_document_metrics(session)
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                import logging
                logging.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    asyncio.create_task(metrics_collector())

# ============================================================================
# FILE: services/worker/metrics.py (Worker metrics)

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import os

# Worker Metrics
CRAWLED_PAGES = Counter(
    'docfoundry_crawled_pages_total',
    'Total pages crawled',
    ['site', 'status']
)

CRAWL_DURATION = Histogram(
    'docfoundry_crawl_duration_seconds',
    'Time spent crawling pages',
    ['site']
)

PROCESSING_DURATION = Histogram(
    'docfoundry_processing_duration_seconds',
    'Document processing time',
    ['operation']
)

PROCESSING_QUEUE_SIZE = Gauge(
    'docfoundry_processing_queue_length',
    'Number of documents in processing queue'
)

EMBEDDING_OPERATIONS = Counter(
    'docfoundry_embedding_operations_total',
    'Total embedding operations',
    ['model', 'status']
)

CHUNK_GENERATION = Counter(
    'docfoundry_chunks_generated_total',
    'Total chunks generated',
    ['document_type']
)

OCR_OPERATIONS = Counter(
    'docfoundry_ocr_operations_total',
    'Total OCR operations',
    ['status']
)

WORKER_ERRORS = Counter(
    'docfoundry_worker_errors_total',
    'Total worker errors',
    ['error_type']
)

def start_worker_metrics_server():
    """Start metrics server for worker"""
    port = int(os.getenv("WORKER_METRICS_PORT", "9108"))
    start_http_server(port)
    print(f"Worker metrics server started on port {port}")

def track_crawl_metrics(func):
    """Decorator to track crawling metrics"""
    def wrapper(url: str, *args, **kwargs):
        start_time = time.time()
        site = extract_site_from_url(url)
        
        try:
            result = func(url, *args, **kwargs)
            status = "success" if result else "failed"
            
            CRAWLED_PAGES.labels(site=site, status=status).inc()
            CRAWL_DURATION.labels(site=site).observe(time.time() - start_time)
            
            return result
            
        except Exception as e:
            CRAWLED_PAGES.labels(site=site, status="error").inc()
            CRAWL_DURATION.labels(site=site).observe(time.time() - start_time)
            WORKER_ERRORS.labels(error_type="crawl_error").inc()
            raise
    
    return wrapper

def track_processing_metrics(operation: str):
    """Decorator to track document processing metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                PROCESSING_DURATION.labels(operation=operation).observe(
                    time.time() - start_time
                )
                return result
                
            except Exception as e:
                PROCESSING_DURATION.labels(operation=operation).observe(
                    time.time() - start_time
                )
                WORKER_ERRORS.labels(error_type=f"{operation}_error").inc()
                raise
        
        return wrapper
    return decorator

def extract_site_from_url(url: str) -> str:
    """Extract site identifier from URL"""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.netloc or "unknown"

# ============================================================================
# FILE: ops/prometheus/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'docfoundry-api'
    static_configs:
      - targets: ['api:8080']
    metrics_path: '/metrics' # Ensure this path is correct for your FastAPI app
    scrape_interval: 10s

  - job_name: 'docfoundry-worker'
    static_configs:
      - targets: ['worker:9108']
    scrape_interval: 15s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 30s

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s

# ============================================================================
# FILE: ops/prometheus/rules/docfoundry-rules.yml

groups:
  - name: docfoundry-performance
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(docfoundry_request_duration_seconds_bucket[5m])) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "95th percentile latency is {{ $value }}s"

      - alert: VeryHighLatency
        expr: histogram_quantile(0.95, rate(docfoundry_request_duration_seconds_bucket[5m])) > 5
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Very high API latency detected"
          description: "95th percentile latency is {{ $value }}s"

      - alert: SlowSearchQueries
        expr: histogram_quantile(0.95, rate(docfoundry_search_latency_seconds_bucket[5m])) > 3
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Slow search queries detected"
          description: "95th percentile search latency is {{ $value }}s"

  - name: docfoundry-errors
    rules:
      - alert: HighErrorRate
        expr: rate(docfoundry_requests_total{status=~"5.."}[5m]) / rate(docfoundry_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: CriticalErrorRate
        expr: rate(docfoundry_requests_total{status=~"5.."}[5m]) / rate(docfoundry_requests_total[5m]) > 0.20
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Critical error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: WorkerErrors
        expr: rate(docfoundry_worker_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High worker error rate"
          description: "Worker error rate is {{ $value }} errors/sec"

  - name: docfoundry-resources
    rules:
      - alert: HighDatabaseConnections
        expr: docfoundry_db_connections_active > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High database connection usage"
          description: "Active connections: {{ $value }}"

      - alert: LargeProcessingQueue
        expr: docfoundry_processing_queue_length > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Large processing queue"
          description: "Queue length: {{ $value }}"

      - alert: ProcessingQueueStuck
        expr: changes(docfoundry_processing_queue_length[15m]) == 0 and docfoundry_processing_queue_length > 10
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Processing queue appears stuck"
          description: "Queue length unchanged for 15m: {{ $value }}"

  - name: docfoundry-business
    rules:
      - alert: LowSearchVolume
        expr: rate(docfoundry_searches_total[1h]) < 0.01
        for: 30m
        labels:
          severity: info
        annotations:
          summary: "Low search volume"
          description: "Search rate: {{ $value }} searches/sec"

      - alert: NoRecentDocuments
        expr: changes(docfoundry_documents_total[24h]) == 0
        for: 4h
        labels:
          severity: info
        annotations:
          summary: "No new documents indexed recently"
          description: "Document count unchanged for 24h"

# ============================================================================
# FILE: ops/alertmanager/alertmanager.yml

global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@yourdomain.com'

route:
  group_by: ['alertname']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'
    - match:
        severity: info
      receiver: 'info-alerts'

receivers:
  - name: 'default'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts' # Default channel for all alerts
        title: 'DocFoundry Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#critical-alerts'
        title: '🚨 CRITICAL: DocFoundry Alert' # Dedicated channel for critical alerts
        text: '{{ range .Alerts }}{{ .Annotations.summary }}: {{ .Annotations.description }}{{ end }}'
    email_configs:
      - to: 'oncall@yourdomain.com'
        subject: 'CRITICAL: DocFoundry Alert'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}

  - name: 'warning-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts' # Warnings can go to the general alerts channel
        title: '⚠️ WARNING: DocFoundry Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'info-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#monitoring' # Info alerts can go to a monitoring channel
        title: 'ℹ️ INFO: DocFoundry Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']