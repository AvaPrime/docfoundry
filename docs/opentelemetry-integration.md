# OpenTelemetry Integration Guide

This guide explains how to set up and use OpenTelemetry observability features in DocFoundry.

## Overview

DocFoundry includes comprehensive OpenTelemetry integration for:
- **Distributed Tracing**: Track requests across services and components
- **Metrics Collection**: Monitor performance, usage, and system health
- **Logging**: Structured logging with correlation IDs
- **Performance Monitoring**: Track search latency, throughput, and errors

## Quick Start

### 1. Install Dependencies

OpenTelemetry dependencies are included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure OpenTelemetry settings:

```bash
# OpenTelemetry Configuration
OTEL_SERVICE_NAME=docfoundry
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
ENVIRONMENT=development
OTEL_EXPORTER_PROMETHEUS_PORT=9090
OTEL_TRACES_EXPORTER=otlp
OTEL_METRICS_EXPORTER=prometheus,otlp
OTEL_LOGS_EXPORTER=otlp
```

### 3. Start the Application

```bash
python -m uvicorn server.rag_api:app --host 0.0.0.0 --port 8001 --reload
```

OpenTelemetry will automatically instrument the FastAPI application.

## Observability Stack Setup

### Option 1: Jaeger + Prometheus (Recommended)

#### Docker Compose Setup

Create `docker-compose.observability.yml`:

```yaml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14250:14250"  # gRPC
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'docfoundry'
    static_configs:
      - targets: ['host.docker.internal:9090']
```

Start the observability stack:

```bash
docker-compose -f docker-compose.observability.yml up -d
```

### Option 2: OTEL Collector + Custom Backend

For production environments, use the OpenTelemetry Collector with your preferred backend (DataDog, New Relic, etc.).

## Metrics and Traces

### Automatic Instrumentation

DocFoundry automatically instruments:

#### HTTP Requests
- Request duration and status codes
- Endpoint-specific metrics
- Error rates and types

#### Search Operations
- **Semantic Search**: Query processing time, result counts, similarity scores
- **Hybrid Search**: RRF processing, component search times
- **Search Errors**: Failed queries with error types

#### User Interactions
- **Click Feedback**: User engagement metrics, dwell time
- **Session Tracking**: User behavior patterns

#### Job Processing
- **Ingestion Jobs**: Document processing metrics, queue times
- **Crawling**: URL processing rates, success/failure ratios

### Custom Metrics

Key metrics collected:

```
# Search Metrics
search_requests_total{type="semantic|hybrid"}
search_requests_success{type="semantic|hybrid"}
search_requests_error{type="semantic|hybrid", error="ErrorType"}
search_results_count{type="semantic|hybrid"}

# Feedback Metrics
feedback_clicks_total
feedback_clicks_success
feedback_clicks_error{error="ErrorType"}
feedback_dwell_time

# Job Metrics
ingest_jobs_total
ingest_jobs_enqueued
ingest_jobs_error{error="ErrorType"}
ingest_url_count
```

### Trace Spans

Key trace spans:

- `semantic_search`: Vector similarity search operations
- `hybrid_search`: Combined search with RRF
- `log_click_feedback`: User interaction tracking
- `ingest_job_enqueue`: Document processing jobs

## Monitoring Dashboards

### Grafana Dashboard

Import the DocFoundry dashboard (create `grafana-dashboard.json`):

```json
{
  "dashboard": {
    "title": "DocFoundry Observability",
    "panels": [
      {
        "title": "Search Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(search_requests_total[5m])",
            "legendFormat": "{{type}} searches/sec"
          }
        ]
      },
      {
        "title": "Search Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{endpoint=~\"/search.*\"}[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(search_requests_error[5m])",
            "legendFormat": "{{type}} errors/sec"
          }
        ]
      }
    ]
  }
}
```

### Key Metrics to Monitor

1. **Performance**:
   - Search latency (p50, p95, p99)
   - Request throughput
   - Database query times

2. **Reliability**:
   - Error rates by endpoint
   - Success rates for search types
   - Job processing failures

3. **Usage**:
   - Search volume by type
   - User engagement (clicks, dwell time)
   - Popular queries and results

## Alerting

### Prometheus Alerts

Create `alerts.yml`:

```yaml
groups:
  - name: docfoundry
    rules:
      - alert: HighErrorRate
        expr: rate(search_requests_error[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in DocFoundry search"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighSearchLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{endpoint=~"/search.*"}[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High search latency in DocFoundry"
          description: "95th percentile latency is {{ $value }} seconds"

      - alert: LowSearchSuccessRate
        expr: rate(search_requests_success[5m]) / rate(search_requests_total[5m]) < 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low search success rate"
          description: "Success rate is {{ $value | humanizePercentage }}"
```

## Development and Debugging

### Local Development

For local development without external collectors:

```bash
# Disable OTLP export, use console only
export OTEL_TRACES_EXPORTER=console
export OTEL_METRICS_EXPORTER=console
export OTEL_LOGS_EXPORTER=console
```

### Trace Debugging

View traces in Jaeger UI:
1. Open http://localhost:16686
2. Select "docfoundry" service
3. Search for traces by operation or tags

### Custom Instrumentation

Add custom spans in your code:

```python
from observability.telemetry import trace_span, record_counter

@trace_function
async def my_function():
    with trace_span("custom_operation") as span:
        span.set_attribute("custom_attr", "value")
        # Your code here
        record_counter("custom_metric", {"label": "value"})
```

## Production Considerations

### Performance Impact

- Tracing overhead: ~1-5% CPU
- Memory overhead: ~10-50MB
- Network overhead: Depends on sampling rate

### Sampling Configuration

```bash
# Sample 10% of traces in production
export OTEL_TRACES_SAMPLER=traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1
```

### Security

- Use HTTPS for OTLP endpoints in production
- Configure authentication headers:
  ```bash
  export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer YOUR_TOKEN"
  ```

### Resource Limits

```bash
# Limit resource usage
export OTEL_BSP_MAX_QUEUE_SIZE=2048
export OTEL_BSP_EXPORT_BATCH_SIZE=512
export OTEL_METRIC_EXPORT_INTERVAL=30000
```

## Troubleshooting

### Common Issues

1. **No traces appearing**:
   - Check OTLP endpoint connectivity
   - Verify service name configuration
   - Check firewall/network settings

2. **High memory usage**:
   - Reduce batch sizes
   - Increase export intervals
   - Enable sampling

3. **Missing metrics**:
   - Verify Prometheus scraping configuration
   - Check metric export format
   - Ensure proper labeling

### Debug Logging

```bash
# Enable OpenTelemetry debug logging
export OTEL_LOG_LEVEL=debug
export OTEL_PYTHON_LOG_CORRELATION=true
```

## Integration with External Services

### DataDog

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.datadoghq.com
export OTEL_EXPORTER_OTLP_HEADERS="dd-api-key=YOUR_API_KEY"
```

### New Relic

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.nr-data.net:4317
export OTEL_EXPORTER_OTLP_HEADERS="api-key=YOUR_LICENSE_KEY"
```

### AWS X-Ray

```bash
export OTEL_TRACES_EXPORTER=xray
export AWS_REGION=us-west-2
```

This comprehensive observability setup provides deep insights into DocFoundry's performance, reliability, and usage patterns, enabling data-driven optimization and proactive issue resolution.