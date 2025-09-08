# DocFoundry Observability & Monitoring Setup Guide

This guide covers comprehensive observability and monitoring setup for DocFoundry, including metrics collection, logging, tracing, alerting, and performance monitoring.

## Table of Contents

1. [Observability Overview](#observability-overview)
2. [Metrics Collection](#metrics-collection)
3. [Logging Configuration](#logging-configuration)
4. [Distributed Tracing](#distributed-tracing)
5. [Performance Monitoring](#performance-monitoring)
6. [Alerting & Notifications](#alerting--notifications)
7. [Dashboards & Visualization](#dashboards--visualization)
8. [Health Checks](#health-checks)
9. [SLA & SLO Monitoring](#sla--slo-monitoring)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

## Observability Overview

DocFoundry implements comprehensive observability through the three pillars:

- **Metrics**: Quantitative measurements of system behavior
- **Logs**: Detailed event records for debugging and auditing
- **Traces**: Request flow tracking across distributed components

### Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DocFoundry    │───▶│   Prometheus    │───▶│     Grafana     │
│   Application   │    │    (Metrics)    │    │  (Dashboards)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Elasticsearch │    │     Jaeger      │    │   AlertManager  │
│     (Logs)      │    │    (Traces)     │    │   (Alerting)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Metrics Collection

### 1. Prometheus Configuration

#### Prometheus Setup

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "docfoundry_rules.yml"
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'docfoundry-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'docfoundry-worker'
    static_configs:
      - targets: ['worker:8001']
    metrics_path: '/metrics'
    scrape_interval: 15s
    
  - job_name: 'docfoundry-db'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 30s
    
  - job_name: 'docfoundry-redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s
```

#### Application Metrics Configuration

```bash
# Metrics Configuration
METRICS_ENABLED=true
METRICS_PORT=9090
METRICS_PATH=/metrics
METRICS_NAMESPACE=docfoundry
METRICS_SUBSYSTEM=api

# Custom Metrics
CUSTOM_METRICS_ENABLED=true
BUSINESS_METRICS_ENABLED=true
PERFORMANCE_METRICS_ENABLED=true
```

### 2. Key Metrics Categories

#### Application Metrics

```python
# Core application metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

database_queries_total = Counter(
    'database_queries_total',
    'Total database queries',
    ['operation', 'table']
)
```

#### Business Metrics

```python
# Business-specific metrics
documents_processed_total = Counter(
    'documents_processed_total',
    'Total documents processed',
    ['type', 'status']
)

lineage_operations_total = Counter(
    'lineage_operations_total',
    'Total lineage operations',
    ['operation_type']
)

user_sessions_active = Gauge(
    'user_sessions_active',
    'Number of active user sessions'
)

api_key_usage_total = Counter(
    'api_key_usage_total',
    'API key usage count',
    ['key_id', 'endpoint']
)
```

## Logging Configuration

### 1. Structured Logging Setup

```bash
# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_STRUCTURED=true
LOG_INCLUDE_TRACE_ID=true
LOG_INCLUDE_SPAN_ID=true

# Log Destinations
LOG_FILE=/var/log/docfoundry/app.log
LOG_STDOUT=true
LOG_ELASTICSEARCH=true

# Log Rotation
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=10
LOG_ROTATION_INTERVAL=daily
```

### 2. ELK Stack Configuration

#### Elasticsearch Setup

```yaml
# docker-compose.yml
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
  container_name: elasticsearch
  environment:
    - discovery.type=single-node
    - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    - xpack.security.enabled=false
  volumes:
    - elasticsearch_data:/usr/share/elasticsearch/data
  ports:
    - "9200:9200"
  networks:
    - docfoundry-network
```

#### Logstash Configuration

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
  
  file {
    path => "/var/log/docfoundry/*.log"
    start_position => "beginning"
    codec => "json"
  }
}

filter {
  if [fields][service] == "docfoundry" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    mutate {
      add_field => { "service" => "docfoundry" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "docfoundry-logs-%{+YYYY.MM.dd}"
  }
  
  stdout {
    codec => rubydebug
  }
}
```

## Distributed Tracing

### 1. OpenTelemetry Configuration

```bash
# OpenTelemetry Settings
OTEL_ENABLED=true
OTEL_SERVICE_NAME=docfoundry
OTEL_SERVICE_VERSION=1.0.0
OTEL_ENVIRONMENT=production

# Tracing Configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:14268/api/traces
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_TRACE_SAMPLING_RATE=0.1

# Instrumentation
OTEL_TRACE_INCLUDE_DB=true
OTEL_TRACE_INCLUDE_HTTP=true
OTEL_TRACE_INCLUDE_REDIS=true
```

### 2. Jaeger Setup

```yaml
# docker-compose.yml
jaeger:
  image: jaegertracing/all-in-one:latest
  container_name: jaeger
  environment:
    - COLLECTOR_OTLP_ENABLED=true
  ports:
    - "16686:16686"  # Jaeger UI
    - "14268:14268"  # HTTP collector
    - "14250:14250"  # gRPC collector
  networks:
    - docfoundry-network
```

## Performance Monitoring

### 1. Performance Gates Configuration

```bash
# Performance Monitoring
PERF_MONITORING_ENABLED=true
PERF_RESPONSE_TIME_THRESHOLD=2000  # milliseconds
PERF_ERROR_RATE_THRESHOLD=0.05     # 5%
PERF_THROUGHPUT_THRESHOLD=100      # requests/second

# Performance Gates
PERF_GATES_ENABLED=true
PERF_GATES_FAIL_FAST=true
PERF_GATES_ALERT_WEBHOOK=https://alerts.yourdomain.com/performance
```

### 2. Application Performance Monitoring (APM)

```python
# Performance monitoring middleware
class PerformanceMonitoringMiddleware:
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        start_time = time.time()
        
        # Track request
        request_counter.labels(
            method=scope['method'],
            endpoint=scope['path']
        ).inc()
        
        try:
            await self.app(scope, receive, send)
            status = 'success'
        except Exception as e:
            status = 'error'
            error_counter.labels(
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            # Record duration
            duration = time.time() - start_time
            request_duration.labels(
                method=scope['method'],
                endpoint=scope['path']
            ).observe(duration)
            
            # Check performance gates
            if duration > RESPONSE_TIME_THRESHOLD:
                performance_gate_violations.labels(
                    gate_type='response_time'
                ).inc()
```

## Alerting & Notifications

### 1. AlertManager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@yourdomain.com'
  smtp_auth_username: 'alerts@yourdomain.com'
  smtp_auth_password: 'your-app-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        
  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@yourdomain.com'
        subject: 'CRITICAL: DocFoundry Alert'
        body: |
          Alert: {{ .GroupLabels.alertname }}
          Summary: {{ .CommonAnnotations.summary }}
          Description: {{ .CommonAnnotations.description }}
    webhook_configs:
      - url: 'https://hooks.slack.com/services/YOUR/CRITICAL/WEBHOOK'
        
  - name: 'warning-alerts'
    email_configs:
      - to: 'team@yourdomain.com'
        subject: 'WARNING: DocFoundry Alert'
```

### 2. Alert Rules

```yaml
# alert_rules.yml
groups:
  - name: docfoundry-alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m])) /
            sum(rate(http_requests_total[5m]))
          ) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
          
      # High response time
      - alert: HighResponseTime
        expr: |
          histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"
          
      # Database connection issues
      - alert: DatabaseConnectionHigh
        expr: |
          pg_stat_activity_count > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High database connections"
          description: "Database has {{ $value }} active connections"
```

## Dashboards & Visualization

### 1. Grafana Setup

```yaml
# docker-compose.yml
grafana:
  image: grafana/grafana:latest
  container_name: grafana
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin123
    - GF_USERS_ALLOW_SIGN_UP=false
  volumes:
    - grafana_data:/var/lib/grafana
    - ./grafana/provisioning:/etc/grafana/provisioning
    - ./grafana/dashboards:/var/lib/grafana/dashboards
  ports:
    - "3000:3000"
  networks:
    - docfoundry-network
```

### 2. Key Dashboards

#### Application Overview Dashboard
- Request rate and response times
- Error rates and status codes
- Active users and sessions
- Database query performance
- Cache hit rates

#### Infrastructure Dashboard
- CPU, memory, and disk usage
- Network I/O
- Container resource usage
- Database connections
- Redis performance

#### Business Metrics Dashboard
- Documents processed
- Lineage operations
- API usage by endpoint
- User activity patterns
- Feature adoption metrics

## Health Checks

### 1. Health Check Configuration

```bash
# Health Check Settings
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30  # seconds
HEALTH_CHECK_TIMEOUT=10   # seconds
HEALTH_CHECK_ENDPOINTS=/health,/ready,/live

# Component Health Checks
HEALTH_CHECK_DATABASE=true
HEALTH_CHECK_REDIS=true
HEALTH_CHECK_ELASTICSEARCH=true
HEALTH_CHECK_EXTERNAL_APIS=true
```

### 2. Health Check Implementation

```python
# Health check endpoints
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

app = FastAPI()

class HealthChecker:
    def __init__(self):
        self.checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'elasticsearch': self.check_elasticsearch,
            'disk_space': self.check_disk_space,
            'memory': self.check_memory
        }
        
    async def check_database(self) -> Dict[str, Any]:
        try:
            # Test database connection
            result = await database.fetch_one("SELECT 1")
            return {
                'status': 'healthy',
                'response_time': 0.05,
                'details': 'Database connection successful'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': 'Database connection failed'
            }
            
    async def run_all_checks(self) -> Dict[str, Any]:
        results = {}
        overall_status = 'healthy'
        
        for check_name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[check_name] = result
                
                if result['status'] != 'healthy':
                    overall_status = 'unhealthy'
                    
            except Exception as e:
                results[check_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                overall_status = 'unhealthy'
                
        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }

health_checker = HealthChecker()

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    result = await health_checker.run_all_checks()
    
    if result['status'] != 'healthy':
        raise HTTPException(status_code=503, detail=result)
        
    return result
```

## SLA & SLO Monitoring

### 1. Service Level Objectives (SLOs)

```yaml
# SLO Configuration
slos:
  availability:
    target: 99.9%
    measurement_window: 30d
    
  response_time:
    target: 95%  # of requests under 2s
    threshold: 2000ms
    measurement_window: 7d
    
  error_rate:
    target: 99.5%  # success rate
    measurement_window: 24h
```

### 2. SLO Monitoring Implementation

```python
# SLO monitoring
class SLOMonitor:
    def __init__(self, prometheus_client):
        self.prometheus = prometheus_client
        
    async def calculate_availability_slo(self, window_hours=24):
        """Calculate availability SLO"""
        query = f'''
            (
                sum(rate(http_requests_total{{status!~"5.."}}[{window_hours}h])) /
                sum(rate(http_requests_total[{window_hours}h]))
            ) * 100
        '''
        
        result = await self.prometheus.query(query)
        availability = float(result[0]['value'][1])
        
        return {
            'metric': 'availability',
            'value': availability,
            'target': 99.9,
            'status': 'met' if availability >= 99.9 else 'missed',
            'window': f'{window_hours}h'
        }
```

## Troubleshooting

### 1. Common Monitoring Issues

#### Metrics Not Appearing

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus configuration
prometheus --config.file=prometheus.yml --web.enable-lifecycle
curl -X POST http://localhost:9090/-/reload
```

#### High Cardinality Metrics

```python
# Avoid high cardinality labels
# BAD: user_id as label (could be millions of users)
request_counter.labels(user_id=user_id, endpoint=endpoint).inc()

# GOOD: Use limited set of labels
request_counter.labels(endpoint=endpoint, method=method).inc()
```

### 2. Performance Troubleshooting

#### Slow Queries

```sql
-- Enable slow query logging
SET log_min_duration_statement = 1000;  -- Log queries > 1s

-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

## Best Practices

### 1. Monitoring Strategy

- **Start with the Four Golden Signals**: Latency, Traffic, Errors, Saturation
- **Implement SLIs/SLOs**: Define what good service looks like
- **Use RED Method**: Rate, Errors, Duration for services
- **Apply USE Method**: Utilization, Saturation, Errors for resources
- **Monitor Business Metrics**: Track what matters to users

### 2. Alerting Best Practices

- **Alert on Symptoms, Not Causes**: Alert on user-facing issues
- **Make Alerts Actionable**: Every alert should require human action
- **Avoid Alert Fatigue**: Tune thresholds and use proper grouping
- **Include Context**: Provide runbooks and troubleshooting steps
- **Test Alert Channels**: Regularly verify alert delivery

### 3. Dashboard Design

- **Follow the Inverted Pyramid**: Overview → Detail → Debug
- **Use Consistent Time Ranges**: Align all panels to same time window
- **Include Context**: Add annotations for deployments and incidents
- **Design for Different Audiences**: Executives, engineers, operations
- **Keep It Simple**: Avoid cluttered dashboards

### 4. Performance Optimization

- **Optimize Metric Collection**: Use appropriate scrape intervals
- **Manage Cardinality**: Limit label combinations
- **Use Recording Rules**: Pre-compute expensive queries
- **Implement Retention Policies**: Balance storage and query performance
- **Monitor the Monitors**: Ensure monitoring infrastructure is healthy

### 5. Security Considerations

- **Secure Monitoring Endpoints**: Use authentication and encryption
- **Sanitize Sensitive Data**: Avoid logging secrets or PII
- **Implement Access Controls**: Limit who can view sensitive metrics
- **Audit Monitoring Access**: Track who accesses what data
- **Regular Security Reviews**: Assess monitoring infrastructure security

## Monitoring Checklist

### Initial Setup
- [ ] Prometheus configured and running
- [ ] Grafana dashboards created
- [ ] AlertManager configured
- [ ] Log aggregation setup (ELK/EFK)
- [ ] Distributed tracing enabled
- [ ] Health checks implemented

### Application Monitoring
- [ ] HTTP request metrics
- [ ] Database query metrics
- [ ] Cache performance metrics
- [ ] Business logic metrics
- [ ] Error tracking and alerting
- [ ] Performance gates configured

### Infrastructure Monitoring
- [ ] System resource metrics
- [ ] Container metrics
- [ ] Network metrics
- [ ] Storage metrics
- [ ] Service discovery

### Security Monitoring
- [ ] Authentication events
- [ ] Authorization failures
- [ ] Rate limit violations
- [ ] Suspicious activity detection
- [ ] Security audit logs

### Operational Monitoring
- [ ] SLA/SLO tracking
- [ ] Error budget monitoring
- [ ] Capacity planning metrics
- [ ] Deployment tracking
- [ ] Incident response integration

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Site Reliability Engineering Book](https://sre.google/sre-book/table-of-contents/)
- [Monitoring and Observability Best Practices](https://docs.datadoghq.com/monitors/)

For monitoring support:
- Email: monitoring@yourdomain.com
- Slack: #monitoring-support
- Documentation: https://docs.yourdomain.com/monitoring