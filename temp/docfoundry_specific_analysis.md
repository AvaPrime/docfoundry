# DocFoundry Specific Analysis & Implementation Plan

## Project Overview
Based on the delta review, DocFoundry is a production-grade document search and retrieval system with:
- **Stack**: FastAPI + PostgreSQL/pgvector + Redis
- **Core Features**: Document ingestion, vector search, BM25 text search, hybrid retrieval
- **Architecture**: Modular services (API, worker, MCP, shared components)
- **Operations**: Prometheus monitoring, k6 testing, Alertmanager integration

## Current State Assessment

### ✅ Strengths Identified
1. **Solid Foundation**: FastAPI + Postgres/pgvector + Redis architecture
2. **Production-Minded**: Includes Makefile, k6 testing, CI touchpoints  
3. **Observability**: Metrics endpoints, Prometheus integration planned
4. **Modular Structure**: Services split appropriately for scaling

### 🔍 Critical Areas for Immediate Attention

## Phase A: Quality & Safety (Priority 1 - This Week)

### 1. Database & Search Quality Issues
**Current Problem**: Basic vector search without optimized retrieval
**Solution**: Implement production retrieval improvements

```sql
-- Missing: True BM25 implementation via ts_rank_cd
-- Missing: IVFFLAT index optimization
-- Missing: Maintained FTS columns
```

**Action Items**:
- [ ] Apply Alembic migration 001 (documents, chunks, FTS trigger/indexes, IVFFLAT)
- [ ] Implement `optimize_vector_index()` with `lists≈√rows` calculation  
- [ ] Add `ts_rank_cd` BM25 scoring with maintained FTS column
- [ ] Configure `ivfflat.probes` at connection time

**Files to Update**:
- `alembic/versions/001_*.py` - Database schema
- `models.py` - Add FTS columns
- `retrieval.py` - Replace with ProductionRetrievalService
- `database.py` - Add connection-time IVFFLAT tuning

### 2. Security Vulnerabilities
**Current Problems**: 
- Permissive CORS configuration
- No rate limiting
- SSRF risks in crawler

**Action Items**:
- [ ] Add SlowAPI + Redis rate limiting (`/query` 30/min, `/ingest` 5/min)
- [ ] Tighten CORS origins for production
- [ ] Add private IP/CIDR deny to crawler
- [ ] Implement scheme filtering for URLs

**Files to Update**:
- `main.py` - Add SlowAPI middleware and CORS restrictions
- `crawler.py` - Add SSRF protection and CIDR guards
- `requirements.txt` - Add slowapi dependency

```python
# Example rate limiting implementation
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, storage_uri="redis://redis:6379")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/query")
@limiter.limit("30/minute")
async def query_endpoint(request: Request, ...):
    pass
```

### 3. Missing Observability
**Current Problem**: Incomplete metrics and alerting setup

**Action Items**:
- [ ] Ensure API exposes `/metrics` endpoint
- [ ] Configure worker metrics server on port 9108  
- [ ] Load Prometheus rules from ops pack
- [ ] Configure Alertmanager routes
- [ ] Set up SLO-based alerting

**Files to Create/Update**:
- `prometheus/rules.yml` - Performance and error rate rules
- `alertmanager/alertmanager.yml` - Alert routing configuration
- `docker-compose.prod.yml` - Add Prometheus/Alertmanager services

## Phase B: Infrastructure Confidence (Priority 2 - This Week)

### 4. Production Docker Configuration
**Current Problem**: Development docker-compose not production-ready

**Action Items**:
- [ ] Merge `docker-compose.prod.yml` with resource limits
- [ ] Add Redis service if missing
- [ ] Configure proper healthchecks
- [ ] Set up environment variable wiring
- [ ] Add Prometheus and Alertmanager services

```yaml
# Example production docker-compose additions
services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./ops/prometheus:/etc/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
```

### 5. Performance Testing Gateway
**Current Problem**: No performance regression protection

**Action Items**:
- [ ] Integrate `ops/k6/query-smoke.js` into CI
- [ ] Set performance thresholds (p95 < 800ms, <2% errors)  
- [ ] Add k6 smoke test as merge gate
- [ ] Configure nightly performance runs

**Files to Update**:
- `.github/workflows/ci.yml` - Add k6 testing step
- `ops/k6/query-smoke.js` - Point to correct endpoints

```javascript
// k6 performance thresholds
export let options = {
  thresholds: {
    http_req_duration: ['p(95)<800'], // 95% of requests under 800ms
    http_req_failed: ['rate<0.02'],   // Error rate under 2%
  },
};
```

## Phase C: Measured Upgrades (Priority 3 - Next Sprint)

### 6. Data Lineage & Reproducibility
**Current Problem**: No embedding versioning or content lineage

**Action Items**:
- [ ] Add `embedding_model`, `embedding_version`, `chunker_version` to chunks table
- [ ] Implement content hash-based incremental refresh
- [ ] Add model version tracking for reproducibility

```python
# Database schema additions
class Chunk(Base):
    # ... existing fields
    embedding_model: str = Field(..., description="Model used for embeddings")
    embedding_version: str = Field(..., description="Model version")  
    chunker_version: str = Field(..., description="Chunking algorithm version")
    content_hash: str = Field(..., description="SHA256 of content")
```

### 7. Configuration Externalization
**Current Problem**: Hard-coded parameters limiting tuning ability

**Action Items**:
- [ ] Move `rrf_k`, `ivfflat.probes`, BM25 config to environment variables
- [ ] Create Settings class with validation
- [ ] Link parameters to evaluation harness for A/B testing

```python
# Settings configuration
class Settings(BaseSettings):
    rrf_k: float = Field(60.0, description="RRF constant")
    ivfflat_probes: int = Field(10, description="IVFFLAT probes")
    bm25_language: str = Field("english", description="BM25 language config")
    
    class Config:
        env_file = ".env"
```

## Critical Security Issues to Address Immediately

### 1. SSRF Prevention in Crawler
```python
# Add to crawler.py
import ipaddress
from urllib.parse import urlparse

BLOCKED_CIDRS = [
    ipaddress.IPv4Network('10.0.0.0/8'),
    ipaddress.IPv4Network('172.16.0.0/12'),
    ipaddress.IPv4Network('192.168.0.0/16'),
    ipaddress.IPv4Network('127.0.0.0/8'),
    ipaddress.IPv4Network('169.254.0.0/16'),
]

def is_safe_url(url: str) -> bool:
    """Check if URL is safe to crawl (no private IPs)"""
    parsed = urlparse(url)
    if parsed.scheme not in ['http', 'https']:
        return False
    
    try:
        ip = ipaddress.ip_address(parsed.hostname)
        return not any(ip in cidr for cidr in BLOCKED_CIDRS)
    except ValueError:
        return True  # Hostname, not IP
```

### 2. Input Validation & Sanitization
```python
# Add comprehensive input validation
from pydantic import BaseModel, validator
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = Field(default=10, le=100)  # Prevent large responses
    
    @validator('query')
    def validate_query(cls, v):
        if len(v) > 1000:  # Prevent extremely long queries
            raise ValueError('Query too long')
        return v.strip()
```

## Performance Optimization Priorities

### 1. Database Query Optimization
**Issues to Address**:
- N+1 queries in document retrieval
- Missing indexes on frequently queried columns
- Inefficient vector similarity calculations

```sql
-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_chunks_document_id ON chunks(document_id);
CREATE INDEX CONCURRENTLY idx_documents_created_at ON documents(created_at);
CREATE INDEX CONCURRENTLY idx_chunks_fts ON chunks USING gin(to_tsvector('english', content));
```

### 2. Memory Management
**Issues to Address**:
- Large document processing without streaming
- Vector index memory usage
- Redis cache sizing

```python
# Streaming document processing
async def process_large_document(file_path: str, chunk_size: int = 8192):
    """Process documents in chunks to manage memory"""
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            yield process_chunk(chunk)
```

## Monitoring & Alerting Setup

### Key Metrics to Track
1. **Performance Metrics**:
   - Query latency (p95, p99)
   - Ingestion throughput
   - Vector search accuracy

2. **Error Metrics**:
   - HTTP error rates by endpoint
   - Document processing failures
   - Database connection errors

3. **Business Metrics**:
   - Documents indexed per hour
   - Query volume and patterns
   - User engagement metrics

### Alerting Rules Priority
```yaml
# High-priority alerts
- alert: HighErrorRate
  expr: http_requests_total{status=~"5.."} / http_requests_total > 0.05
  
- alert: SlowQueries  
  expr: histogram_quantile(0.95, http_request_duration_seconds) > 2.0
  
- alert: DocumentProcessingBacklog
  expr: document_processing_queue_length > 1000
```

## Implementation Timeline

### Week 1 (Critical)
- [ ] Database migrations and schema updates
- [ ] Production retrieval service implementation  
- [ ] Rate limiting and CORS hardening
- [ ] Basic monitoring setup

### Week 2 (High Priority)
- [ ] Production docker-compose configuration
- [ ] Performance testing integration
- [ ] SSRF protection in crawler
- [ ] Alerting configuration

### Week 3 (Medium Priority)  
- [ ] Data lineage implementation
- [ ] Configuration externalization
- [ ] Advanced monitoring dashboards
- [ ] Documentation updates

## Risk Assessment

### High Risk
- **Database migration**: Could cause downtime if not handled properly
- **Rate limiting**: May impact existing users if limits too aggressive
- **CORS changes**: Could break existing integrations

### Medium Risk
- **Docker configuration changes**: May require deployment process updates
- **Performance testing**: Could reveal performance issues requiring immediate fixes

### Low Risk
- **Monitoring additions**: Additive changes with minimal impact
- **Documentation updates**: No functional impact

This implementation plan addresses the specific issues identified in your delta review while providing a structured approach to improving DocFoundry's production readiness.