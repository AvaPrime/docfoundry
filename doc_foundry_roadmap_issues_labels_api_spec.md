# 📋 DocFoundry – Roadmap, Issues, Labels & API Specification

> **Last Updated:** January 2025  
> **Status:** Production Ready with Active Development  
> **Version:** v1.2.0  

---

## 🎯 PRIORITY NEXT STEPS

### **✅ COMPLETED FEATURES**

#### **High Priority Infrastructure (Production Ready)**
1. **✅ [Hybrid Search]** Vector + keyword search with score fusion (FTS5 + embeddings with RRF)
2. **✅ [Jobs & Scheduling]** Background task processing for periodic crawls with Redis/Celery
3. **✅ [Policy Guardrails]** License/robots/noai compliance and content filtering with SPDX matching
4. **✅ [MCP Server]** Model Context Protocol implementation for AI assistant integration
5. **✅ [Learning-to-Rank]** Click feedback system with offline training and online scoring
6. **✅ [Crawler Enhancement]** Robust HTML fetcher with robots.txt support and error handling
7. **✅ [PostgreSQL Migration]** Database migration from SQLite to PostgreSQL with pgvector support
8. **✅ [OpenTelemetry Integration]** Complete observability stack with distributed tracing and metrics
9. **✅ [API Streaming]** Server-Sent Events for real-time search results and enhanced UX
10. **✅ [Heading-aware Chunker]** Enhanced document chunker with stable IDs and metadata extraction

#### **Medium Priority Quality & Reliability Enhancements (Production Ready)**
11. **✅ [Gold Set + Metrics Runner Enhancement]** Enhanced evaluation framework with comprehensive metrics, automated reporting, and quality tracking
12. **✅ [Source Schema Validation Enhancement]** Enhanced source validation system with comprehensive error reporting, schema evolution support, and performance validation
13. **✅ [Enhanced Web UI & Search Experience]** Modern React-based search interface with advanced filtering, responsive design, dark mode, and comprehensive UX features
14. **✅ [Caching & Performance Optimization]** Comprehensive caching system with Redis integration, embedding cache, performance monitoring, and optimization strategies

---

### **⏳ REMAINING PRIORITIES** - Future Enhancements

#### **HIGH PRIORITY** - Critical for Production Readiness

*All high-priority infrastructure items have been completed and moved to the COMPLETED FEATURES section above.*

#### **MEDIUM PRIORITY** - Quality & Reliability Enhancements

*All medium-priority quality and reliability enhancements have been completed and moved to the COMPLETED FEATURES section above.*

---

#### **LOW PRIORITY** - Ecosystem Expansion

##### 1. Third-party Integrations & Connectors
**Status**: PENDING ⏳  
**Labels**: `enhancement`, `integrations`, `connectors`  
**Priority**: LOW - Ecosystem expansion  
**Impact**: High - Expands market reach and use cases  
**Effort**: High - Requires multiple integration implementations  
**Assignee**: TBD

**Description:**
Expand ecosystem integration with popular platforms and services.

**Acceptance Criteria:**
- [ ] Slack and Discord bot integration with search commands
- [ ] Notion and Confluence connector implementations
- [ ] Google Drive and SharePoint document sync
- [ ] Zapier and Make.com workflow integrations
- [ ] Webhook support for real-time updates and notifications
- [ ] SSO and enterprise authentication (SAML, OIDC)
- [ ] API rate limiting and usage analytics
- [ ] Plugin architecture for custom integrations

**Business Value:** Enables enterprise adoption and workflow integration

---

## 📋 COMPLETED COMPONENTS

This package contains:

1) **Issue labels** (names, descriptions, colors) ✅ **COMPLETED**
2) **GitHub issues** ready to paste (grouped by milestone/area) ✅ **COMPLETED**
3) **OpenAPI spec (`openapi.yaml`)** for new/updated endpoints ✅ **COMPLETED**
4) **Source schema v1** (YAML + JSON Schema) ✅ **COMPLETED**
5) **Chunker module stub** (Python) ✅ **COMPLETED**
6) **pgvector migration** (SQL) + minimal ORM model ✅ **COMPLETED**
7) **Makefile targets** ✅ **COMPLETED**
8) **Observability hooks** (OpenTelemetry + Prometheus) ✅ **COMPLETED**
9) **Evaluation harness skeleton** ✅ **COMPLETED**
10) **`gh` CLI script** to create labels & issues programmatically ✅ **COMPLETED**

---

## 1) Issue Labels (recommended)

Create these labels first; colors use hex without `#`.

```text
core                 | #0EA5E9 | Core plumbing & correctness
pipelines            | #22C55E | Crawling, parsing, normalization
indexer              | #8B5CF6 | Index & ranking
server               | #F59E0B | API & auth
ux                   | #EC4899 | VS Code & Chrome extensions, docs UX
ops                  | #64748B | Deploy, CI/CD, containers
observability        | #84CC16 | Logs, traces, metrics
policy               | #EF4444 | Licenses, robots, OPA
eval                 | #14B8A6 | Test sets, metrics, reports
performance          | #A855F7 | Latency, throughput, footprints
security             | #EF4444 | Secrets, scanning, authz
mcp                  | #06B6D4 | Model Context Protocol server/client
priority:now         | #D946EF | Must ship in current sprint
kind:enhancement     | #3B82F6 | Feature work
kind:bug             | #DC2626 | Bugfix
kind:refactor        | #0EA5E9 | Internal improvement
"good first issue"  | #10B981 | Starter scoped tasks
```

---

## 2) Issues (copy/paste to GitHub)

### Milestone: **MVP Hardening** (Tier 1)

#### [Pipelines] Robust crawler with caching & robots support ✅ **COMPLETED**
**Labels:** `pipelines`, `core`, `kind:enhancement`  
**Priority:** High  
**Status:** Production Ready  

**Goal**
Build a resilient HTML fetcher with polite crawling and incremental updates.

**Completed Features:**
- ✅ Respect `robots.txt`; support per-host rate limiting/backoff
- ✅ Sitemap discovery; `lastmod` hints
- ✅ ETag/`Last-Modified` conditional requests; local content cache
- ✅ Canonical URL de-duplication
- ✅ Boilerplate removal (readability rules); selector allow/deny lists
- ✅ PDF -> Markdown fallback via `pdftotext`/`pdfminer`
- ✅ `python pipelines/html_ingest.py sources/example.yaml` crawls target site end-to-end
- ✅ 304s skip writes; 200s update cache and emit new version hash
- ✅ Robots disallow paths are skipped and logged with reason
- ✅ Crawl throughput and error metrics exported (see observability)
- ✅ Unit tests for cache/update logic
- ✅ Docs added under `docs/pipelines/crawler.md`

**Implementation Status:** Complete - Enhanced crawler implemented in `pipelines/crawler.py` with robots.txt support, error handling, retry mechanisms, and policy compliance

---

#### [Pipelines] Source file schema v1 & validator ✅ **COMPLETED**
**Labels:** `pipelines`, `core`, `kind:enhancement`  
**Priority:** High  
**Status:** Production Ready  

**Goal**
Define a stable YAML schema for sources and provide a validator.

**Completed Features:**
- ✅ Schema covers: `name`, `base_urls[]`, `sitemaps[]`, `include[]`, `exclude[]`, `rate_limit`, `depth`, `priority`, `auth`, `license_hint`
- ✅ CLI: `python pipelines/validate_source.py sources/*.yaml`
- ✅ Fails with line/column and helpful message on invalid fields

**Implementation Status:** Complete - Source schema implemented in `indexer/source_schema.py` with comprehensive validation

---

#### [Indexer] Heading-aware chunker + metadata ✅ **COMPLETED**
**Labels:** `indexer`, `core`, `priority:now`, `kind:enhancement`  
**Priority:** High  
**Status:** Production Ready  

**Goal**
Split Markdown into retrieval-friendly chunks while preserving section context.

**Completed Features:**
- ✅ Configurable token budget & overlap; respects headings and code fences
- ✅ Emits: `chunk_id`, `doc_id`, `h_path` (H1→Hn), `url`, `retrieved_at`, `hash`, `token_len`, `lang`
- ✅ Deterministic hashing; idempotent re-chunking

**Implementation Status:** Complete - Full implementation in `indexer/chunker.py` with heading-aware splitting and metadata

---

#### [Indexer] Hybrid search (FTS5 + embeddings) with RRF ✅ **COMPLETED**
**Labels:** `indexer`, `performance`, `kind:enhancement`  
**Priority:** High  
**Status:** Production Ready  

**Goal**
Blend lexical (BM25) and vector search; optional cross-encoder rerank.

**Completed Features:**
- ✅ pgvector column + index; cosine distance
- ✅ Reciprocal Rank Fusion to combine FTS and vector top-k
- ✅ Feature flag for reranker; configurable model name

**Implementation Status:** Complete - Full hybrid search implementation with RRF scoring in `indexer/embeddings.py`

---

#### [Server] API v1 polish + streaming responses ✅ **COMPLETED**
**Labels:** `server`, `core`, `kind:enhancement`  
**Priority:** High  
**Status:** Production Ready  

**Goal**
Harden the FastAPI surface and add endpoints for ingestion & jobs.

**Completed Features:**
- ✅ Endpoints implemented per `openapi.yaml`
- ✅ API keys + CORS + structured error model
- ✅ Server-Sent Events (SSE) for streaming `/search`
- ✅ OpenAPI served at `/openapi.json` and docs at `/docs`

**Implementation Status:** Complete - Full API implementation with authentication, CORS, and streaming responses

---

#### [Observability] OpenTelemetry + Prometheus ✅ **COMPLETED**
**Labels:** `observability`, `ops`  
**Priority:** High  
**Status:** Production Ready  

**Goal**
Trace ingest→index→query; export standard metrics.

**Completed Features:**
- ✅ Traces for each request; span attributes for source, bytes, durations
- ✅ `/metrics` exposes counters/histograms; Kubernetes ready
- ✅ Dashboards stub in `hosting/grafana/`

**Implementation Status:** Complete - Full observability stack implemented with OpenTelemetry tracing and Prometheus metrics

---

#### [Eval] Gold set + metrics runner ✅ **COMPLETED**
**Labels:** `eval`, `indexer`, `server`  
**Priority:** Medium  
**Status:** Production Ready  

**Goal**
Create a minimal evaluation harness to track search quality.

**Completed Features:**
- ✅ `df eval run` computes `nDCG@k`, `MRR` on a small gold set per source
- ✅ Markdown report in `docs/reports/` with trend chart

**Implementation Status:** Complete - Evaluation harness implemented with quality metrics and reporting

---

### Milestone: **Team Grade** (Tier 2)

#### [Server] Jobs & scheduling (periodic crawls) ✅ **COMPLETED**
**Labels:** `server`, `ops`, `kind:enhancement`  
**Priority:** High  
**Status:** Production Ready  

**Goal**
Introduce job records and periodic crawl triggers; webhooks on change.

**Completed Features:**
- ✅ `POST /ingest` enqueues; `GET /jobs/{id}` returns status/logs
- ✅ Simple APScheduler; emit webhook on new/changed docs

**Implementation Status:** Complete - Full job queue implementation with Redis and Celery in `server/jobs.py` and `server/job_handlers.py`

---

#### [Policy] License/robots/noai guardrails ✅ **COMPLETED**
**Labels:** `policy`, `pipelines`, `security`  
**Priority:** High  
**Status:** Production Ready  

**Goal**
Block non-compliant sources; annotate chunks with policy metadata.

**Completed Features:**
- ✅ SPDX text matching; robots/noai respected; per-source allowlist
- ✅ Policy violations surfaced in `/ingest` logs and `/doc` metadata

**Implementation Status:** Complete - Full policy implementation with robots.txt parsing and content filtering in `pipelines/policy.py` and `config/policy_loader.py`

---

#### [MCP] Minimal Model Context Protocol server ✅ **COMPLETED**
**Labels:** `mcp`, `server`  
**Priority:** Medium  
**Status:** Production Ready  

**Goal**
Expose `search`, `fetch_doc`, `list_sources`, `capture_url` over MCP.

**Completed Features:**
- ✅ Stdio/socket server; JSON-RPC; client example included

**Implementation Status:** Complete - Full MCP server implementation in `server/mcp_server.py` with all required endpoints

---

### Milestone: **Platform Grade** (Tier 3)

#### [Indexer] Learning-to-Rank (click feedback) ✅ **COMPLETED**
**Labels:** `indexer`, `performance`, `kind:enhancement`  
**Priority:** Medium  
**Status:** Production Ready  

**Goal**
Optional LTR stage trained on clicks; export/fit pipeline.

**Completed Features:**
- ✅ Click log schema; offline trainer; online scorer behind flag

**Implementation Status:** Complete - Full LTR implementation in `indexer/learning_to_rank.py` with click logging, feature extraction, and model reranking. API endpoints added to `server/rag_api.py` for feedback collection.

---

## 3) `openapi.yaml`

```yaml
openapi: 3.0.3
info:
  title: DocFoundry API
  version: 1.0.0
servers:
  - url: http://localhost:8001
paths:
  /search:
    post:
      summary: Hybrid search over indexed content
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                q: { type: string }
                top_k: { type: integer, default: 10 }
                hybrid: { type: boolean, default: true }
                use_reranker: { type: boolean, default: false }
                filters:
                  type: object
                  properties:
                    source: { type: array, items: { type: string } }
                    lang: { type: string }
      responses:
        '200':
          description: Search results
          content:
            application/json:
              schema:
                type: object
                properties:
                  query: { type: string }
                  results:
                    type: array
                    items:
                      type: object
                      properties:
                        doc_id: { type: string }
                        chunk_id: { type: string }
                        score: { type: number }
                        title: { type: string }
                        url: { type: string }
                        h_path: { type: array, items: { type: string } }
                        snippet: { type: string }
                        retrieved_at: { type: string, format: date-time }
  /doc/{doc_id}:
    get:
      summary: Fetch a normalized document with metadata
      parameters:
        - in: path
          name: doc_id
          required: true
          schema: { type: string }
      responses:
        '200':
          description: Document
          content:
            application/json:
              schema:
                type: object
                properties:
                  doc_id: { type: string }
                  url: { type: string }
                  frontmatter: { type: object }
                  markdown: { type: string }
  /ingest:
    post:
      summary: Enqueue ingestion for a source or URLs
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                source: { type: string, description: Source name from YAML }
                urls: { type: array, items: { type: string } }
      responses:
        '202':
          description: Job accepted
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id: { type: string }
  /jobs/{id}:
    get:
      summary: Get job status/logs
      parameters:
        - in: path
          name: id
          required: true
          schema: { type: string }
      responses:
        '200':
          description: Job details
          content:
            application/json:
              schema:
                type: object
                properties:
                  id: { type: string }
                  status: { type: string, enum: [queued, running, done, failed] }
                  logs: { type: array, items: { type: string } }
  /sources:
    get:
      summary: List available sources
      responses:
        '200':
          description: Sources
          content:
            application/json:
              schema:
                type: object
                properties:
                  items:
                    type: array
                    items:
                      type: object
                      properties:
                        name: { type: string }
                        base_urls: { type: array, items: { type: string } }
                        enabled: { type: boolean }
  /capture:
    post:
      summary: Capture an ad-hoc page from the browser extension
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                url: { type: string }
                html: { type: string }
                screenshot_path: { type: string }
      responses:
        '201':
          description: Created research note
          content:
            application/json:
              schema:
                type: object
                properties:
                  doc_id: { type: string }
                  path: { type: string }
  /healthz:
    get:
      summary: Health check
      responses:
        '200':
          description: OK
```

---

## 4) Source Schema v1

### YAML (example)

```yaml
name: openrouter
base_urls:
  - https://openrouter.ai/docs/
sitemaps:
  - https://openrouter.ai/sitemap.xml
include:
  - /docs/**
exclude:
  - /blog/**
rate_limit: 0.5        # requests per second
backoff: { base: 0.25, max: 8 }
depth: 4
priority: normal       # low|normal|high
auth: null             # or { type: header, name: "X-Token", value: "..." }
license_hint: permissive
```

### JSON Schema (for validator)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["name", "base_urls"],
  "properties": {
    "name": {"type": "string", "pattern": "^[a-z0-9_-]+$"},
    "base_urls": {"type": "array", "items": {"type": "string", "format": "uri"}, "minItems": 1},
    "sitemaps": {"type": "array", "items": {"type": "string", "format": "uri"}},
    "include": {"type": "array", "items": {"type": "string"}},
    "exclude": {"type": "array", "items": {"type": "string"}},
    "rate_limit": {"type": "number", "minimum": 0.01},
    "backoff": {"type": "object", "properties": {"base": {"type": "number"}, "max": {"type": "number"}}, "required": ["base", "max"]},
    "depth": {"type": "integer", "minimum": 0, "maximum": 10},
    "priority": {"type": "string", "enum": ["low", "normal", "high"]},
    "auth": {"type": ["null", "object"]},
    "license_hint": {"type": "string"}
  },
  "additionalProperties": false
}
```

---

## 5) Chunker Module Stub (`indexer/chunker.py`)

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    h_path: List[str]
    url: str | None
    retrieved_at: str | None
    token_len: int
    lang: str | None
    hash: str

class Chunker:
    def __init__(self, max_tokens: int = 512, overlap: int = 64):
        assert overlap < max_tokens
        self.max_tokens = max_tokens
        self.overlap = overlap

    def split_markdown(self, md: str) -> Iterable[Chunk]:
        """Heading-aware split that respects code fences and lists.
        1) Parse headings to build h_path (H1→Hn)
        2) Fuse small sections; split long sections by sentences with overlap
        3) Yield stable chunk_ids (e.g., short hash of h_path + offset)
        """
        # TODO: implement using a lightweight tokenizer (eg. tiktoken/regex)
        raise NotImplementedError
```

---

## 6) Postgres + pgvector Migration

### SQL (Alembic/SQL file)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  doc_id      TEXT,
  chunk_id    TEXT PRIMARY KEY,
  text        TEXT,
  h_path      TEXT[],
  url         TEXT,
  retrieved_at TIMESTAMPTZ,
  token_len   INT,
  lang        TEXT,
  hash        TEXT,
  embedding   VECTOR(1536)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_vec ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### Minimal SQLAlchemy model

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Chunk(Base):
    __tablename__ = "chunks"
    doc_id: Mapped[str] = mapped_column(primary_key=False)
    chunk_id: Mapped[str] = mapped_column(primary_key=True)
    text: Mapped[str]
    h_path: Mapped[list[str]] = mapped_column(ARRAY(str))
    url: Mapped[str | None]
    retrieved_at: Mapped[str | None]
    token_len: Mapped[int]
    lang: Mapped[str | None]
    hash: Mapped[str]
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
```

---

## 7) Makefile Targets (snippet)

```makefile
.PHONY: up seed index api docs eval fmt lint

up:
	docker compose up -d postgres grafana prometheus

seed:
	python pipelines/html_ingest.py sources/openrouter.yaml
	python indexer/build_index.py

index:
	python indexer/build_index.py

api:
	uvicorn server.rag_api:app --reload --port 8001

docs:
	mkdocs serve -a 127.0.0.1:8000

eval:
	python tools/eval_runner.py --gold data/goldsets/default.jsonl

fmt:
	ruff check --fix . && ruff format .

lint:
	ruff check .
```

---

## 8) Observability Hooks (FastAPI)

```python
# server/telemetry.py
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from prometheus_fastapi_instrumentator import Instrumentator

def setup_telemetry(app):
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

Hook in `rag_api.py`:

```python
from .telemetry import setup_telemetry
app = FastAPI()
setup_telemetry(app)
```

---

## 9) Evaluation Harness Skeleton

```bash
# tools/eval_runner.py (outline)
# - Reads gold Q/A or Q->doc references
# - Executes /search, computes nDCG@k and MRR
# - Writes docs/reports/<timestamp>.md with summary and trend
```

```python
# tools/metrics.py
from typing import List

def mrr(ranks: List[int]) -> float:
    return sum(1/r for r in ranks) / max(1, len(ranks))
```

---

## 10) `gh` CLI: create labels & issues

> Requires: `gh auth login`

```bash
# labels.sh
while read -r name color desc; do
  gh label create "$name" --color "${color#\#}" --description "$desc" || \
  gh label edit "$name" --color "${color#\#}" --description "$desc"
done <<'EOF'
core #0EA5E9 Core plumbing & correctness
pipelines #22C55E Crawling, parsing, normalization
indexer #8B5CF6 Index & ranking
server #F59E0B API & auth
ux #EC4899 VS Code & Chrome extensions, docs UX
ops #64748B Deploy, CI/CD, containers
observability #84CC16 Logs, traces, metrics
policy #EF4444 Licenses, robots, OPA
eval #14B8A6 Test sets, metrics, reports
performance #A855F7 Latency, throughput, footprints
security #EF4444 Secrets, scanning, authz
mcp #06B6D4 Model Context Protocol server/client
priority:now #D946EF Must ship in current sprint
kind:enhancement #3B82F6 Feature work
kind:bug #DC2626 Bugfix
kind:refactor #0EA5E9 Internal improvement
"good first issue" #10B981 Starter scoped tasks
EOF
```

```bash
# issues.sh (excerpt for top 6 issues)
new_issue() {
  local title="$1"; shift
  local labels="$1"; shift
  local body="$1"; shift
  gh issue create --title "$title" --label "$labels" --body "$body"
}

new_issue "[Pipelines] Robust crawler with caching & robots" \
  "pipelines,core,priority:now,kind:enhancement" \
  $'**Goal**\nResilient HTML fetcher with polite crawling and incremental updates.\n\n**Acceptance Criteria**\n- [ ] robots/sitemap\n- [ ] ETag/Last-Modified cache\n- [ ] canonical de-dup\n- [ ] PDF fallback\n- [ ] metrics exported\n'

new_issue "[Indexer] Heading-aware chunker + metadata" \
  "indexer,core,priority:now,kind:enhancement" \
  $'**Goal**\nChunk Markdown with heading paths and stable IDs.\n\n**Acceptance Criteria**\n- [ ] token budget & overlap\n- [ ] h_path + url + hash\n- [ ] deterministic ids\n'

# ...repeat for remaining issues...
```

---

## 📊 **PROJECT SUMMARY & NEXT STEPS**

### **Current Status**
- **✅ Core Infrastructure:** 100% Complete - All high-priority items delivered
- **⏳ Quality Enhancements:** 0% Complete - 2 medium-priority items pending
- **⏳ User Experience:** 0% Complete - 3 low-priority items pending
- **📈 Overall Progress:** ~70% of planned features completed

### **Immediate Priorities (Next 30 Days)**
1. **Gold Set + Metrics Runner Enhancement** - Critical for quality measurement
2. **Source Schema Validation Enhancement** - Improves data reliability

### **Medium-term Goals (Next 90 Days)**
3. **Enhanced Web UI & Search Experience** - High user impact
4. **Caching & Performance Optimization** - Operational efficiency

### **Long-term Vision (Next 180 Days)**
5. **Third-party Integrations & Connectors** - Market expansion

### **Key Success Metrics**
- **Search Quality:** nDCG@10 > 0.8, MRR > 0.7
- **Performance:** <200ms average response time, 99.9% uptime
- **User Adoption:** 50+ active users, 1000+ daily searches
- **Data Quality:** <1% validation errors, 100% policy compliance

---

*This document serves as the single source of truth for DocFoundry's development roadmap and should be updated regularly to reflect current progress and priorities.*

### Notes
- Paste issues directly into GitHub, or use `gh` scripts above.
- Adjust `VECTOR(1536)` to match your embedding dimensionality.
- Keep OpenAPI in `server/openapi.yaml`; CI can validate on PR.

