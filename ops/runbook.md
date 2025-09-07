# DocFoundry Ops Runbook

## Quick Facts
- API: FastAPI on :8080 (`/healthz`, `/metrics`)
- Worker metrics: :9108
- DB: Postgres + pgvector, FTS via tsvector
- Rate limiting: Redis + slowapi

## Common Incidents

### A) API latency p95 > 750ms
1. Check Prometheus graph for `docfoundry_request_duration_seconds_bucket`.
2. Verify DB CPU/IO; run `VACUUM ANALYZE` if bloat suspected.
3. Reduce `ivfflat.probes` temporarily (connection option) to lower ANN cost.
4. Disable reranker via `ENABLE_RERANKING=false` and redeploy to isolate cause.
5. Inspect hot queries in Postgres: `pg_stat_statements` (enable if not already).

### B) Crawler stalled
1. Alert will fire (`WorkerNoCrawlProgress`). Check worker logs.
2. Inspect target robots and `Retry-After`. Increase crawl delay or whitelist.
3. Confirm sitemap is reachable and non-empty.
4. If 429s, raise backoff caps; reduce concurrency per domain.

### C) High 5xx or query failures
1. Tail API logs; note exceptions.
2. Validate Redis availability (rate limiter); fall back to in-memory if needed.
3. Check DB connections — pool exhausted? Raise `DB_POOL_SIZE` or reduce VUs.

## Emergency Rollback
- Revert to previous container tag.
- Run `alembic downgrade -1` only if the last migration is known-safe to reverse.
- Restore DB from latest dump (see backup procedure).

## Routine Maintenance
- `make backup.db` nightly via cron.
- Weekly: `make optimize.db` (VACUUM ANALYZE + rebuild ivfflat lists via helper).
- Rotate logs and prune Docker images.

## Tuning Cheatsheet
- ANN recall: `ivfflat.probes` ↑ → recall ↑ / latency ↑
- Lists parameter: ≈ sqrt(N rows). Rebuild after big ingests.
- Reranker cost: O(k) cross-encoder; keep `k<=10` for production traffic.