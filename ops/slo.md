# DocFoundry SLOs

## Availability
- API availability: 99.9% monthly (Prom `up` + synthetic checks)

## Performance
- Search p95 latency < 800ms over 7d
- Error rate < 2% over 7d

## Freshness
- New/updated pages ingested within 6 hours of sitemap update (for tracked domains)

## Observability
- 100% of services scraped by Prometheus
- Alert acknowledgment within 15 minutes

## Compliance
- Robots.txt respected 100% of the time
- Configurable rate limits per endpoint deployed in prod

### SLIs
- `histogram_quantile(.95, sum(rate(docfoundry_request_duration_seconds_bucket[5m])) by (le))`
- `sum(rate(docfoundry_requests_total{endpoint="/query"}[5m]))`
- `increase(docfoundry_crawled_pages_total[6h])`