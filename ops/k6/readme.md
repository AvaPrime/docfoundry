# k6 Performance Testing for DocFoundry

This directory contains comprehensive k6 performance tests for the DocFoundry API, designed to validate performance characteristics and identify bottlenecks under various load conditions.

## Test Types

### Smoke Test (`query-smoke.js`)
- **Purpose**: Light load testing to catch basic performance issues
- **Load**: 5 concurrent users
- **Duration**: 2 minutes
- **Thresholds**: Strict SLA requirements (95% < 800ms, <2% errors)
- **Use Case**: CI/CD pipeline validation

### Stress Test (`stress-test.js`)
- **Purpose**: High load testing to find breaking points
- **Load**: Up to 300 concurrent users at peak
- **Duration**: 17 minutes total
- **Thresholds**: Relaxed for stress conditions (95% < 1.5s, <5% errors)
- **Use Case**: Capacity planning and bottleneck identification

## Running Tests Locally

### Prerequisites
```bash
# Install k6
brew install k6  # macOS
# or
sudo apt install k6  # Ubuntu
# or
choco install k6  # Windows
```

### Quick Start
```bash
# Start DocFoundry services
docker-compose up -d

# Wait for services to be ready
curl -f http://localhost:8080/healthz

# Run smoke test
k6 run ops/k6/query-smoke.js

# Run stress test (optional)
k6 run ops/k6/stress-test.js
```

### With Test Data
```bash
# Use test overlay with seeded data
docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d

# Seed performance data
docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm seeder

# Run tests with k6 service
docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm k6
```

## CI/CD Integration

### GitHub Actions
Performance tests are integrated into the CI/CD pipeline:

- **Pull Requests**: Smoke tests run on every PR
- **Nightly**: Comprehensive stress tests run daily at 2 AM UTC
- **Manual**: Tests can be triggered manually via workflow_dispatch

### Workflow Jobs
1. **test**: Runs smoke tests after unit tests pass
2. **performance-nightly**: Runs comprehensive performance suite

## Environment Variables

| Variable | Description | Default |
|----------|-------------|----------|
| `API_URL` | Base URL for the DocFoundry API | `http://localhost:8080` |
| `API_KEY` | API key for authentication | `""` (empty) |

## Key Metrics

### Built-in k6 Metrics
- `http_req_duration`: HTTP request duration percentiles
- `http_req_failed`: HTTP request failure rate
- `http_reqs`: Total HTTP requests per second
- `vus`: Number of active virtual users

### Custom Metrics
- `search_latency`: End-to-end search operation latency
- `errors`: Custom error rate tracking
- `concurrent_users`: Active user count (stress test only)

## Performance Thresholds

### Smoke Test (Strict SLA)
- 95% of requests < 800ms
- Error rate < 2%
- Search latency 95% < 1s

### Stress Test (Capacity Limits)
- 95% of requests < 1.5s
- Error rate < 5%
- Search latency 95% < 2s

## Test Data

### Minimal Dataset (`seed_test_data.py`)
- 5 documents, 50 chunks
- Used for smoke tests
- Fast seeding (~30 seconds)

### Performance Dataset (`seed_performance_data.py`)
- 100 documents, 1000+ chunks
- Used for stress tests
- Comprehensive seeding (~5 minutes)
- Realistic content diversity

## User Behavior Patterns

The stress test simulates different user types:

1. **Simple Users**: Basic queries, moderate limits, relaxed timing
2. **Complex Users**: Advanced queries, higher limits, longer think time
3. **Power Users**: Complex queries, high limits, fast interaction
4. **Rapid Users**: Mixed queries, variable limits, minimal delays

## Troubleshooting

### Common Issues

**API Not Ready**
```bash
# Check API health
curl -f http://localhost:8080/healthz

# Check logs
docker-compose logs api
```

**High Error Rates**
- Check database connectivity
- Verify Redis is running
- Review API logs for specific errors
- Ensure sufficient resources (CPU/Memory)

**Slow Performance**
- Check database query performance
- Monitor system resources
- Review embedding model performance
- Verify network latency

### Performance Debugging

```bash
# Run with detailed output
k6 run --verbose ops/k6/query-smoke.js

# Generate HTML report
k6 run --out json=results.json ops/k6/stress-test.js
# Convert to HTML (requires k6-reporter)
k6-reporter results.json
```

### Resource Monitoring

```bash
# Monitor container resources
docker stats

# Check database performance
docker-compose exec postgres psql -U postgres -d docfoundry_test -c "SELECT * FROM pg_stat_activity;"

# Monitor Redis
docker-compose exec redis redis-cli info stats
```

## Performance Targets

### Production SLA Goals
- **Availability**: 99.9% uptime
- **Response Time**: 95% of requests < 500ms
- **Throughput**: 1000+ requests/minute
- **Error Rate**: < 0.1%

### Load Capacity
- **Concurrent Users**: 200+ sustained
- **Peak Load**: 500+ users for short bursts
- **Data Scale**: 10,000+ documents, 100,000+ chunks

## Results Analysis

After running tests, analyze results for:

1. **Response Time Distribution**: Check p95, p99 percentiles
2. **Error Patterns**: Identify error types and frequency
3. **Throughput Trends**: Monitor requests/second over time
4. **Resource Utilization**: CPU, memory, database performance
5. **Bottleneck Identification**: Find limiting factors

Use results to guide optimization efforts and capacity planning decisions.