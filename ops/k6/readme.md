# k6 Load Tests

## Smoke + Ramp
```bash
docker run --rm -i -e API_URL=http://localhost:8080 grafana/k6 run - < query-smoke.js
```
Adjust `API_URL` for your environment. Thresholds are set for p95 < 800ms and <2% errors.