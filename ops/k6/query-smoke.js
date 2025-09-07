// k6 smoke & ramp test for DocFoundry /query
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  scenarios: {
    smoke: {
      executor: 'shared-iterations',
      vus: 2,
      iterations: 10,
      maxDuration: '1m',
    },
    ramp: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '1m', target: 10 },
        { duration: '2m', target: 25 },
        { duration: '2m', target: 0 },
      ],
      gracefulRampDown: '30s',
      startTime: '1m',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.02'],
    http_req_duration: ['p(95)<800'],
  },
};

const API = __ENV.API_URL || 'http://localhost:8080';

export default function () {
  const payload = JSON.stringify({
    q: 'Explain saga choreography vs orchestration',
    domain: 'microservices.io',
    k: 8,
    enable_rerank: true,
  });

  const res = http.post(`${API}/query`, payload, {
    headers: { 'Content-Type': 'application/json' },
    timeout: '60s',
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'has results': (r) => {
      try {
        const data = JSON.parse(r.body);
        return (data.total || 0) > 0;
      } catch (e) { return false; }
    },
  });

  sleep(0.5);
}