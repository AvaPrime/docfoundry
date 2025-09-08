import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
export const errorRate = new Rate('errors');
export const searchLatency = new Trend('search_latency');
export const concurrentUsers = new Trend('concurrent_users');

// Stress test configuration
export const options = {
  // Stress test: high load to find breaking points
  stages: [
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 100 },  // Scale to 100 users
    { duration: '5m', target: 200 },  // Stress test at 200 users
    { duration: '3m', target: 300 },  // Peak load
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    // More lenient thresholds for stress testing
    http_req_duration: ['p(95)<1500'],    // 95% under 1.5s
    http_req_failed: ['rate<0.05'],       // Error rate under 5%
    errors: ['rate<0.05'],                // Custom error rate under 5%
    search_latency: ['p(95)<2000'],       // Search latency under 2s
  },
};

// Configuration
const API_URL = __ENV.API_URL || 'http://localhost:8080';
const API_KEY = __ENV.API_KEY || '';

// Comprehensive test queries for stress testing
const testQueries = [
  // Simple queries
  'API',
  'database',
  'security',
  'performance',
  'monitoring',
  
  // Medium complexity
  'machine learning algorithms',
  'database optimization techniques',
  'API security best practices',
  'microservices architecture patterns',
  'document processing workflows',
  
  // Complex queries
  'advanced machine learning model deployment strategies',
  'distributed database optimization and performance tuning',
  'comprehensive API security framework implementation',
  'scalable microservices architecture with event sourcing',
  'real-time document processing and natural language understanding'
];

const testSites = [
  null,
  'example.com',
  'docs.example.com',
  'blog.example.com',
  'api.example.com',
  'support.example.com'
];

// Different user behavior patterns
const userBehaviors = {
  simple: {
    queryTypes: testQueries.slice(0, 5),
    limits: [5, 10],
    sleepRange: [1, 3]
  },
  complex: {
    queryTypes: testQueries.slice(5, 10),
    limits: [10, 20],
    sleepRange: [2, 5]
  },
  power: {
    queryTypes: testQueries.slice(10),
    limits: [20, 50],
    sleepRange: [0.5, 2]
  },
  rapid: {
    queryTypes: testQueries,
    limits: [5, 15],
    sleepRange: [0.1, 1]
  }
};

export function setup() {
  console.log('🔥 Starting DocFoundry stress test');
  console.log(`API URL: ${API_URL}`);
  console.log('Target: 300 concurrent users at peak');
  
  // Verify API health before stress testing
  const healthCheck = http.get(`${API_URL}/healthz`);
  check(healthCheck, {
    'API is healthy': (r) => r.status === 200,
  });
  
  if (healthCheck.status !== 200) {
    console.error('❌ API health check failed, aborting stress test');
    throw new Error('API not healthy');
  }
  
  console.log('✅ API health check passed - beginning stress test');
  return { apiUrl: API_URL };
}

export default function (data) {
  // Select user behavior pattern based on VU ID
  const behaviorKeys = Object.keys(userBehaviors);
  const behaviorKey = behaviorKeys[__VU % behaviorKeys.length];
  const behavior = userBehaviors[behaviorKey];
  
  // Select query parameters
  const query = behavior.queryTypes[Math.floor(Math.random() * behavior.queryTypes.length)];
  const limit = behavior.limits[Math.floor(Math.random() * behavior.limits.length)];
  const site = testSites[Math.floor(Math.random() * testSites.length)];
  
  // Prepare request payload
  const payload = JSON.stringify({
    query: query,
    limit: limit,
    site: site
  });
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY && { 'X-API-Key': API_KEY })
    },
    timeout: '45s', // Longer timeout for stress conditions
  };
  
  // Record concurrent users
  concurrentUsers.add(__VU);
  
  // Execute search with timing
  const searchStart = Date.now();
  const response = http.post(`${data.apiUrl}/query`, payload, params);
  const searchDuration = Date.now() - searchStart;
  
  // Record metrics
  searchLatency.add(searchDuration);
  errorRate.add(response.status !== 200);
  
  // Comprehensive response validation
  const isSuccess = check(response, {
    'status is 200': (r) => r.status === 200,
    'status is not 5xx': (r) => r.status < 500,
    'has results': (r) => {
      if (r.status !== 200) return true; // Skip validation for non-200
      try {
        const body = JSON.parse(r.body);
        return body.results && Array.isArray(body.results);
      } catch (e) {
        return false;
      }
    },
    'response time acceptable': (r) => r.timings.duration < 5000, // 5s max under stress
    'valid JSON response': (r) => {
      if (r.status !== 200) return true; // Skip validation for non-200
      try {
        JSON.parse(r.body);
        return true;
      } catch (e) {
        return false;
      }
    }
  });
  
  // Log errors for debugging
  if (!isSuccess && response.status >= 500) {
    console.error(`🚨 Server error: ${response.status} - VU:${__VU} Iter:${__ITER}`);
  }
  
  // Variable sleep based on behavior pattern
  const sleepTime = Math.random() * (behavior.sleepRange[1] - behavior.sleepRange[0]) + behavior.sleepRange[0];
  sleep(sleepTime);
}

export function teardown(data) {
  console.log('🏁 DocFoundry stress test completed');
  console.log('Check metrics for performance bottlenecks and error rates');
}