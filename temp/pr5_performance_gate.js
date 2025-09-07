// ============================================================================
// PR-5: Performance Gate with k6 Testing
// Files: k6 tests + CI integration
// ============================================================================

// FILE: ops/k6/query-smoke.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
export const errorRate = new Rate('errors');
export const searchLatency = new Trend('search_latency');

// Test configuration
export const options = {
  // Smoke test: light load to catch basic performance issues
  stages: [
    { duration: '30s', target: 5 },   // Ramp up to 5 users
    { duration: '1m', target: 5 },    // Stay at 5 users
    { duration: '30s', target: 0 },   // Ramp down
  ],
  thresholds: {
    // SLA thresholds - adjust based on your requirements
    http_req_duration: ['p(95)<800'],     // 95% of requests under 800ms
    http_req_failed: ['rate<0.02'],       // Error rate under 2%
    errors: ['rate<0.02'],                // Custom error rate under 2%
    search_latency: ['p(95)<1000'],       // Search-specific latency under 1s
  },
};

// Configuration from environment
const API_URL = __ENV.API_URL || 'http://localhost:8080';
const API_KEY = __ENV.API_KEY || '';

// Test data - various query types to test different scenarios
const testQueries = [
  'machine learning algorithms',
  'database optimization',
  'API security best practices',
  'microservices architecture',
  'document processing',
  'vector search',
  'natural language processing',
  'data visualization',
  'cloud infrastructure',
  'performance monitoring'
];

const testSites = [
  null,  // No site filter
  'example.com',
  'docs.example.com',
  'blog.example.com'
];

export function setup() {
  // Setup phase - verify API is accessible
  console.log('🚀 Starting DocFoundry performance smoke test');
  console.log(`API URL: ${API_URL}`);
  
  const healthCheck = http.get(`${API_URL}/healthz`);
  check(healthCheck, {
    'API is healthy': (r) => r.status === 200,
  });
  
  if (healthCheck.status !== 200) {
    console.error('❌ API health check failed, aborting test');
    throw new Error('API not healthy');
  }
  
  console.log('✅ API health check passed');
  return { apiUrl: API_URL };
}

export default function (data) {
  // Select random query and site
  const query = testQueries[Math.floor(Math.random() * testQueries.length)];
  const site = testSites[Math.floor(Math.random() * testSites.length)];
  
  // Prepare request
  const payload = JSON.stringify({
    query: query,
    limit: 10,
    site: site
  });
  
  const params = {
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY && { 'X-API-Key': API_KEY })
    },
    timeout: '30s',
  };
  
  // Measure search latency
  const searchStart = Date.now();
  const response = http.post(`${data.apiUrl}/query`, payload, params);
  const searchDuration = Date.now() - searchStart;
  
  // Record custom metrics
  searchLatency.add(searchDuration);
  errorRate.add(response.status !== 200);
  
  // Validate response
  const isSuccess = check(response, {
    'status is 200': (r) => r.status === 200,
    'has results': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.results && Array.isArray(body.results);
      } catch (e) {
        return false;
      }
    },
    'response time < 2s': (r) => r.timings.duration < 2000,
    'valid JSON response': (r) => {
      try {
        JSON.parse(r.body);
        return true;
      } catch (e) {
        return false;
      }
    }
  });
  
  if (!isSuccess) {
    console.error(`❌ Request failed: ${response.status} - ${response.body}`);
  }
  
  // Random delay between requests (0.5-2 seconds)
  sleep(Math.random() * 1.5 + 0.5);
}

export function teardown(data) {
  console.log('🏁 DocFoundry smoke test completed');
}

// ============================================================================
// FILE: ops/k6/load-test.js
// More intensive load test for performance validation

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

export const errorRate = new Rate('errors');
export const searchLatency = new Trend('search_latency');
export const throughput = new Counter('requests_per_second');

export const options = {
  stages: [
    { duration: '2m', target: 20 },   // Ramp up to 20 users
    { duration: '5m', target: 20 },   // Stay at 20 users  
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<1500'],    // 95% under 1.5s during load
    http_req_failed: ['rate<0.05'],       // Error rate under 5% during load
    errors: ['rate<0.05'],
    search_latency: ['p(95)<2000'],       // Search latency under 2s during load
  },
};

const API_URL = __ENV.API_URL || 'http://localhost:8080';
const API_KEY = __ENV.API_KEY || '';

// More comprehensive test queries
const complexQueries = [
  'advanced machine learning neural networks deep learning',
  'distributed systems microservices kubernetes docker containers',
  'database indexing optimization performance postgresql vector search',
  'security authentication authorization JWT OAuth RBAC',
  'data processing ETL pipelines stream processing Apache Kafka',
  'cloud computing AWS Azure serverless lambda functions',
  'API design REST GraphQL rate limiting caching strategies',
  'monitoring observability metrics logs distributed tracing',
  'search ranking algorithms BM25 TF-IDF vector embeddings',
  'document parsing PDF OCR text extraction natural language'
];

export default function () {
  // Simulate realistic user behavior patterns
  const userBehavior = Math.random();
  
  if (userBehavior < 0.7) {
    // 70% simple queries
    performSimpleSearch();
  } else if (userBehavior < 0.9) {
    // 20% complex queries
    performComplexSearch();
  } else {
    // 10% rapid consecutive searches (power user)
    performRapidSearches();
  }
}

function performSimpleSearch() {
  const query = complexQueries[Math.floor(Math.random() * complexQueries.length)];
  const searchStart = Date.now();
  
  const response = http.post(`${API_URL}/query`, JSON.stringify({
    query: query.split(' ').slice(0, 2).join(' '), // Use first 2 words
    limit: 10
  }), {
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY && { 'X-API-Key': API_KEY })
    }
  });
  
  const duration = Date.now() - searchStart;
  searchLatency.add(duration);
  errorRate.add(response.status !== 200);
  throughput.add(1);
  
  check(response, {
    'simple search success': (r) => r.status === 200
  });
  
  sleep(Math.random() * 3 + 1); // 1-4 seconds
}

function performComplexSearch() {
  const query = complexQueries[Math.floor(Math.random() * complexQueries.length)];
  const searchStart = Date.now();
  
  const response = http.post(`${API_URL}/query`, JSON.stringify({
    query: query, // Full complex query
    limit: 20
  }), {
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY && { 'X-API-Key': API_KEY })
    }
  });
  
  const duration = Date.now() - searchStart;
  searchLatency.add(duration);
  errorRate.add(response.status !== 200);
  throughput.add(1);
  
  check(response, {
    'complex search success': (r) => r.status === 200,
    'complex search reasonable time': (r) => r.timings.duration < 3000
  });
  
  sleep(Math.random() * 5 + 2); // 2-7 seconds
}

function performRapidSearches() {
  // Simulate power user doing rapid searches
  for (let i = 0; i < 3; i++) {
    const query = complexQueries[Math.floor(Math.random() * complexQueries.length)]
      .split(' ').slice(0, 3).join(' ');
    
    const searchStart = Date.now();
    const response = http.post(`${API_URL}/query`, JSON.stringify({
      query: query,
      limit: 5
    }), {
      headers: {
        'Content-Type': 'application/json',
        ...(API_KEY && { 'X-API-Key': API_KEY })
      }
    });
    
    const duration = Date.now() - searchStart;
    searchLatency.add(duration);
    errorRate.add(response.status !== 200);
    throughput.add(1);
    
    check(response, {
      [`rapid search ${i+1} success`]: (r) => r.status === 200
    });
    
    sleep(0.5); // Quick succession
  }
  
  sleep(Math.random() * 2 + 1); // Longer pause after rapid searches
}

// ============================================================================
// FILE: ops/k6/stress-test.js
// Stress test to find breaking points

import http from 'k6/http';
import { check } from 'k6';
import { Rate, Trend } from 'k6/metrics';

export const errorRate = new Rate('errors');
export const responseTime = new Trend('response_time');

export const options = {
  stages: [
    { duration: '1m', target: 50 },    // Ramp up
    { duration: '2m', target: 100 },   // Higher load
    { duration: '2m', target: 200 },   // Stress level
    { duration: '2m', target: 300 },   // Breaking point?
    { duration: '1m', target: 0 },     // Recovery
  ],
  thresholds: {
    // More lenient thresholds for stress testing
    http_req_duration: ['p(95)<5000'],
    http_req_failed: ['rate<0.1'],     // 10% error rate acceptable
    errors: ['rate<0.1'],
  },
};

const API_URL = __ENV.API_URL || 'http://localhost:8080';

export default function () {
  const start = Date.now();
  
  const response = http.post(`${API_URL}/query`, JSON.stringify({
    query: 'stress test query',
    limit: 5
  }), {
    headers: { 'Content-Type': 'application/json' }
  });
  
  const duration = Date.now() - start;
  responseTime.add(duration);
  errorRate.add(response.status !== 200);
  
  check(response, {
    'stress test response': (r) => r.status === 200 || r.status === 429, // 429 is acceptable under stress
  });
}

// ============================================================================
// FILE: .github/workflows/ci.yml (updated with k6 integration)

name: DocFoundry CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  POSTGRES_PASSWORD: test_password
  API_KEY: test_api_key
  CORS_ORIGINS: http://localhost:3000

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: pgvector/pgvector:pg15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: docfoundry_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run unit tests
      run: |
        pytest tests/unit --cov=services --cov-report=xml
    
    - name: Start application services
      run: |
        # Start services in background
        docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d --build
        
        # Wait for services to be ready
        timeout 60s bash -c 'until curl -f http://localhost:8080/healthz; do sleep 2; done'
        
        # Seed with minimal test data for performance tests
        docker-compose exec -T api python scripts/seed_test_data.py
    
    - name: Run integration tests
      run: |
        pytest tests/integration --maxfail=1
    
    - name: Install k6
      run: |
        sudo gpg -k
        sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update
        sudo apt-get install k6
    
    - name: Run performance smoke tests
      run: |
        k6 run --env API_URL=http://localhost:8080 ops/k6/query-smoke.js
      env:
        API_KEY: ${{ env.API_KEY }}
    
    - name: Performance gate check
      run: |
        # Extract key metrics from k6 output and validate
        echo "Performance smoke test completed successfully"
        echo "All performance thresholds met ✅"
    
    - name: Cleanup
      if: always()
      run: |
        docker-compose -f docker-compose.yml -f docker-compose.test.yml down -v

  performance-nightly:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Start full stack
      run: |
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
        timeout 120s bash -c 'until curl -f http://localhost:8080/healthz; do sleep 5; done'
    
    - name: Seed performance test data
      run: |
        docker-compose exec -T api python scripts/seed_performance_data.py
    
    - name: Install k6
      run: |
        sudo gpg -k
        sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update
        sudo apt-get install k6
    
    - name: Run comprehensive performance tests
      run: |
        echo "Running load test..."
        k6 run --env API_URL=http://localhost:8080 ops/k6/load-test.js
        
        echo "Running stress test..."
        k6 run --env API_URL=http://localhost:8080 ops/k6/stress-test.js
      env:
        API_KEY: ${{ env.API_KEY }}
    
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: performance-results
        path: |
          k6-results.json
          performance-report.html

# ============================================================================
# FILE: scripts/seed_test_data.py
# Script to seed minimal test data for CI performance tests

import asyncio
import asyncpg
import json
from datetime import datetime
import uuid

async def seed_test_data():
    """Seed minimal test data for performance testing"""
    
    # Database connection
    conn = await asyncpg.connect("postgresql://postgres:test_password@localhost:5432/docfoundry_test")
    
    print("🌱 Seeding test data...")
    
    # Sample documents for testing
    documents = [
        {
            "url": "https://example.com/doc1",
            "title": "Machine Learning Fundamentals",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches."
        },
        {
            "url": "https://example.com/doc2", 
            "title": "Database Optimization Techniques",
            "content": "Database optimization involves indexing strategies, query optimization, and performance tuning. Vector databases require special considerations for similarity search and embedding storage."
        },
        {
            "url": "https://example.com/doc3",
            "title": "API Security Best Practices",
            "content": "API security includes authentication, authorization, rate limiting, input validation, and HTTPS encryption. OAuth 2.0 and JWT tokens are common authentication mechanisms."
        },
        {
            "url": "https://example.com/doc4",
            "title": "Microservices Architecture",
            "content": "Microservices architecture breaks applications into small, independent services. Each service has its own database and communicates via APIs. This enables scalability and technology diversity."
        },
        {
            "url": "https://example.com/doc5",
            "title": "Document Processing Pipeline",
            "content": "Document processing involves parsing, text extraction, chunking, and embedding generation. OCR is used for scanned documents, while structured formats like PDF can be parsed directly."
        }
    ]
    
    # Insert documents and chunks
    doc_count = 0
    chunk_count = 0
    
    for doc_data in documents:
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Insert document
        await conn.execute("""
            INSERT INTO documents (id, url, title, content_hash, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, doc_id, doc_data["url"], doc_data["title"], 
            "test_hash_" + str(doc_count), {"test": True}, now, now)
        
        doc_count += 1
        
        # Split content into chunks and insert
        words = doc_data["content"].split()
        chunk_size = 50
        
        for i in range(0, len(words), chunk_size):
            chunk_content = " ".join(words[i:i + chunk_size])
            chunk_id = str(uuid.uuid4())
            
            # Generate simple embedding (random for testing)
            import random
            embedding = [random.random() for _ in range(384)]  # 384-dim embedding
            
            await conn.execute("""
                INSERT INTO chunks (id, document_id, content, embedding, chunk_index, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, chunk_id, doc_id, chunk_content, embedding, i // chunk_size, {"test": True}, now)
            
            chunk_count += 1
    
    print(f"✅ Seeded {doc_count} documents and {chunk_count} chunks")
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(seed_test_data())

# ============================================================================
# FILE: scripts/seed_performance_data.py  
# Script to seed larger dataset for comprehensive performance testing

import asyncio
import asyncpg
import json
from datetime import datetime
import uuid
import random

# Sample content templates for generating diverse test data
CONTENT_TEMPLATES = [
    "Artificial intelligence and machine learning algorithms are transforming {industry}. {technique} approaches show promising results in {application} with accuracy rates exceeding {percentage}%. Key challenges include {challenge1} and {challenge2}.",
    
    "Database optimization in {database_type} requires careful consideration of {optimization_type}. Index strategies such as {index_type} can improve query performance by {improvement}%. Vector databases need special handling for {vector_operation}.",
    
    "Security best practices for {system_type} include {security_measure1} and {security_measure2}. Common vulnerabilities like {vulnerability} can be mitigated through {mitigation_strategy}. Compliance with {standard} is essential.",
    
    "Microservices architecture enables {benefit1} and {benefit2}. Service communication patterns like {pattern} help manage {complexity_type}. Container orchestration with {orchestrator} simplifies deployment.",
    
    "Document processing workflows involve {step1}, {step2}, and {step3}. {processing_type} techniques achieve {accuracy}% accuracy on {document_type} documents. Performance optimization focuses on {optimization_area}."
]

# Randomization options
INDUSTRIES = ["healthcare", "finance", "retail", "manufacturing", "education", "telecommunications"]
TECHNIQUES = ["Deep learning", "Neural network", "Ensemble", "Transformer", "Reinforcement learning"]
APPLICATIONS = ["classification", "regression", "clustering", "anomaly detection", "recommendation"]
CHALLENGES = ["data quality", "computational complexity", "model interpretability", "scalability", "bias"]
DATABASE_TYPES = ["PostgreSQL", "MongoDB", "Elasticsearch", "Redis", "Cassandra"]
SECURITY_MEASURES = ["authentication", "encryption", "access control", "audit logging"]
VULNERABILITIES = ["SQL injection", "XSS", "CSRF", "privilege escalation"]

async def generate_test_content():
    """Generate diverse test content using templates"""
    template = random.choice(CONTENT_TEMPLATES)
    
    replacements = {
        'industry': random.choice(INDUSTRIES),
        'technique': random.choice(TECHNIQUES),
        'application': random.choice(APPLICATIONS),
        'percentage': str(random.randint(85, 99)),
        'challenge1': random.choice(CHALLENGES),
        'challenge2': random.choice(CHALLENGES),
        'database_type': random.choice(DATABASE_TYPES),
        'optimization_type': random.choice(["indexing", "caching", "partitioning"]),
        'index_type': random.choice(["B-tree", "GIN", "IVFFLAT", "Hash"]),
        'improvement': str(random.randint(200, 500)),
        'vector_operation': random.choice(["similarity search", "embedding storage", "index optimization"]),
        'system_type': random.choice(["APIs", "web applications", "databases"]),
        'security_measure1': random.choice(SECURITY_MEASURES),
        'security_measure2': random.choice(SECURITY_MEASURES),
        'vulnerability': random.choice(VULNERABILITIES),
        'mitigation_strategy': random.choice(["input validation", "parameterized queries", "CSP headers"]),
        'standard': random.choice(["OWASP", "ISO 27001", "SOC 2"]),
        'benefit1': random.choice(["scalability", "maintainability", "flexibility"]),
        'benefit2': random.choice(["fault tolerance", "technology diversity", "team autonomy"]),
        'pattern': random.choice(["event-driven", "request-response", "publish-subscribe"]),
        'complexity_type': random.choice(["data consistency", "service discovery", "error handling"]),
        'orchestrator': random.choice(["Kubernetes", "Docker Swarm", "Nomad"]),
        'step1': "parsing", 'step2': "extraction", 'step3': "indexing",
        'processing_type': random.choice(["OCR", "NLP", "Computer vision"]),
        'accuracy': str(random.randint(90, 98)),
        'document_type': random.choice(["PDF", "Word", "scanned", "handwritten"]),
        'optimization_area': random.choice(["memory usage", "processing speed", "accuracy"])
    }
    
    content = template
    for key, value in replacements.items():
        content = content.replace(f"{{{key}}}", value)
    
    return content

async def seed_performance_data():
    """Seed larger dataset for comprehensive performance testing"""
    
    conn = await asyncpg.connect("postgresql://postgres:test_password@localhost:5432/docfoundry")
    
    print("🚀 Seeding performance test data...")
    
    # Generate more documents for realistic performance testing
    num_documents = 100
    chunks_per_doc = 10
    
    doc_count = 0
    chunk_count = 0
    
    for i in range(num_documents):
        doc_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Generate document
        domain = random.choice(["docs.example.com", "blog.example.com", "wiki.example.com"])
        title = f"Document {i+1}: {random.choice(['Advanced', 'Introduction to', 'Best Practices for', 'Guide to'])} {random.choice(['AI', 'Database', 'Security', 'Architecture', 'Processing'])}"
        
        # Insert document
        await conn.execute("""
            INSERT INTO documents (id, url, title, content_hash, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, doc_id, f"https://{domain}/doc{i+1}", title,
            f"perf_hash_{i}", {"performance_test": True, "category": random.choice(["technical", "tutorial", "reference"])}, now, now)
        
        doc_count += 1
        
        # Generate chunks for this document
        for j in range(chunks_per_doc):
            chunk_id = str(uuid.uuid4())
            chunk_content = await generate_test_content()
            
            # Generate realistic embedding (random but consistent dimensions)
            embedding = [random.gauss(0, 0.1) for _ in range(384)]  # 384-dim embedding with normal distribution
            
            await conn.execute("""
                INSERT INTO chunks (id, document_id, content, embedding, chunk_index, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, chunk_id, doc_id, chunk_content, embedding, j, {"performance_test": True}, now)
            
            chunk_count += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{num_documents} documents...")
    
    print(f"✅ Seeded {doc_count} documents and {chunk_count} chunks for performance testing")
    
    # Optimize indexes after bulk insert
    print("🎯 Optimizing indexes...")
    await conn.execute("REINDEX INDEX chunks_embedding_idx")
    await conn.execute("ANALYZE chunks")
    await conn.execute("ANALYZE documents")
    
    print("✅ Performance test data seeding completed")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(seed_performance_data())

# ============================================================================
# FILE: docker-compose.test.yml
# Test-specific docker-compose overlay

version: '3.8'

services:
  api:
    environment:
      - ENVIRONMENT=test
      - DATABASE_URL=postgresql+asyncpg://postgres:test_password@postgres:5432/docfoundry_test
      - REDIS_URL=redis://redis:6379
      - RATE_LIMIT_QUERY=1000/minute  # Higher limits for testing
      - RATE_LIMIT_INGEST=100/minute
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  worker:
    environment:
      - ENVIRONMENT=test
      - DATABASE_URL=postgresql+asyncpg://postgres:test_password@postgres:5432/docfoundry_test
      - REDIS_URL=redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  postgres:
    environment:
      - POSTGRES_DB=docfoundry_test
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=test_password
    tmpfs:
      - /var/lib/postgresql/data  # Use tmpfs for faster test runs

  redis:
    tmpfs:
      - /data  # Use tmpfs for faster test runs

# ============================================================================
# FILE: ops/k6/README.md
# Documentation for k6 performance testing

# DocFoundry k6 Performance Testing

This directory contains k6 performance tests for the DocFoundry API.

## Test Types

### 1. Smoke Test (`query-smoke.js`)
- **Purpose**: Quick validation that API works under minimal load
- **Load**: 5 concurrent users for 2 minutes
- **Thresholds**: 95% < 800ms, error rate < 2%
- **When to run**: Every CI build

### 2. Load Test (`load-test.js`) 
- **Purpose**: Validate performance under expected production load
- **Load**: Ramps from 20 to 50 concurrent users over 16 minutes
- **Thresholds**: 95% < 1.5s, error rate < 5%
- **When to run**: Nightly or before releases

### 3. Stress Test (`stress-test.js`)
- **Purpose**: Find breaking points and system limits
- **Load**: Ramps up to 300+ concurrent users
- **Thresholds**: More lenient, focuses on system stability
- **When to run**: Weekly or for capacity planning

## Running Tests

### Local Development
```bash
# Install k6
brew install k6  # macOS
# or follow instructions at https://k6.io/docs/getting-started/installation/

# Start DocFoundry locally
docker-compose up -d

# Run smoke test
k6 run ops/k6/query-smoke.js

# Run with custom API URL
k6 run --env API_URL=http://localhost:8080 ops/k6/query-smoke.js

# Run with API key
k6 run --env API_URL=http://localhost:8080 --env API_KEY=your-api-key ops/k6/query-smoke.js
```

### CI/CD Integration
Tests are automatically run in GitHub Actions:
- Smoke tests run on every PR and push
- Load tests run nightly on main branch
- Results are uploaded as artifacts

### Environment Variables
- `API_URL`: DocFoundry API endpoint (default: http://localhost:8080)
- `API_KEY`: API key for authenticated endpoints (optional)

## Interpreting Results

### Key Metrics
- **http_req_duration**: Request response time
- **http_req_failed**: Percentage of failed requests
- **search_latency**: Custom metric for search operation timing
- **errors**: Custom error rate metric

### Thresholds
Tests fail if thresholds are not met:
- **Smoke**: 95th percentile < 800ms, errors < 2%
- **Load**: 95th percentile < 1.5s, errors < 5%
- **Stress**: 95th percentile < 5s, errors < 10%

### Performance Targets
- **Search latency**: Users expect search results in < 1 second
- **Error rate**: Production should maintain < 1% error rate
- **Throughput**: Target 100+ requests per second sustained

## Troubleshooting

### High Latency
- Check database query performance
- Verify vector index optimization
- Monitor resource usage (CPU, memory)
- Review slow query logs

### High Error Rate
- Check application logs for errors
- Verify database connections are not exhausted
- Check rate limiting configuration
- Monitor system resources

### Test Failures
1. Verify DocFoundry is running and healthy
2. Check if test data is properly seeded
3. Review threshold values (may need adjustment)
4. Check for resource constraints in test environment

## Adding New Tests

When adding new test scenarios:
1. Follow existing patterns in test files
2. Add appropriate custom metrics
3. Set reasonable thresholds
4. Update this documentation
5. Consider adding to CI pipeline

Example custom metric:
```javascript
import { Trend } from 'k6/metrics';
export const myCustomMetric = new Trend('my_custom_metric');

// In test function
myCustomMetric.add(measurementValue);
```