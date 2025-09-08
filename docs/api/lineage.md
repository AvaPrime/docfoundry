# Data Lineage API Reference

The DocFoundry Data Lineage API provides comprehensive tracking and management of document processing workflows, enabling you to monitor data flow, identify reprocessing candidates, and maintain audit trails.

## Base URL
```
http://localhost:8000/api/v1/lineage
```

## Authentication
All API endpoints require authentication via API key:
```http
Authorization: Bearer YOUR_API_KEY
```

## Rate Limiting
- **Default**: 100 requests per minute per API key
- **Burst**: Up to 200 requests in a 10-second window
- Rate limit headers are included in all responses

## Endpoints

### 1. Get Lineage Summary
Retrieve high-level statistics about data lineage and processing status.

**Endpoint**: `GET /summary`

**Parameters**:
- `days` (optional, integer): Number of days to include in summary (default: 7, max: 90)
- `include_metrics` (optional, boolean): Include performance metrics (default: false)

**Response**:
```json
{
  "total_documents": 15420,
  "processed_today": 342,
  "reprocessing_candidates": 28,
  "orphaned_data": 5,
  "processing_rate": {
    "documents_per_hour": 145,
    "avg_processing_time_ms": 2340
  },
  "lineage_health": {
    "status": "healthy",
    "last_cleanup": "2024-01-15T10:30:00Z",
    "integrity_score": 0.98
  }
}
```

### 2. Get Reprocessing Candidates
Identify documents that may need reprocessing based on various criteria.

**Endpoint**: `GET /reprocessing-candidates`

**Parameters**:
- `reason` (optional, string): Filter by reason (`content_changed`, `schema_updated`, `processing_failed`, `manual_flag`)
- `limit` (optional, integer): Maximum results to return (default: 50, max: 500)
- `offset` (optional, integer): Pagination offset (default: 0)
- `priority` (optional, string): Filter by priority (`high`, `medium`, `low`)

**Response**:
```json
{
  "candidates": [
    {
      "document_id": "doc_123",
      "source_url": "https://example.com/doc1.pdf",
      "reason": "content_changed",
      "priority": "high",
      "last_processed": "2024-01-10T14:20:00Z",
      "content_hash_changed": true,
      "estimated_processing_time_ms": 3200,
      "dependencies": ["doc_124", "doc_125"]
    }
  ],
  "total_count": 28,
  "has_more": false
}
```

### 3. Get Reprocessing Statistics
Retrieve detailed statistics about reprocessing operations.

**Endpoint**: `GET /reprocessing/stats`

**Parameters**:
- `period` (optional, string): Time period (`1h`, `24h`, `7d`, `30d`) (default: `24h`)
- `group_by` (optional, string): Group results by (`hour`, `day`, `reason`, `status`)

**Response**:
```json
{
  "period": "24h",
  "total_reprocessed": 156,
  "success_rate": 0.94,
  "avg_processing_time_ms": 2840,
  "breakdown": {
    "by_reason": {
      "content_changed": 89,
      "schema_updated": 34,
      "processing_failed": 23,
      "manual_flag": 10
    },
    "by_status": {
      "completed": 147,
      "failed": 9
    }
  },
  "performance_trends": [
    {
      "hour": "2024-01-15T10:00:00Z",
      "processed": 12,
      "avg_time_ms": 2650
    }
  ]
}
```

### 4. Get Data Audit Trail
Retrieve comprehensive audit information for data processing.

**Endpoint**: `GET /audit`

**Parameters**:
- `document_id` (optional, string): Filter by specific document
- `action` (optional, string): Filter by action type (`created`, `updated`, `deleted`, `reprocessed`)
- `start_date` (optional, ISO date): Start date for audit trail
- `end_date` (optional, ISO date): End date for audit trail
- `limit` (optional, integer): Maximum results (default: 100, max: 1000)

**Response**:
```json
{
  "audit_entries": [
    {
      "id": "audit_789",
      "document_id": "doc_123",
      "action": "reprocessed",
      "timestamp": "2024-01-15T11:45:00Z",
      "user_id": "system",
      "reason": "content_changed",
      "metadata": {
        "previous_hash": "abc123",
        "new_hash": "def456",
        "processing_time_ms": 2340
      },
      "changes": {
        "content_length": {
          "old": 15420,
          "new": 16890
        }
      }
    }
  ],
  "total_count": 1247,
  "has_more": true
}
```

### 5. Get Document Lineage
Retrieve complete lineage information for a specific document.

**Endpoint**: `GET /document/{document_id}`

**Path Parameters**:
- `document_id` (required, string): Unique document identifier

**Query Parameters**:
- `include_dependencies` (optional, boolean): Include dependent documents (default: true)
- `depth` (optional, integer): Maximum dependency depth to traverse (default: 3, max: 10)

**Response**:
```json
{
  "document_id": "doc_123",
  "source_url": "https://example.com/doc1.pdf",
  "current_status": "processed",
  "created_at": "2024-01-10T09:15:00Z",
  "last_processed": "2024-01-15T11:45:00Z",
  "processing_history": [
    {
      "timestamp": "2024-01-15T11:45:00Z",
      "action": "reprocessed",
      "reason": "content_changed",
      "duration_ms": 2340,
      "status": "success"
    }
  ],
  "content_metadata": {
    "hash": "def456",
    "size_bytes": 16890,
    "mime_type": "application/pdf",
    "page_count": 12
  },
  "dependencies": {
    "upstream": [],
    "downstream": ["doc_124", "doc_125"]
  },
  "lineage_metrics": {
    "reprocessing_count": 3,
    "avg_processing_time_ms": 2450,
    "last_content_change": "2024-01-15T10:30:00Z"
  }
}
```

### 6. Trigger Single Document Reprocessing
Initiate reprocessing for a specific document.

**Endpoint**: `POST /reprocess/{document_id}`

**Path Parameters**:
- `document_id` (required, string): Document to reprocess

**Request Body**:
```json
{
  "reason": "manual_trigger",
  "priority": "high",
  "force": false,
  "metadata": {
    "user_id": "user_456",
    "notes": "Content updated by user"
  }
}
```

**Response**:
```json
{
  "job_id": "job_789",
  "document_id": "doc_123",
  "status": "queued",
  "estimated_completion": "2024-01-15T12:15:00Z",
  "priority": "high",
  "queue_position": 3
}
```

### 7. Trigger Bulk Reprocessing
Initiate reprocessing for multiple documents.

**Endpoint**: `POST /reprocess/bulk`

**Request Body**:
```json
{
  "document_ids": ["doc_123", "doc_124", "doc_125"],
  "reason": "schema_updated",
  "priority": "medium",
  "batch_size": 10,
  "metadata": {
    "user_id": "user_456",
    "batch_name": "Q1_2024_update"
  }
}
```

**Response**:
```json
{
  "batch_id": "batch_456",
  "total_documents": 3,
  "jobs_created": [
    {
      "job_id": "job_789",
      "document_id": "doc_123",
      "status": "queued"
    }
  ],
  "estimated_completion": "2024-01-15T12:45:00Z",
  "batch_status": "queued"
}
```

### 8. Clean Up Orphaned Data
Remove orphaned lineage data and optimize storage.

**Endpoint**: `POST /cleanup`

**Request Body**:
```json
{
  "dry_run": false,
  "older_than_days": 90,
  "cleanup_types": ["orphaned_entries", "old_audit_logs", "failed_jobs"]
}
```

**Response**:
```json
{
  "cleanup_id": "cleanup_123",
  "status": "completed",
  "summary": {
    "orphaned_entries_removed": 45,
    "audit_logs_archived": 1200,
    "failed_jobs_cleaned": 8,
    "storage_freed_mb": 156.7
  },
  "duration_ms": 5670
}
```

### 9. Health Check
Check the health status of the lineage system.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:00:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12,
      "connection_pool": {
        "active": 5,
        "idle": 15,
        "max": 20
      }
    },
    "queue": {
      "status": "healthy",
      "pending_jobs": 23,
      "processing_rate": 145.6
    },
    "storage": {
      "status": "healthy",
      "usage_percent": 67.3,
      "available_gb": 234.5
    }
  },
  "metrics": {
    "uptime_seconds": 86400,
    "total_requests": 15420,
    "error_rate": 0.02
  }
}
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid document_id format",
    "details": {
      "field": "document_id",
      "expected": "string matching pattern ^doc_[a-zA-Z0-9]+$"
    },
    "request_id": "req_789",
    "timestamp": "2024-01-15T12:00:00Z"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Client Examples

### Python Client
```python
import requests
import json

class LineageClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_summary(self, days=7):
        response = requests.get(
            f'{self.base_url}/summary',
            headers=self.headers,
            params={'days': days}
        )
        return response.json()
    
    def get_reprocessing_candidates(self, reason=None, limit=50):
        params = {'limit': limit}
        if reason:
            params['reason'] = reason
        
        response = requests.get(
            f'{self.base_url}/reprocessing-candidates',
            headers=self.headers,
            params=params
        )
        return response.json()
    
    def trigger_reprocessing(self, document_id, reason='manual_trigger'):
        data = {'reason': reason, 'priority': 'medium'}
        response = requests.post(
            f'{self.base_url}/reprocess/{document_id}',
            headers=self.headers,
            json=data
        )
        return response.json()

# Usage
client = LineageClient('http://localhost:8000/api/v1/lineage', 'your-api-key')
summary = client.get_summary(days=30)
print(f"Total documents: {summary['total_documents']}")
```

### JavaScript Client
```javascript
class LineageClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async getSummary(days = 7) {
        const response = await fetch(`${this.baseUrl}/summary?days=${days}`, {
            headers: this.headers
        });
        return response.json();
    }
    
    async getReprocessingCandidates(options = {}) {
        const params = new URLSearchParams({
            limit: options.limit || 50,
            ...(options.reason && { reason: options.reason })
        });
        
        const response = await fetch(
            `${this.baseUrl}/reprocessing-candidates?${params}`,
            { headers: this.headers }
        );
        return response.json();
    }
    
    async triggerReprocessing(documentId, reason = 'manual_trigger') {
        const response = await fetch(
            `${this.baseUrl}/reprocess/${documentId}`,
            {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify({ reason, priority: 'medium' })
            }
        );
        return response.json();
    }
}

// Usage
const client = new LineageClient('http://localhost:8000/api/v1/lineage', 'your-api-key');
client.getSummary(30).then(summary => {
    console.log(`Total documents: ${summary.total_documents}`);
});
```

## Best Practices

### Performance
- Use pagination for large result sets
- Cache frequently accessed data
- Implement exponential backoff for retries
- Monitor rate limits and adjust request frequency

### Error Handling
- Always check HTTP status codes
- Implement proper retry logic for transient errors
- Log request_id for debugging
- Handle rate limiting gracefully

### Security
- Store API keys securely
- Use HTTPS in production
- Implement proper access controls
- Regularly rotate API keys

### Observability
- Monitor API response times
- Track error rates and patterns
- Set up alerts for critical failures
- Use request_id for distributed tracing

## Rate Limiting Details

The API implements a token bucket algorithm with the following limits:

- **Standard endpoints**: 100 requests/minute
- **Bulk operations**: 10 requests/minute
- **Health checks**: 300 requests/minute

Rate limit headers in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
X-RateLimit-Retry-After: 60
```

## Webhook Support

Configure webhooks to receive real-time notifications:

```json
{
  "webhook_url": "https://your-app.com/webhooks/lineage",
  "events": ["document.reprocessed", "cleanup.completed"],
  "secret": "your-webhook-secret"
}
```

Webhook payload example:
```json
{
  "event": "document.reprocessed",
  "timestamp": "2024-01-15T12:00:00Z",
  "data": {
    "document_id": "doc_123",
    "status": "completed",
    "processing_time_ms": 2340
  }
}
```