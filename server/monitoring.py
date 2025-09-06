"""Enhanced monitoring and metrics collection for DocFoundry API.

Provides comprehensive performance monitoring, health checks, and metrics
collection for production deployment.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import psutil
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for individual requests."""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    query_params: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_connections: int
    timestamp: datetime


@dataclass
class SearchMetrics:
    """Search-specific performance metrics."""
    query: str
    search_type: str  # 'semantic', 'hybrid', 'fulltext'
    results_count: int
    response_time: float
    cache_hit: bool
    embedding_time: Optional[float] = None
    db_query_time: Optional[float] = None
    rerank_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Centralized metrics collection and aggregation."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.request_metrics: deque = deque(maxlen=max_history)
        self.search_metrics: deque = deque(maxlen=max_history)
        self.system_metrics: deque = deque(maxlen=max_history)
        
        # Real-time counters
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
        # Performance tracking
        self.response_times = defaultdict(list)
        self.active_requests = 0
        
        # Start background system monitoring
        self._monitoring_task = None
        self._start_system_monitoring()
    
    def _start_system_monitoring(self):
        """Start background task for system metrics collection."""
        async def monitor_system():
            while True:
                try:
                    metrics = SystemMetrics(
                        cpu_percent=psutil.cpu_percent(interval=1),
                        memory_percent=psutil.virtual_memory().percent,
                        memory_used_mb=psutil.virtual_memory().used / 1024 / 1024,
                        disk_usage_percent=psutil.disk_usage('/').percent,
                        active_connections=self.active_requests,
                        timestamp=datetime.now()
                    )
                    self.system_metrics.append(metrics)
                    await asyncio.sleep(30)  # Collect every 30 seconds
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    await asyncio.sleep(60)
        
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(monitor_system())
    
    def record_request(self, metrics: RequestMetrics):
        """Record request metrics."""
        self.request_metrics.append(metrics)
        
        # Update counters
        endpoint_key = f"{metrics.method}:{metrics.endpoint}"
        self.request_counts[endpoint_key] += 1
        
        if metrics.status_code >= 400:
            self.error_counts[endpoint_key] += 1
        
        # Track response times
        self.response_times[endpoint_key].append(metrics.response_time)
        if len(self.response_times[endpoint_key]) > 1000:
            self.response_times[endpoint_key] = self.response_times[endpoint_key][-1000:]
    
    def record_search(self, metrics: SearchMetrics):
        """Record search-specific metrics."""
        self.search_metrics.append(metrics)
        
        # Update cache stats
        self.cache_stats['total_requests'] += 1
        if metrics.cache_hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
    
    def get_summary_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for the specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter recent requests
        recent_requests = [
            m for m in self.request_metrics 
            if m.timestamp >= cutoff
        ]
        
        recent_searches = [
            m for m in self.search_metrics 
            if m.timestamp >= cutoff
        ]
        
        recent_system = [
            m for m in self.system_metrics 
            if m.timestamp >= cutoff
        ]
        
        # Calculate statistics
        total_requests = len(recent_requests)
        error_requests = len([m for m in recent_requests if m.status_code >= 400])
        
        response_times = [m.response_time for m in recent_requests]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        search_response_times = [m.response_time for m in recent_searches]
        avg_search_time = sum(search_response_times) / len(search_response_times) if search_response_times else 0
        
        # System averages
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system) if recent_system else 0
        avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system) if recent_system else 0
        
        # Cache hit rate
        cache_hit_rate = (
            self.cache_stats['hits'] / self.cache_stats['total_requests'] 
            if self.cache_stats['total_requests'] > 0 else 0
        )
        
        return {
            'period_hours': hours,
            'requests': {
                'total': total_requests,
                'errors': error_requests,
                'error_rate': error_requests / total_requests if total_requests > 0 else 0,
                'avg_response_time': avg_response_time,
                'requests_per_hour': total_requests / hours if hours > 0 else 0
            },
            'search': {
                'total_searches': len(recent_searches),
                'avg_response_time': avg_search_time,
                'cache_hit_rate': cache_hit_rate,
                'searches_per_hour': len(recent_searches) / hours if hours > 0 else 0
            },
            'system': {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'current_active_requests': self.active_requests
            },
            'cache': self.cache_stats.copy()
        }
    
    def get_endpoint_stats(self) -> Dict[str, Any]:
        """Get per-endpoint performance statistics."""
        endpoint_stats = {}
        
        for endpoint, count in self.request_counts.items():
            response_times = self.response_times.get(endpoint, [])
            errors = self.error_counts.get(endpoint, 0)
            
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                p95_time = sorted(response_times)[int(len(response_times) * 0.95)] if len(response_times) > 20 else avg_time
            else:
                avg_time = p95_time = 0
            
            endpoint_stats[endpoint] = {
                'total_requests': count,
                'error_count': errors,
                'error_rate': errors / count if count > 0 else 0,
                'avg_response_time': avg_time,
                'p95_response_time': p95_time
            }
        
        return endpoint_stats
    
    async def cleanup_old_metrics(self, hours: int = 168):  # 7 days
        """Clean up old metrics to prevent memory bloat."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter out old metrics
        self.request_metrics = deque(
            [m for m in self.request_metrics if m.timestamp >= cutoff],
            maxlen=self.max_history
        )
        
        self.search_metrics = deque(
            [m for m in self.search_metrics if m.timestamp >= cutoff],
            maxlen=self.max_history
        )
        
        self.system_metrics = deque(
            [m for m in self.system_metrics if m.timestamp >= cutoff],
            maxlen=self.max_history
        )
        
        logger.info(f"Cleaned up metrics older than {hours} hours")


class MonitoringMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic request monitoring."""
    
    def __init__(self, app, metrics_collector: MetricsCollector):
        super().__init__(app)
        self.metrics = metrics_collector
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        self.metrics.active_requests += 1
        
        try:
            response = await call_next(request)
            
            # Record metrics
            response_time = time.time() - start_time
            
            metrics = RequestMetrics(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time=response_time,
                timestamp=datetime.now(),
                user_agent=request.headers.get('user-agent'),
                ip_address=request.client.host if request.client else None,
                query_params=dict(request.query_params) if request.query_params else None
            )
            
            self.metrics.record_request(metrics)
            
            # Add performance headers
            response.headers['X-Response-Time'] = f"{response_time:.3f}s"
            response.headers['X-Request-ID'] = str(id(request))
            
            return response
            
        except Exception as e:
            # Record error metrics
            response_time = time.time() - start_time
            
            metrics = RequestMetrics(
                endpoint=request.url.path,
                method=request.method,
                status_code=500,
                response_time=response_time,
                timestamp=datetime.now(),
                user_agent=request.headers.get('user-agent'),
                ip_address=request.client.host if request.client else None
            )
            
            self.metrics.record_request(metrics)
            raise
            
        finally:
            self.metrics.active_requests -= 1


@asynccontextmanager
async def track_search_performance(metrics_collector: MetricsCollector, 
                                 query: str, search_type: str):
    """Context manager for tracking search performance."""
    start_time = time.time()
    search_metrics = SearchMetrics(
        query=query,
        search_type=search_type,
        results_count=0,
        response_time=0,
        cache_hit=False
    )
    
    try:
        yield search_metrics
    finally:
        search_metrics.response_time = time.time() - start_time
        metrics_collector.record_search(search_metrics)


# Global metrics collector instance
metrics_collector = MetricsCollector()