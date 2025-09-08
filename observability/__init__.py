"""Observability package for DocFoundry."""

from .telemetry import (
    init_telemetry,
    get_tracer,
    get_meter,
    shutdown_telemetry,
    trace_function,
    record_metric,
    record_counter,
    record_histogram,
    record_gauge
)
from .metrics import (
    MetricsCollector,
    SearchMetrics,
    IndexingMetrics,
    APIMetrics
)
from .logging import setup_logging, get_logger
from .prometheus_metrics import (
    setup_prometheus_metrics,
    record_search_metrics,
    record_indexing_metrics,
    record_db_metrics,
    update_system_metrics,
    get_metrics_summary,
    PrometheusMiddleware,
    docfoundry_registry
)

__all__ = [
    'init_telemetry',
    'get_tracer',
    'get_meter', 
    'shutdown_telemetry',
    'trace_function',
    'record_metric',
    'record_counter',
    'record_histogram',
    'record_gauge',
    'MetricsCollector',
    'SearchMetrics',
    'IndexingMetrics',
    'APIMetrics',
    'setup_logging',
    'get_logger',
    'setup_prometheus_metrics',
    'record_search_metrics',
    'record_indexing_metrics',
    'record_db_metrics',
    'update_system_metrics',
    'get_metrics_summary',
    'PrometheusMiddleware',
    'docfoundry_registry'
]