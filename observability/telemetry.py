from __future__ import annotations
from typing import Optional, Dict, Any, Callable
from functools import wraps
import os
import time
from contextlib import contextmanager

try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Fallback classes for when OpenTelemetry is not available
    class MockTracer:
        def start_span(self, name: str, **kwargs):
            return MockSpan()
    
    class MockSpan:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_attribute(self, key: str, value: Any):
            pass
        def set_status(self, status):
            pass
        def record_exception(self, exception):
            pass
    
    class MockMeter:
        def create_counter(self, name: str, **kwargs):
            return MockInstrument()
        def create_histogram(self, name: str, **kwargs):
            return MockInstrument()
        def create_gauge(self, name: str, **kwargs):
            return MockInstrument()
    
    class MockInstrument:
        def add(self, amount: float, attributes: Dict[str, Any] = None):
            pass
        def record(self, amount: float, attributes: Dict[str, Any] = None):
            pass
        def set(self, amount: float, attributes: Dict[str, Any] = None):
            pass

class TelemetryConfig:
    """Configuration for telemetry setup."""
    
    def __init__(
        self,
        service_name: str = "docfoundry",
        service_version: str = "1.0.0",
        environment: str = "development",
        otlp_endpoint: Optional[str] = None,
        prometheus_port: int = 8000,
        enable_tracing: bool = True,
        enable_metrics: bool = True,
        enable_prometheus: bool = True,
        sample_rate: float = 1.0
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self.prometheus_port = prometheus_port
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        self.enable_prometheus = enable_prometheus
        self.sample_rate = sample_rate

class TelemetryManager:
    """Manages OpenTelemetry setup and instrumentation."""
    
    def __init__(self):
        self.config: Optional[TelemetryConfig] = None
        self.tracer: Optional[Any] = None
        self.meter: Optional[Any] = None
        self.initialized = False
    
    def init(self, config: TelemetryConfig) -> None:
        """Initialize telemetry with the given configuration."""
        if not OTEL_AVAILABLE:
            print("Warning: OpenTelemetry not available. Using mock implementations.")
            self.tracer = MockTracer()
            self.meter = MockMeter()
            self.config = config
            self.initialized = True
            return
        
        self.config = config
        
        # Create resource
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: config.service_name,
            ResourceAttributes.SERVICE_VERSION: config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: config.environment,
        })
        
        # Setup tracing
        if config.enable_tracing:
            self._setup_tracing(resource)
        
        # Setup metrics
        if config.enable_metrics:
            self._setup_metrics(resource)
        
        # Setup auto-instrumentation
        self._setup_auto_instrumentation()
        
        self.initialized = True
    
    def _setup_tracing(self, resource: Resource) -> None:
        """Setup distributed tracing."""
        tracer_provider = TracerProvider(resource=resource)
        
        # Add OTLP exporter if endpoint is configured
        if self.config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
        
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(self.config.service_name)
    
    def _setup_metrics(self, resource: Resource) -> None:
        """Setup metrics collection."""
        readers = []
        
        # Add Prometheus reader if enabled
        if self.config.enable_prometheus:
            prometheus_reader = PrometheusMetricReader()
            readers.append(prometheus_reader)
        
        # Add OTLP reader if endpoint is configured
        if self.config.otlp_endpoint:
            otlp_exporter = OTLPMetricExporter(endpoint=self.config.otlp_endpoint)
            otlp_reader = PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=5000)
            readers.append(otlp_reader)
        
        if readers:
            meter_provider = MeterProvider(resource=resource, metric_readers=readers)
            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(self.config.service_name)
    
    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for common libraries."""
        try:
            # Instrument HTTP requests
            RequestsInstrumentor().instrument()
            
            # Instrument SQLite
            SQLite3Instrumentor().instrument()
            
            # FastAPI instrumentation will be done when the app is created
        except Exception as e:
            print(f"Warning: Failed to setup auto-instrumentation: {e}")
    
    def get_tracer(self):
        """Get the configured tracer."""
        if not self.initialized:
            return MockTracer()
        return self.tracer or MockTracer()
    
    def get_meter(self):
        """Get the configured meter."""
        if not self.initialized:
            return MockMeter()
        return self.meter or MockMeter()
    
    def shutdown(self) -> None:
        """Shutdown telemetry providers."""
        if OTEL_AVAILABLE and self.initialized:
            try:
                if hasattr(trace, 'get_tracer_provider'):
                    tracer_provider = trace.get_tracer_provider()
                    if hasattr(tracer_provider, 'shutdown'):
                        tracer_provider.shutdown()
                
                if hasattr(metrics, 'get_meter_provider'):
                    meter_provider = metrics.get_meter_provider()
                    if hasattr(meter_provider, 'shutdown'):
                        meter_provider.shutdown()
            except Exception as e:
                print(f"Warning: Error during telemetry shutdown: {e}")
        
        self.initialized = False

# Global telemetry manager instance
_telemetry_manager = TelemetryManager()

# Public API functions
def init_telemetry(
    service_name: str = "docfoundry",
    service_version: str = "1.0.0",
    environment: str = None,
    **kwargs
) -> None:
    """Initialize telemetry with configuration."""
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    config = TelemetryConfig(
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        **kwargs
    )
    _telemetry_manager.init(config)

def get_tracer():
    """Get the global tracer instance."""
    return _telemetry_manager.get_tracer()

def get_meter():
    """Get the global meter instance."""
    return _telemetry_manager.get_meter()

def shutdown_telemetry() -> None:
    """Shutdown telemetry."""
    _telemetry_manager.shutdown()

def trace_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace function execution."""
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator

@contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for creating traced spans."""
    tracer = get_tracer()
    with tracer.start_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span

def record_metric(metric_name: str, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Record a generic metric value."""
    # This is a simplified interface - in practice you'd want to create instruments once
    pass

def record_counter(name: str, value: float = 1.0, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Record a counter metric."""
    meter = get_meter()
    counter = meter.create_counter(name)
    counter.add(value, attributes or {})

def record_histogram(name: str, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Record a histogram metric."""
    meter = get_meter()
    histogram = meter.create_histogram(name)
    histogram.record(value, attributes or {})

def record_gauge(name: str, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Record a gauge metric."""
    meter = get_meter()
    gauge = meter.create_gauge(name)
    gauge.set(value, attributes or {})

# Utility functions for common patterns
def time_function(func_name: Optional[str] = None):
    """Decorator to time function execution and record as histogram."""
    def decorator(func: Callable) -> Callable:
        metric_name = func_name or f"{func.__module__}.{func.__name__}.duration"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                record_histogram(metric_name, duration, {"status": "success"})
                return result
            except Exception as e:
                duration = time.time() - start_time
                record_histogram(metric_name, duration, {"status": "error", "error_type": type(e).__name__})
                raise
        
        return wrapper
    return decorator

def instrument_fastapi_app(app):
    """Instrument a FastAPI application."""
    if OTEL_AVAILABLE:
        try:
            FastAPIInstrumentor.instrument_app(app)
        except Exception as e:
            print(f"Warning: Failed to instrument FastAPI app: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize telemetry
    init_telemetry(
        service_name="docfoundry-example",
        environment="development",
        enable_prometheus=True
    )
    
    # Example traced function
    @trace_function("example.process_document")
    @time_function("example.process_document.duration")
    def process_document(doc_id: str) -> str:
        # Simulate processing
        time.sleep(0.1)
        record_counter("documents.processed", 1.0, {"doc_type": "markdown"})
        return f"Processed {doc_id}"
    
    # Example usage
    with trace_span("example.batch_process") as span:
        span.set_attribute("batch_size", 3)
        
        for i in range(3):
            result = process_document(f"doc_{i}")
            print(result)
    
    print("Telemetry example completed")
    shutdown_telemetry()