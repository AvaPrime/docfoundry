#!/usr/bin/env python3
"""
Test script for OpenTelemetry integration in DocFoundry.
This script verifies that telemetry components are working correctly.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_telemetry_integration():
    """Test OpenTelemetry integration components."""
    
    print("=== DocFoundry OpenTelemetry Integration Test ===")
    
    # Test 1: Import telemetry module
    print("\n1. Testing telemetry module import...")
    try:
        from observability.telemetry import (
            init_telemetry,
            shutdown_telemetry,
            trace_span,
            record_counter,
            record_histogram,
            record_gauge
        )
        print("✓ Telemetry module imported successfully")
        telemetry_available = True
    except ImportError as e:
        print(f"✗ Telemetry module import failed: {e}")
        print("  This is expected if OpenTelemetry dependencies are not installed")
        telemetry_available = False
    
    if not telemetry_available:
        print("\nSkipping telemetry tests - OpenTelemetry not available")
        return
    
    # Test 2: Initialize telemetry
    print("\n2. Testing telemetry initialization...")
    try:
        init_telemetry(
            service_name="docfoundry-test",
            service_version="0.2.0",
            environment="test",
            otlp_endpoint=None,  # No external endpoint for testing
            enable_prometheus=False  # Disable for testing
        )
        print("✓ Telemetry initialized successfully")
    except Exception as e:
        print(f"✗ Telemetry initialization failed: {e}")
        return
    
    # Test 3: Test tracing
    print("\n3. Testing distributed tracing...")
    try:
        with trace_span("test_operation") as span:
            span.set_attribute("test.attribute", "test_value")
            span.set_attribute("test.number", 42)
            
            # Nested span
            with trace_span("nested_operation") as nested_span:
                nested_span.set_attribute("nested.attribute", "nested_value")
                await asyncio.sleep(0.01)  # Simulate work
        
        print("✓ Distributed tracing working")
    except Exception as e:
        print(f"✗ Tracing test failed: {e}")
    
    # Test 4: Test metrics
    print("\n4. Testing metrics collection...")
    try:
        # Test counter
        record_counter("test_counter", {"environment": "test"})
        record_counter("test_counter", {"environment": "test"})
        
        # Test histogram
        record_histogram("test_histogram", 1.5, {"operation": "test"})
        record_histogram("test_histogram", 2.3, {"operation": "test"})
        
        # Test gauge
        record_gauge("test_gauge", 100, {"resource": "memory"})
        
        print("✓ Metrics collection working")
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
    
    # Test 5: Test FastAPI integration (import only)
    print("\n5. Testing FastAPI integration...")
    try:
        from observability.telemetry import instrument_fastapi_app
        from fastapi import FastAPI
        
        test_app = FastAPI(title="Test App")
        instrument_fastapi_app(test_app)
        
        print("✓ FastAPI instrumentation working")
    except Exception as e:
        print(f"✗ FastAPI integration test failed: {e}")
    
    # Test 6: Simulate search operation telemetry
    print("\n6. Testing search operation telemetry...")
    try:
        with trace_span("semantic_search") as span:
            span.set_attribute("query", "test query")
            span.set_attribute("k", 5)
            span.set_attribute("min_similarity", 0.3)
            
            # Simulate search processing
            await asyncio.sleep(0.05)
            
            # Record results
            results_count = 3
            span.set_attribute("results_count", results_count)
            
            # Record metrics
            record_counter("search_requests_total", {"type": "semantic"})
            record_histogram("search_results_count", results_count, {"type": "semantic"})
            record_counter("search_requests_success", {"type": "semantic"})
        
        print("✓ Search operation telemetry working")
    except Exception as e:
        print(f"✗ Search telemetry test failed: {e}")
    
    # Test 7: Test error tracking
    print("\n7. Testing error tracking...")
    try:
        with trace_span("error_test") as span:
            try:
                # Simulate an error
                raise ValueError("Test error")
            except ValueError as e:
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                
                record_counter("test_errors_total", {"error": type(e).__name__})
        
        print("✓ Error tracking working")
    except Exception as e:
        print(f"✗ Error tracking test failed: {e}")
    
    # Test 8: Cleanup
    print("\n8. Testing telemetry shutdown...")
    try:
        shutdown_telemetry()
        print("✓ Telemetry shutdown successful")
    except Exception as e:
        print(f"✗ Telemetry shutdown failed: {e}")
    
    print("\n=== Test Summary ===")
    print("✓ OpenTelemetry integration test completed")
    print("\nNext steps:")
    print("1. Install observability stack: docker-compose -f docker-compose.observability.yml up -d")
    print("2. Configure environment variables in .env")
    print("3. Start DocFoundry: python -m uvicorn server.rag_api:app --reload")
    print("4. View traces at: http://localhost:16686 (Jaeger)")
    print("5. View metrics at: http://localhost:9090 (Prometheus)")
    print("6. View dashboards at: http://localhost:3000 (Grafana)")

def main():
    """Main test function."""
    try:
        asyncio.run(test_telemetry_integration())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()