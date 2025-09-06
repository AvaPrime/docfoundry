#!/usr/bin/env python3
"""
Performance Optimization Testing Suite

This module provides comprehensive testing for the DocFoundry system optimizations,
including database performance, caching, monitoring, and search functionality.
"""

import asyncio
import time
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import pytest
import httpx
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance test results."""
    test_name: str
    duration: float
    success: bool
    error_message: str = ""
    additional_metrics: Dict[str, Any] = None

class PerformanceTestSuite:
    """Comprehensive performance testing suite for DocFoundry optimizations."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self.results: List[PerformanceMetrics] = []
    
    async def test_database_connection_pooling(self) -> PerformanceMetrics:
        """Test database connection pooling performance."""
        test_name = "Database Connection Pooling"
        start_time = time.time()
        
        try:
            # Simulate concurrent database operations
            tasks = []
            for i in range(20):  # 20 concurrent requests
                task = self.client.get("/health/detailed")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check all responses are successful
            success_count = 0
            for response in responses:
                if isinstance(response, httpx.Response) and response.status_code == 200:
                    success_count += 1
            
            duration = time.time() - start_time
            success = success_count == len(tasks)
            
            return PerformanceMetrics(
                test_name=test_name,
                duration=duration,
                success=success,
                additional_metrics={
                    "concurrent_requests": len(tasks),
                    "successful_responses": success_count,
                    "avg_response_time": duration / len(tasks)
                }
            )
            
        except Exception as e:
            return PerformanceMetrics(
                test_name=test_name,
                duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def test_search_caching_performance(self) -> PerformanceMetrics:
        """Test search result caching performance."""
        test_name = "Search Caching Performance"
        start_time = time.time()
        
        try:
            # Test query
            test_query = "machine learning algorithms"
            search_payload = {
                "q": test_query,
                "k": 10,
                "min_similarity": 0.7
            }
            
            # First request (cache miss)
            first_response = await self.client.post(
                "/search/semantic",
                json=search_payload
            )
            first_duration = time.time() - start_time
            
            if first_response.status_code != 200:
                raise Exception(f"First request failed: {first_response.status_code}")
            
            # Second request (should be cached)
            cache_start = time.time()
            second_response = await self.client.post(
                "/search/semantic",
                json=search_payload
            )
            cache_duration = time.time() - cache_start
            
            if second_response.status_code != 200:
                raise Exception(f"Second request failed: {second_response.status_code}")
            
            # Cache should be significantly faster
            cache_improvement = first_duration / cache_duration if cache_duration > 0 else 0
            
            return PerformanceMetrics(
                test_name=test_name,
                duration=time.time() - start_time,
                success=cache_improvement > 1.5,  # Cache should be at least 1.5x faster
                additional_metrics={
                    "first_request_time": first_duration,
                    "cached_request_time": cache_duration,
                    "cache_improvement_ratio": cache_improvement
                }
            )
            
        except Exception as e:
            return PerformanceMetrics(
                test_name=test_name,
                duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def test_monitoring_endpoints(self) -> PerformanceMetrics:
        """Test monitoring and metrics endpoints."""
        test_name = "Monitoring Endpoints"
        start_time = time.time()
        
        try:
            # Test metrics endpoint
            metrics_response = await self.client.get("/metrics")
            if metrics_response.status_code != 200:
                raise Exception(f"Metrics endpoint failed: {metrics_response.status_code}")
            
            metrics_data = metrics_response.json()
            
            # Test detailed health endpoint
            health_response = await self.client.get("/health/detailed")
            if health_response.status_code != 200:
                raise Exception(f"Health endpoint failed: {health_response.status_code}")
            
            health_data = health_response.json()
            
            # Verify expected metrics are present
            required_metrics = ["request_metrics", "system_metrics", "search_metrics"]
            metrics_present = all(metric in metrics_data for metric in required_metrics)
            
            required_health = ["status", "database", "embedding_manager"]
            health_present = all(field in health_data for field in required_health)
            
            return PerformanceMetrics(
                test_name=test_name,
                duration=time.time() - start_time,
                success=metrics_present and health_present,
                additional_metrics={
                    "metrics_fields": list(metrics_data.keys()),
                    "health_fields": list(health_data.keys())
                }
            )
            
        except Exception as e:
            return PerformanceMetrics(
                test_name=test_name,
                duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def test_search_performance_optimization(self) -> PerformanceMetrics:
        """Test search performance optimizations."""
        test_name = "Search Performance Optimization"
        start_time = time.time()
        
        try:
            # Test different search types
            test_queries = [
                "artificial intelligence",
                "database optimization",
                "machine learning models",
                "web development frameworks",
                "data science techniques"
            ]
            
            search_times = []
            
            for query in test_queries:
                # Test semantic search
                semantic_start = time.time()
                semantic_response = await self.client.post(
                    "/search/semantic",
                    json={
                        "q": query,
                        "k": 10,
                        "min_similarity": 0.7
                    }
                )
                semantic_time = time.time() - semantic_start
                
                if semantic_response.status_code == 200:
                    search_times.append(semantic_time)
                
                # Test hybrid search
                hybrid_start = time.time()
                hybrid_response = await self.client.post(
                    "/search/hybrid",
                    json={
                        "q": query,
                        "k": 10,
                        "rrf_k": 60,
                        "min_similarity": 0.7
                    }
                )
                hybrid_time = time.time() - hybrid_start
                
                if hybrid_response.status_code == 200:
                    search_times.append(hybrid_time)
            
            avg_search_time = sum(search_times) / len(search_times) if search_times else 0
            
            # Performance threshold: average search should be under 2 seconds
            performance_acceptable = avg_search_time < 2.0
            
            return PerformanceMetrics(
                test_name=test_name,
                duration=time.time() - start_time,
                success=performance_acceptable and len(search_times) > 0,
                additional_metrics={
                    "total_searches": len(search_times),
                    "average_search_time": avg_search_time,
                    "max_search_time": max(search_times) if search_times else 0,
                    "min_search_time": min(search_times) if search_times else 0
                }
            )
            
        except Exception as e:
            return PerformanceMetrics(
                test_name=test_name,
                duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def test_concurrent_search_load(self) -> PerformanceMetrics:
        """Test system performance under concurrent search load."""
        test_name = "Concurrent Search Load"
        start_time = time.time()
        
        try:
            # Create concurrent search requests
            concurrent_requests = 15
            tasks = []
            
            for i in range(concurrent_requests):
                query = f"test query {i % 5}"  # Rotate through 5 different queries
                task = self.client.post(
                    "/search/semantic",
                    json={
                        "q": query,
                        "k": 5,
                        "min_similarity": 0.7
                    }
                )
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_responses = 0
            failed_responses = 0
            
            for response in responses:
                if isinstance(response, httpx.Response) and response.status_code == 200:
                    successful_responses += 1
                else:
                    failed_responses += 1
            
            success_rate = successful_responses / len(responses)
            duration = time.time() - start_time
            
            return PerformanceMetrics(
                test_name=test_name,
                duration=duration,
                success=success_rate >= 0.9,  # 90% success rate threshold
                additional_metrics={
                    "concurrent_requests": concurrent_requests,
                    "successful_responses": successful_responses,
                    "failed_responses": failed_responses,
                    "success_rate": success_rate,
                    "requests_per_second": concurrent_requests / duration
                }
            )
            
        except Exception as e:
            return PerformanceMetrics(
                test_name=test_name,
                duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def run_all_tests(self) -> List[PerformanceMetrics]:
        """Run all performance tests and return results."""
        logger.info("Starting comprehensive performance test suite...")
        
        test_methods = [
            self.test_database_connection_pooling,
            self.test_search_caching_performance,
            self.test_monitoring_endpoints,
            self.test_search_performance_optimization,
            self.test_concurrent_search_load
        ]
        
        results = []
        
        for test_method in test_methods:
            logger.info(f"Running {test_method.__name__}...")
            try:
                result = await test_method()
                results.append(result)
                
                status = "PASSED" if result.success else "FAILED"
                logger.info(f"{result.test_name}: {status} ({result.duration:.2f}s)")
                
                if not result.success and result.error_message:
                    logger.error(f"Error: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Test {test_method.__name__} crashed: {e}")
                results.append(PerformanceMetrics(
                    test_name=test_method.__name__,
                    duration=0,
                    success=False,
                    error_message=str(e)
                ))
        
        await self.client.aclose()
        return results
    
    def generate_report(self, results: List[PerformanceMetrics]) -> str:
        """Generate a comprehensive test report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        report = []
        report.append("=" * 80)
        report.append("DOCFOUNDRY PERFORMANCE OPTIMIZATION TEST REPORT")
        report.append("=" * 80)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {failed_tests}")
        report.append(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        report.append("")
        
        for result in results:
            status = "✓ PASSED" if result.success else "✗ FAILED"
            report.append(f"{status} {result.test_name} ({result.duration:.2f}s)")
            
            if result.additional_metrics:
                for key, value in result.additional_metrics.items():
                    report.append(f"  - {key}: {value}")
            
            if not result.success and result.error_message:
                report.append(f"  Error: {result.error_message}")
            
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Main test execution function."""
    test_suite = PerformanceTestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        report = test_suite.generate_report(results)
        
        print(report)
        
        # Save report to file
        report_path = Path("performance_test_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        logger.info(f"Test report saved to {report_path}")
        
        # Return exit code based on test results
        failed_tests = sum(1 for r in results if not r.success)
        return 0 if failed_tests == 0 else 1
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)