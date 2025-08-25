"""Search Benchmarks for DocFoundry Performance Testing

This module provides comprehensive benchmarking tools for evaluating search
performance across different modalities and configurations.
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .datasets import EvaluationDataset, QueryRelevancePair
from .metrics import (
    RelevanceMetrics, SearchQualityMetrics, PerformanceMetrics,
    EmbeddingQualityMetrics, ComparisonMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a search operation."""
    
    doc_id: str
    score: float
    title: str = ""
    content: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    
    query: str
    search_type: str
    results: List[SearchResult]
    response_time: float
    relevant_doc_ids: List[str]
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def retrieved_doc_ids(self) -> List[str]:
        """Get list of retrieved document IDs."""
        return [result.doc_id for result in self.results]
    
    @property
    def success(self) -> bool:
        """Check if the search was successful."""
        return self.error is None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    search_types: List[str] = field(default_factory=lambda: ["hybrid", "semantic", "keyword"])
    max_results: int = 10
    timeout: float = 30.0
    concurrent_requests: int = 1
    warmup_queries: int = 5
    repeat_count: int = 1
    include_content: bool = False
    
    # Performance testing
    load_test_duration: int = 60  # seconds
    load_test_qps: float = 1.0    # queries per second
    
    # Quality testing
    relevance_threshold: float = 0.5
    diversity_threshold: float = 0.3
    
    # Comparison testing
    baseline_search_type: str = "hybrid"
    significance_level: float = 0.05


class SearchBenchmark:
    """Comprehensive search benchmarking framework."""
    
    def __init__(self, 
                 search_client: Any,  # SearchClient or API client
                 config: Optional[BenchmarkConfig] = None):
        self.search_client = search_client
        self.config = config or BenchmarkConfig()
        
        # Metrics calculators
        self.relevance_metrics = RelevanceMetrics()
        self.quality_metrics = SearchQualityMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.embedding_metrics = EmbeddingQualityMetrics()
        self.comparison_metrics = ComparisonMetrics()
        
        # Results storage
        self.benchmark_results: List[BenchmarkResult] = []
    
    async def _execute_search(self, 
                            query: str, 
                            search_type: str,
                            timeout: float = 30.0) -> Tuple[List[SearchResult], float, Optional[str]]:
        """Execute a single search operation.
        
        Args:
            query: Search query
            search_type: Type of search (hybrid, semantic, keyword)
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (results, response_time, error)
        """
        start_time = time.time()
        error = None
        results = []
        
        try:
            # This would be replaced with actual search client calls
            if hasattr(self.search_client, 'search'):
                # Async search client
                response = await asyncio.wait_for(
                    self.search_client.search(
                        query=query,
                        search_type=search_type,
                        limit=self.config.max_results,
                        include_content=self.config.include_content
                    ),
                    timeout=timeout
                )
                
                # Convert response to SearchResult objects
                for item in response.get('results', []):
                    result = SearchResult(
                        doc_id=item.get('doc_id', ''),
                        score=item.get('score', 0.0),
                        title=item.get('title', ''),
                        content=item.get('content', ''),
                        metadata=item.get('metadata', {})
                    )
                    results.append(result)
            
            elif hasattr(self.search_client, 'get'):
                # HTTP client (requests/httpx)
                endpoint_map = {
                    'hybrid': '/search/hybrid',
                    'semantic': '/search/semantic', 
                    'keyword': '/search/keyword'
                }
                
                endpoint = endpoint_map.get(search_type, '/search/hybrid')
                
                response = await asyncio.wait_for(
                    self.search_client.get(endpoint, params={
                        'query': query,
                        'limit': self.config.max_results,
                        'include_content': self.config.include_content
                    }),
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('results', []):
                        result = SearchResult(
                            doc_id=item.get('doc_id', ''),
                            score=item.get('score', 0.0),
                            title=item.get('title', ''),
                            content=item.get('content', ''),
                            metadata=item.get('metadata', {})
                        )
                        results.append(result)
                else:
                    error = f"HTTP {response.status_code}: {response.text}"
            
            else:
                # Mock search for testing
                await asyncio.sleep(0.1)  # Simulate network delay
                results = [
                    SearchResult(
                        doc_id=f"doc_{i}",
                        score=1.0 - (i * 0.1),
                        title=f"Result {i+1} for '{query}'",
                        content=f"Content for {query} result {i+1}"
                    )
                    for i in range(min(5, self.config.max_results))
                ]
        
        except asyncio.TimeoutError:
            error = f"Search timeout after {timeout}s"
        except Exception as e:
            error = f"Search error: {str(e)}"
        
        response_time = time.time() - start_time
        return results, response_time, error
    
    async def _warmup(self, dataset: EvaluationDataset) -> None:
        """Perform warmup queries to stabilize performance."""
        if self.config.warmup_queries <= 0:
            return
        
        logger.info(f"Performing {self.config.warmup_queries} warmup queries")
        
        warmup_queries = dataset.pairs[:self.config.warmup_queries]
        for pair in warmup_queries:
            for search_type in self.config.search_types:
                await self._execute_search(pair.query, search_type, timeout=5.0)
        
        logger.info("Warmup completed")
    
    async def run_accuracy_benchmark(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """Run accuracy/relevance benchmark.
        
        Args:
            dataset: Evaluation dataset with query-relevance pairs
            
        Returns:
            Dictionary with accuracy metrics by search type
        """
        logger.info(f"Running accuracy benchmark on {len(dataset)} queries")
        
        await self._warmup(dataset)
        
        results_by_type = {search_type: [] for search_type in self.config.search_types}
        
        # Execute searches
        for pair in dataset.pairs:
            for search_type in self.config.search_types:
                for _ in range(self.config.repeat_count):
                    results, response_time, error = await self._execute_search(
                        pair.query, search_type, self.config.timeout
                    )
                    
                    benchmark_result = BenchmarkResult(
                        query=pair.query,
                        search_type=search_type,
                        results=results,
                        response_time=response_time,
                        relevant_doc_ids=pair.relevant_doc_ids,
                        error=error
                    )
                    
                    results_by_type[search_type].append(benchmark_result)
                    self.benchmark_results.append(benchmark_result)
        
        # Calculate metrics
        accuracy_results = {}
        
        for search_type, results in results_by_type.items():
            successful_results = [r for r in results if r.success]
            
            if not successful_results:
                accuracy_results[search_type] = {
                    'error': 'No successful queries',
                    'success_rate': 0.0
                }
                continue
            
            # Calculate relevance metrics
            relevance_scores = []
            for result in successful_results:
                retrieved_ids = result.retrieved_doc_ids
                relevant_ids = result.relevant_doc_ids
                
                if relevant_ids:  # Only calculate if we have ground truth
                    precision_at_k = self.relevance_metrics.precision_at_k(
                        retrieved_ids, relevant_ids, k=self.config.max_results
                    )
                    recall_at_k = self.relevance_metrics.recall_at_k(
                        retrieved_ids, relevant_ids, k=self.config.max_results
                    )
                    f1_at_k = self.relevance_metrics.f1_at_k(
                        retrieved_ids, relevant_ids, k=self.config.max_results
                    )
                    
                    relevance_scores.append({
                        'precision': precision_at_k,
                        'recall': recall_at_k,
                        'f1': f1_at_k
                    })
            
            # Aggregate metrics
            if relevance_scores:
                avg_precision = statistics.mean([s['precision'] for s in relevance_scores])
                avg_recall = statistics.mean([s['recall'] for s in relevance_scores])
                avg_f1 = statistics.mean([s['f1'] for s in relevance_scores])
            else:
                avg_precision = avg_recall = avg_f1 = 0.0
            
            accuracy_results[search_type] = {
                'success_rate': len(successful_results) / len(results),
                'avg_precision_at_k': avg_precision,
                'avg_recall_at_k': avg_recall,
                'avg_f1_at_k': avg_f1,
                'total_queries': len(results),
                'successful_queries': len(successful_results)
            }
        
        logger.info("Accuracy benchmark completed")
        return accuracy_results
    
    async def run_performance_benchmark(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """Run performance benchmark.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Dictionary with performance metrics by search type
        """
        logger.info(f"Running performance benchmark with {self.config.concurrent_requests} concurrent requests")
        
        await self._warmup(dataset)
        
        performance_results = {}
        
        for search_type in self.config.search_types:
            logger.info(f"Testing {search_type} search performance")
            
            # Collect response times
            response_times = []
            errors = 0
            
            # Run concurrent requests
            if self.config.concurrent_requests > 1:
                # Concurrent execution
                semaphore = asyncio.Semaphore(self.config.concurrent_requests)
                
                async def bounded_search(pair: QueryRelevancePair):
                    async with semaphore:
                        return await self._execute_search(pair.query, search_type, self.config.timeout)
                
                tasks = [bounded_search(pair) for pair in dataset.pairs]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        errors += 1
                    else:
                        _, response_time, error = result
                        if error is None:
                            response_times.append(response_time)
                        else:
                            errors += 1
            
            else:
                # Sequential execution
                for pair in dataset.pairs:
                    _, response_time, error = await self._execute_search(
                        pair.query, search_type, self.config.timeout
                    )
                    
                    if error is None:
                        response_times.append(response_time)
                    else:
                        errors += 1
            
            # Calculate performance metrics
            if response_times:
                performance_results[search_type] = {
                    'avg_response_time': statistics.mean(response_times),
                    'median_response_time': statistics.median(response_times),
                    'p95_response_time': self.performance_metrics.percentile(response_times, 95),
                    'p99_response_time': self.performance_metrics.percentile(response_times, 99),
                    'min_response_time': min(response_times),
                    'max_response_time': max(response_times),
                    'throughput_qps': len(response_times) / sum(response_times) if sum(response_times) > 0 else 0,
                    'error_rate': errors / (len(response_times) + errors),
                    'total_requests': len(response_times) + errors,
                    'successful_requests': len(response_times),
                    'failed_requests': errors
                }
            else:
                performance_results[search_type] = {
                    'error': 'No successful requests',
                    'error_rate': 1.0,
                    'total_requests': errors,
                    'failed_requests': errors
                }
        
        logger.info("Performance benchmark completed")
        return performance_results
    
    async def run_load_test(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """Run load test with sustained query rate.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Dictionary with load test results
        """
        logger.info(f"Running load test for {self.config.load_test_duration}s at {self.config.load_test_qps} QPS")
        
        load_results = {}
        
        for search_type in self.config.search_types:
            logger.info(f"Load testing {search_type} search")
            
            start_time = time.time()
            end_time = start_time + self.config.load_test_duration
            
            response_times = []
            errors = 0
            query_count = 0
            
            # Calculate interval between queries
            query_interval = 1.0 / self.config.load_test_qps
            
            while time.time() < end_time:
                # Select random query
                pair = dataset.pairs[query_count % len(dataset.pairs)]
                query_count += 1
                
                # Execute search
                query_start = time.time()
                _, response_time, error = await self._execute_search(
                    pair.query, search_type, self.config.timeout
                )
                
                if error is None:
                    response_times.append(response_time)
                else:
                    errors += 1
                
                # Wait for next query interval
                elapsed = time.time() - query_start
                sleep_time = max(0, query_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Calculate load test metrics
            actual_duration = time.time() - start_time
            actual_qps = query_count / actual_duration
            
            if response_times:
                load_results[search_type] = {
                    'duration': actual_duration,
                    'target_qps': self.config.load_test_qps,
                    'actual_qps': actual_qps,
                    'total_queries': query_count,
                    'successful_queries': len(response_times),
                    'failed_queries': errors,
                    'error_rate': errors / query_count,
                    'avg_response_time': statistics.mean(response_times),
                    'p95_response_time': self.performance_metrics.percentile(response_times, 95),
                    'p99_response_time': self.performance_metrics.percentile(response_times, 99),
                    'max_response_time': max(response_times)
                }
            else:
                load_results[search_type] = {
                    'error': 'No successful queries during load test',
                    'duration': actual_duration,
                    'total_queries': query_count,
                    'failed_queries': errors,
                    'error_rate': 1.0
                }
        
        logger.info("Load test completed")
        return load_results
    
    async def run_comparison_benchmark(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """Run comparative benchmark between search types.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Running comparison benchmark with baseline: {self.config.baseline_search_type}")
        
        # First run accuracy benchmark to get results
        accuracy_results = await self.run_accuracy_benchmark(dataset)
        
        if self.config.baseline_search_type not in accuracy_results:
            return {'error': f'Baseline search type {self.config.baseline_search_type} not found'}
        
        baseline_results = accuracy_results[self.config.baseline_search_type]
        comparison_results = {'baseline': self.config.baseline_search_type}
        
        for search_type, results in accuracy_results.items():
            if search_type == self.config.baseline_search_type:
                continue
            
            # Compare key metrics
            precision_improvement = self.comparison_metrics.relative_improvement(
                baseline_results.get('avg_precision_at_k', 0),
                results.get('avg_precision_at_k', 0)
            )
            
            recall_improvement = self.comparison_metrics.relative_improvement(
                baseline_results.get('avg_recall_at_k', 0),
                results.get('avg_recall_at_k', 0)
            )
            
            f1_improvement = self.comparison_metrics.relative_improvement(
                baseline_results.get('avg_f1_at_k', 0),
                results.get('avg_f1_at_k', 0)
            )
            
            comparison_results[search_type] = {
                'precision_improvement': precision_improvement,
                'recall_improvement': recall_improvement,
                'f1_improvement': f1_improvement,
                'absolute_precision': results.get('avg_precision_at_k', 0),
                'absolute_recall': results.get('avg_recall_at_k', 0),
                'absolute_f1': results.get('avg_f1_at_k', 0),
                'baseline_precision': baseline_results.get('avg_precision_at_k', 0),
                'baseline_recall': baseline_results.get('avg_recall_at_k', 0),
                'baseline_f1': baseline_results.get('avg_f1_at_k', 0)
            }
        
        logger.info("Comparison benchmark completed")
        return comparison_results
    
    async def run_comprehensive_benchmark(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """Run comprehensive benchmark including all test types.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("Starting comprehensive benchmark suite")
        
        start_time = time.time()
        
        # Run all benchmark types
        results = {
            'config': {
                'search_types': self.config.search_types,
                'max_results': self.config.max_results,
                'concurrent_requests': self.config.concurrent_requests,
                'dataset_size': len(dataset),
                'repeat_count': self.config.repeat_count
            },
            'dataset_stats': dataset.get_statistics()
        }
        
        try:
            results['accuracy'] = await self.run_accuracy_benchmark(dataset)
        except Exception as e:
            logger.error(f"Accuracy benchmark failed: {e}")
            results['accuracy'] = {'error': str(e)}
        
        try:
            results['performance'] = await self.run_performance_benchmark(dataset)
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            results['performance'] = {'error': str(e)}
        
        try:
            results['load_test'] = await self.run_load_test(dataset)
        except Exception as e:
            logger.error(f"Load test failed: {e}")
            results['load_test'] = {'error': str(e)}
        
        try:
            results['comparison'] = await self.run_comparison_benchmark(dataset)
        except Exception as e:
            logger.error(f"Comparison benchmark failed: {e}")
            results['comparison'] = {'error': str(e)}
        
        # Add summary
        total_time = time.time() - start_time
        results['summary'] = {
            'total_benchmark_time': total_time,
            'total_queries_executed': len(self.benchmark_results),
            'benchmark_types_completed': sum(1 for key in ['accuracy', 'performance', 'load_test', 'comparison'] 
                                           if key in results and 'error' not in results[key]),
            'timestamp': time.time()
        }
        
        logger.info(f"Comprehensive benchmark completed in {total_time:.2f}s")
        return results
    
    def get_benchmark_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        return self.benchmark_results.copy()
    
    def clear_results(self) -> None:
        """Clear stored benchmark results."""
        self.benchmark_results.clear()


# Convenience functions

async def quick_benchmark(search_client: Any, 
                         dataset: EvaluationDataset,
                         search_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run a quick benchmark with default settings.
    
    Args:
        search_client: Search client or API client
        dataset: Evaluation dataset
        search_types: List of search types to test
        
    Returns:
        Dictionary with benchmark results
    """
    config = BenchmarkConfig(
        search_types=search_types or ["hybrid", "semantic"],
        max_results=5,
        concurrent_requests=1,
        warmup_queries=2,
        repeat_count=1
    )
    
    benchmark = SearchBenchmark(search_client, config)
    return await benchmark.run_accuracy_benchmark(dataset)


async def performance_test(search_client: Any,
                          dataset: EvaluationDataset,
                          concurrent_requests: int = 5) -> Dict[str, Any]:
    """Run a focused performance test.
    
    Args:
        search_client: Search client or API client
        dataset: Evaluation dataset
        concurrent_requests: Number of concurrent requests
        
    Returns:
        Dictionary with performance results
    """
    config = BenchmarkConfig(
        concurrent_requests=concurrent_requests,
        warmup_queries=5,
        repeat_count=1
    )
    
    benchmark = SearchBenchmark(search_client, config)
    return await benchmark.run_performance_benchmark(dataset)