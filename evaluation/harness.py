"""Evaluation Harness for DocFoundry Search Quality Assessment

This module provides the main evaluation framework for testing search quality,
relevance metrics, and performance across different search modalities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict

from .metrics import (
    RelevanceMetrics,
    SearchQualityMetrics,
    PerformanceMetrics,
    EmbeddingQualityMetrics
)
from .datasets import EvaluationDataset, QueryRelevancePair
from .benchmarks import SearchBenchmark

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    
    # Dataset configuration
    dataset_path: Optional[str] = None
    synthetic_dataset_size: int = 100
    
    # Search configuration
    search_endpoints: List[str] = None
    timeout_seconds: float = 30.0
    
    # Metrics configuration
    relevance_threshold: float = 0.7
    top_k_values: List[int] = None
    
    # Output configuration
    output_dir: str = "evaluation_results"
    save_detailed_results: bool = True
    
    # Performance configuration
    concurrent_requests: int = 5
    warmup_queries: int = 10
    
    def __post_init__(self):
        if self.search_endpoints is None:
            self.search_endpoints = [
                "http://localhost:8001/search",
                "http://localhost:8001/search/semantic", 
                "http://localhost:8001/search/hybrid"
            ]
        
        if self.top_k_values is None:
            self.top_k_values = [1, 3, 5, 10, 20]


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""
    
    timestamp: str
    config: EvaluationConfig
    dataset_info: Dict[str, Any]
    search_results: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Dict[str, float]]
    performance_stats: Dict[str, Dict[str, float]]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_path}")


class EvaluationHarness:
    """Main evaluation harness for DocFoundry search quality assessment."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.relevance_metrics = RelevanceMetrics()
        self.quality_metrics = SearchQualityMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.embedding_metrics = EmbeddingQualityMetrics()
        
        # Initialize benchmarks for different search types
        self.benchmarks = {
            'fulltext': None,  # Will be initialized when needed
            'semantic': None,
            'hybrid': None
        }
        
        logger.info(f"Evaluation harness initialized with config: {self.config}")
    
    async def run_evaluation(
        self,
        dataset: Optional[EvaluationDataset] = None,
        benchmarks: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Run complete evaluation suite.
        
        Args:
            dataset: Evaluation dataset to use. If None, will load or create one.
            benchmarks: List of benchmark types to run. If None, runs all.
            
        Returns:
            EvaluationResult containing all metrics and analysis.
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        logger.info("Starting evaluation run")
        
        # Load or create dataset
        if dataset is None:
            dataset = await self._load_or_create_dataset()
        
        # Determine which benchmarks to run
        if benchmarks is None:
            benchmarks = ['fulltext', 'semantic', 'hybrid']
        
        # Initialize results structure
        search_results = {}
        metrics = {}
        performance_stats = {}
        
        # Run benchmarks
        for benchmark_type in benchmarks:
            logger.info(f"Running {benchmark_type} benchmark")
            
            try:
                benchmark = await self._get_benchmark(benchmark_type)
                
                # Run search queries
                results = await self._run_benchmark_queries(benchmark, dataset)
                search_results[benchmark_type] = results
                
                # Calculate metrics
                benchmark_metrics = await self._calculate_metrics(
                    results, dataset, benchmark_type
                )
                metrics[benchmark_type] = benchmark_metrics
                
                # Calculate performance stats
                perf_stats = await self._calculate_performance_stats(
                    results, benchmark_type
                )
                performance_stats[benchmark_type] = perf_stats
                
            except Exception as e:
                logger.error(f"Error running {benchmark_type} benchmark: {e}")
                search_results[benchmark_type] = {'error': str(e)}
                metrics[benchmark_type] = {}
                performance_stats[benchmark_type] = {}
        
        # Generate summary
        summary = self._generate_summary(
            metrics, performance_stats, time.time() - start_time
        )
        
        # Create evaluation result
        result = EvaluationResult(
            timestamp=timestamp,
            config=self.config,
            dataset_info={
                'size': len(dataset.pairs),
                'unique_queries': len(set(p.query for p in dataset.pairs)),
                'avg_relevant_docs': sum(len(p.relevant_doc_ids) for p in dataset.pairs) / len(dataset.pairs)
            },
            search_results=search_results,
            metrics=metrics,
            performance_stats=performance_stats,
            summary=summary
        )
        
        # Save results if configured
        if self.config.save_detailed_results:
            output_path = Path(self.config.output_dir) / f"evaluation_{timestamp.replace(':', '-')}.json"
            result.save(output_path)
        
        logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        return result
    
    async def _load_or_create_dataset(self) -> EvaluationDataset:
        """Load existing dataset or create synthetic one."""
        if self.config.dataset_path and Path(self.config.dataset_path).exists():
            from .datasets import load_evaluation_dataset
            return await load_evaluation_dataset(self.config.dataset_path)
        else:
            from .datasets import create_synthetic_dataset
            return await create_synthetic_dataset(self.config.synthetic_dataset_size)
    
    async def _get_benchmark(self, benchmark_type: str) -> SearchBenchmark:
        """Get or create benchmark instance."""
        if self.benchmarks[benchmark_type] is None:
            if benchmark_type == 'fulltext':
                from .benchmarks import FullTextSearchBenchmark
                self.benchmarks[benchmark_type] = FullTextSearchBenchmark(
                    endpoint=self.config.search_endpoints[0]
                )
            elif benchmark_type == 'semantic':
                from .benchmarks import SemanticSearchBenchmark
                self.benchmarks[benchmark_type] = SemanticSearchBenchmark(
                    endpoint=self.config.search_endpoints[1]
                )
            elif benchmark_type == 'hybrid':
                from .benchmarks import HybridSearchBenchmark
                self.benchmarks[benchmark_type] = HybridSearchBenchmark(
                    endpoint=self.config.search_endpoints[2]
                )
            else:
                raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        return self.benchmarks[benchmark_type]
    
    async def _run_benchmark_queries(
        self, 
        benchmark: SearchBenchmark, 
        dataset: EvaluationDataset
    ) -> Dict[str, Any]:
        """Run all queries for a benchmark."""
        results = {
            'queries': [],
            'total_time': 0,
            'successful_queries': 0,
            'failed_queries': 0
        }
        
        # Warmup queries
        if self.config.warmup_queries > 0:
            warmup_queries = dataset.pairs[:self.config.warmup_queries]
            for pair in warmup_queries:
                try:
                    await benchmark.search(pair.query, limit=10)
                except Exception:
                    pass  # Ignore warmup failures
        
        # Run actual evaluation queries
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        
        async def run_single_query(pair: QueryRelevancePair) -> Dict[str, Any]:
            async with semaphore:
                start_time = time.time()
                try:
                    search_results = await benchmark.search(
                        pair.query, 
                        limit=max(self.config.top_k_values)
                    )
                    
                    query_time = time.time() - start_time
                    
                    return {
                        'query': pair.query,
                        'results': search_results,
                        'relevant_doc_ids': pair.relevant_doc_ids,
                        'query_time': query_time,
                        'success': True,
                        'error': None
                    }
                    
                except Exception as e:
                    query_time = time.time() - start_time
                    return {
                        'query': pair.query,
                        'results': [],
                        'relevant_doc_ids': pair.relevant_doc_ids,
                        'query_time': query_time,
                        'success': False,
                        'error': str(e)
                    }
        
        # Execute all queries
        tasks = [run_single_query(pair) for pair in dataset.pairs]
        query_results = await asyncio.gather(*tasks)
        
        # Aggregate results
        results['queries'] = query_results
        results['total_time'] = sum(r['query_time'] for r in query_results)
        results['successful_queries'] = sum(1 for r in query_results if r['success'])
        results['failed_queries'] = sum(1 for r in query_results if not r['success'])
        
        return results
    
    async def _calculate_metrics(
        self, 
        results: Dict[str, Any], 
        dataset: EvaluationDataset,
        benchmark_type: str
    ) -> Dict[str, float]:
        """Calculate all metrics for benchmark results."""
        metrics = {}
        
        # Relevance metrics
        for k in self.config.top_k_values:
            precision_k = self.relevance_metrics.precision_at_k(
                results['queries'], k
            )
            recall_k = self.relevance_metrics.recall_at_k(
                results['queries'], k
            )
            f1_k = self.relevance_metrics.f1_at_k(
                results['queries'], k
            )
            
            metrics[f'precision@{k}'] = precision_k
            metrics[f'recall@{k}'] = recall_k
            metrics[f'f1@{k}'] = f1_k
        
        # Mean Average Precision
        metrics['map'] = self.relevance_metrics.mean_average_precision(
            results['queries']
        )
        
        # Normalized Discounted Cumulative Gain
        for k in self.config.top_k_values:
            ndcg_k = self.relevance_metrics.ndcg_at_k(
                results['queries'], k
            )
            metrics[f'ndcg@{k}'] = ndcg_k
        
        # Search quality metrics
        metrics['query_success_rate'] = self.quality_metrics.query_success_rate(
            results['queries']
        )
        
        metrics['result_diversity'] = self.quality_metrics.result_diversity(
            results['queries']
        )
        
        # Embedding quality metrics (for semantic/hybrid searches)
        if benchmark_type in ['semantic', 'hybrid']:
            metrics['embedding_coherence'] = await self.embedding_metrics.coherence_score(
                results['queries']
            )
        
        return metrics
    
    async def _calculate_performance_stats(
        self, 
        results: Dict[str, Any],
        benchmark_type: str
    ) -> Dict[str, float]:
        """Calculate performance statistics."""
        query_times = [r['query_time'] for r in results['queries'] if r['success']]
        
        if not query_times:
            return {}
        
        stats = {
            'avg_query_time': sum(query_times) / len(query_times),
            'min_query_time': min(query_times),
            'max_query_time': max(query_times),
            'median_query_time': sorted(query_times)[len(query_times) // 2],
            'p95_query_time': sorted(query_times)[int(len(query_times) * 0.95)],
            'p99_query_time': sorted(query_times)[int(len(query_times) * 0.99)],
            'total_queries': len(results['queries']),
            'successful_queries': results['successful_queries'],
            'failed_queries': results['failed_queries'],
            'success_rate': results['successful_queries'] / len(results['queries'])
        }
        
        # Calculate throughput
        if results['total_time'] > 0:
            stats['queries_per_second'] = results['successful_queries'] / results['total_time']
        
        return stats
    
    def _generate_summary(
        self, 
        metrics: Dict[str, Dict[str, float]],
        performance_stats: Dict[str, Dict[str, float]],
        total_time: float
    ) -> Dict[str, Any]:
        """Generate evaluation summary."""
        summary = {
            'total_evaluation_time': total_time,
            'benchmarks_run': list(metrics.keys()),
            'best_performing': {},
            'recommendations': []
        }
        
        # Find best performing benchmark for key metrics
        key_metrics = ['precision@5', 'recall@5', 'f1@5', 'map', 'ndcg@5']
        
        for metric in key_metrics:
            best_benchmark = None
            best_score = -1
            
            for benchmark, benchmark_metrics in metrics.items():
                if metric in benchmark_metrics and benchmark_metrics[metric] > best_score:
                    best_score = benchmark_metrics[metric]
                    best_benchmark = benchmark
            
            if best_benchmark:
                summary['best_performing'][metric] = {
                    'benchmark': best_benchmark,
                    'score': best_score
                }
        
        # Generate recommendations
        if 'hybrid' in metrics and 'semantic' in metrics and 'fulltext' in metrics:
            hybrid_f1 = metrics['hybrid'].get('f1@5', 0)
            semantic_f1 = metrics['semantic'].get('f1@5', 0)
            fulltext_f1 = metrics['fulltext'].get('f1@5', 0)
            
            if hybrid_f1 > max(semantic_f1, fulltext_f1):
                summary['recommendations'].append(
                    "Hybrid search shows best F1@5 performance - recommended for production"
                )
            elif semantic_f1 > fulltext_f1:
                summary['recommendations'].append(
                    "Semantic search outperforms full-text - consider as primary search method"
                )
            else:
                summary['recommendations'].append(
                    "Full-text search performs best - semantic features may need tuning"
                )
        
        # Performance recommendations
        for benchmark, stats in performance_stats.items():
            if stats.get('avg_query_time', 0) > 1.0:
                summary['recommendations'].append(
                    f"{benchmark} search has high latency ({stats['avg_query_time']:.2f}s) - consider optimization"
                )
            
            if stats.get('success_rate', 1.0) < 0.95:
                summary['recommendations'].append(
                    f"{benchmark} search has low success rate ({stats['success_rate']:.1%}) - investigate failures"
                )
        
        return summary
    
    async def run_quick_evaluation(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Run a quick evaluation on a single query across all search types."""
        results = {}
        
        for benchmark_type in ['fulltext', 'semantic', 'hybrid']:
            try:
                benchmark = await self._get_benchmark(benchmark_type)
                start_time = time.time()
                search_results = await benchmark.search(query, limit=limit)
                query_time = time.time() - start_time
                
                results[benchmark_type] = {
                    'results': search_results,
                    'query_time': query_time,
                    'result_count': len(search_results)
                }
                
            except Exception as e:
                results[benchmark_type] = {
                    'error': str(e),
                    'query_time': 0,
                    'result_count': 0
                }
        
        return results