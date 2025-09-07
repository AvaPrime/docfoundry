"""DocFoundry Evaluation Framework

This module provides tools for evaluating search quality, relevance metrics,
and system performance across different search modalities.
"""

from .harness import EvaluationHarness
from .metrics import (
    RelevanceMetrics,
    SearchQualityMetrics,
    PerformanceMetrics,
    EmbeddingQualityMetrics
)
from .datasets import (
    EvaluationDataset,
    QueryRelevancePair,
    create_synthetic_dataset,
    load_evaluation_dataset
)
from .benchmarks import (
    SearchBenchmark,
    BenchmarkResult,
    BenchmarkConfig
)

__all__ = [
    'EvaluationHarness',
    'RelevanceMetrics',
    'SearchQualityMetrics', 
    'PerformanceMetrics',
    'EmbeddingQualityMetrics',
    'EvaluationDataset',
    'QueryRelevancePair',
    'create_synthetic_dataset',
    'load_evaluation_dataset',
    'SearchBenchmark',
    'BenchmarkResult',
    'BenchmarkConfig'
]