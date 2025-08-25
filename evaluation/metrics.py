"""Evaluation Metrics for DocFoundry Search Quality Assessment

This module provides various metrics for evaluating search quality,
including relevance, performance, and embedding quality metrics.
"""

import math
import statistics
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict
import numpy as np


class RelevanceMetrics:
    """Metrics for evaluating search result relevance."""
    
    def precision_at_k(self, query_results: List[Dict[str, Any]], k: int) -> float:
        """Calculate Precision@K across all queries.
        
        Args:
            query_results: List of query result dictionaries
            k: Number of top results to consider
            
        Returns:
            Average precision@k across all queries
        """
        if not query_results:
            return 0.0
        
        precisions = []
        for result in query_results:
            if not result.get('success', False):
                continue
                
            retrieved_docs = [r.get('doc_id') for r in result.get('results', [])[:k]]
            relevant_docs = set(result.get('relevant_doc_ids', []))
            
            if not retrieved_docs:
                precisions.append(0.0)
                continue
            
            relevant_retrieved = sum(1 for doc_id in retrieved_docs if doc_id in relevant_docs)
            precision = relevant_retrieved / len(retrieved_docs)
            precisions.append(precision)
        
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    def recall_at_k(self, query_results: List[Dict[str, Any]], k: int) -> float:
        """Calculate Recall@K across all queries.
        
        Args:
            query_results: List of query result dictionaries
            k: Number of top results to consider
            
        Returns:
            Average recall@k across all queries
        """
        if not query_results:
            return 0.0
        
        recalls = []
        for result in query_results:
            if not result.get('success', False):
                continue
                
            retrieved_docs = [r.get('doc_id') for r in result.get('results', [])[:k]]
            relevant_docs = set(result.get('relevant_doc_ids', []))
            
            if not relevant_docs:
                recalls.append(1.0)  # Perfect recall if no relevant docs
                continue
            
            relevant_retrieved = sum(1 for doc_id in retrieved_docs if doc_id in relevant_docs)
            recall = relevant_retrieved / len(relevant_docs)
            recalls.append(recall)
        
        return sum(recalls) / len(recalls) if recalls else 0.0
    
    def f1_at_k(self, query_results: List[Dict[str, Any]], k: int) -> float:
        """Calculate F1@K across all queries.
        
        Args:
            query_results: List of query result dictionaries
            k: Number of top results to consider
            
        Returns:
            Average F1@k across all queries
        """
        precision = self.precision_at_k(query_results, k)
        recall = self.recall_at_k(query_results, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """Calculate Average Precision for a single query.
        
        Args:
            retrieved_docs: List of retrieved document IDs in rank order
            relevant_docs: Set of relevant document IDs
            
        Returns:
            Average precision score
        """
        if not relevant_docs:
            return 1.0
        
        if not retrieved_docs:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs)
    
    def mean_average_precision(self, query_results: List[Dict[str, Any]]) -> float:
        """Calculate Mean Average Precision (MAP) across all queries.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Mean average precision score
        """
        if not query_results:
            return 0.0
        
        average_precisions = []
        for result in query_results:
            if not result.get('success', False):
                continue
                
            retrieved_docs = [r.get('doc_id') for r in result.get('results', [])]
            relevant_docs = set(result.get('relevant_doc_ids', []))
            
            ap = self.average_precision(retrieved_docs, relevant_docs)
            average_precisions.append(ap)
        
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    
    def dcg_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """Calculate Discounted Cumulative Gain at K.
        
        Args:
            retrieved_docs: List of retrieved document IDs in rank order
            relevant_docs: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            DCG@k score
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            if doc_id in relevant_docs:
                # Binary relevance: 1 if relevant, 0 if not
                relevance = 1.0
                dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        return dcg
    
    def ndcg_at_k(self, query_results: List[Dict[str, Any]], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K.
        
        Args:
            query_results: List of query result dictionaries
            k: Number of top results to consider
            
        Returns:
            Average NDCG@k across all queries
        """
        if not query_results:
            return 0.0
        
        ndcg_scores = []
        for result in query_results:
            if not result.get('success', False):
                continue
                
            retrieved_docs = [r.get('doc_id') for r in result.get('results', [])]
            relevant_docs = set(result.get('relevant_doc_ids', []))
            
            if not relevant_docs:
                ndcg_scores.append(1.0)  # Perfect NDCG if no relevant docs
                continue
            
            # Calculate DCG
            dcg = self.dcg_at_k(retrieved_docs, relevant_docs, k)
            
            # Calculate IDCG (Ideal DCG)
            ideal_docs = list(relevant_docs) + [''] * k  # Pad with empty docs
            idcg = self.dcg_at_k(ideal_docs, relevant_docs, k)
            
            if idcg == 0:
                ndcg_scores.append(0.0)
            else:
                ndcg_scores.append(dcg / idcg)
        
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


class SearchQualityMetrics:
    """Metrics for evaluating overall search quality."""
    
    def query_success_rate(self, query_results: List[Dict[str, Any]]) -> float:
        """Calculate the rate of successful queries.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Success rate as a float between 0 and 1
        """
        if not query_results:
            return 0.0
        
        successful = sum(1 for result in query_results if result.get('success', False))
        return successful / len(query_results)
    
    def result_diversity(self, query_results: List[Dict[str, Any]]) -> float:
        """Calculate diversity of search results across queries.
        
        Measures how diverse the returned documents are across all queries.
        Higher diversity indicates less repetition in results.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Diversity score as a float between 0 and 1
        """
        if not query_results:
            return 0.0
        
        all_returned_docs = set()
        total_results = 0
        
        for result in query_results:
            if not result.get('success', False):
                continue
                
            for doc_result in result.get('results', []):
                doc_id = doc_result.get('doc_id')
                if doc_id:
                    all_returned_docs.add(doc_id)
                    total_results += 1
        
        if total_results == 0:
            return 0.0
        
        return len(all_returned_docs) / total_results
    
    def coverage_ratio(self, query_results: List[Dict[str, Any]], total_docs: int) -> float:
        """Calculate what fraction of the document corpus is covered by search results.
        
        Args:
            query_results: List of query result dictionaries
            total_docs: Total number of documents in the corpus
            
        Returns:
            Coverage ratio as a float between 0 and 1
        """
        if total_docs == 0:
            return 0.0
        
        covered_docs = set()
        for result in query_results:
            if not result.get('success', False):
                continue
                
            for doc_result in result.get('results', []):
                doc_id = doc_result.get('doc_id')
                if doc_id:
                    covered_docs.add(doc_id)
        
        return len(covered_docs) / total_docs
    
    def result_consistency(self, query_results: List[Dict[str, Any]]) -> float:
        """Measure consistency of results for similar queries.
        
        This is a simplified metric that could be enhanced with
        query similarity analysis.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Consistency score as a float between 0 and 1
        """
        # Group queries by similarity (simplified: exact match)
        query_groups = defaultdict(list)
        for result in query_results:
            if result.get('success', False):
                query = result.get('query', '').lower().strip()
                query_groups[query].append(result)
        
        if len(query_groups) <= 1:
            return 1.0  # Perfect consistency if no duplicate queries
        
        consistency_scores = []
        for query, results in query_groups.items():
            if len(results) < 2:
                continue
            
            # Calculate overlap between result sets
            result_sets = []
            for result in results:
                doc_ids = {r.get('doc_id') for r in result.get('results', [])}
                result_sets.append(doc_ids)
            
            # Calculate pairwise Jaccard similarity
            similarities = []
            for i in range(len(result_sets)):
                for j in range(i + 1, len(result_sets)):
                    set1, set2 = result_sets[i], result_sets[j]
                    if not set1 and not set2:
                        similarities.append(1.0)
                    elif not set1 or not set2:
                        similarities.append(0.0)
                    else:
                        jaccard = len(set1 & set2) / len(set1 | set2)
                        similarities.append(jaccard)
            
            if similarities:
                consistency_scores.append(sum(similarities) / len(similarities))
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0


class PerformanceMetrics:
    """Metrics for evaluating search performance."""
    
    def average_response_time(self, query_results: List[Dict[str, Any]]) -> float:
        """Calculate average response time across successful queries.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Average response time in seconds
        """
        times = [r['query_time'] for r in query_results if r.get('success', False)]
        return sum(times) / len(times) if times else 0.0
    
    def percentile_response_time(self, query_results: List[Dict[str, Any]], percentile: float) -> float:
        """Calculate percentile response time.
        
        Args:
            query_results: List of query result dictionaries
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile response time in seconds
        """
        times = [r['query_time'] for r in query_results if r.get('success', False)]
        if not times:
            return 0.0
        
        return np.percentile(times, percentile)
    
    def throughput(self, query_results: List[Dict[str, Any]], total_time: float) -> float:
        """Calculate query throughput.
        
        Args:
            query_results: List of query result dictionaries
            total_time: Total time for all queries in seconds
            
        Returns:
            Queries per second
        """
        if total_time <= 0:
            return 0.0
        
        successful_queries = sum(1 for r in query_results if r.get('success', False))
        return successful_queries / total_time
    
    def error_rate(self, query_results: List[Dict[str, Any]]) -> float:
        """Calculate error rate.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Error rate as a float between 0 and 1
        """
        if not query_results:
            return 0.0
        
        failed_queries = sum(1 for r in query_results if not r.get('success', False))
        return failed_queries / len(query_results)


class EmbeddingQualityMetrics:
    """Metrics for evaluating embedding and semantic search quality."""
    
    async def coherence_score(self, query_results: List[Dict[str, Any]]) -> float:
        """Calculate semantic coherence of search results.
        
        This is a placeholder for more sophisticated embedding analysis.
        In a full implementation, this would analyze the semantic similarity
        between query embeddings and result embeddings.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Coherence score as a float between 0 and 1
        """
        # Placeholder implementation
        # In practice, this would:
        # 1. Generate embeddings for queries and results
        # 2. Calculate cosine similarity
        # 3. Analyze clustering and coherence patterns
        
        successful_results = [r for r in query_results if r.get('success', False)]
        if not successful_results:
            return 0.0
        
        # Simple heuristic: assume results with scores are more coherent
        coherence_scores = []
        for result in successful_results:
            result_scores = [r.get('score', 0.5) for r in result.get('results', [])]
            if result_scores:
                # Higher variance in scores might indicate better discrimination
                variance = statistics.variance(result_scores) if len(result_scores) > 1 else 0
                # Normalize variance to 0-1 range (heuristic)
                normalized_coherence = min(variance * 4, 1.0)
                coherence_scores.append(normalized_coherence)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
    
    def embedding_coverage(self, query_results: List[Dict[str, Any]]) -> float:
        """Measure how well embeddings cover the semantic space.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Coverage score as a float between 0 and 1
        """
        # Placeholder implementation
        # This would analyze the distribution of embeddings in the vector space
        return 0.8  # Placeholder value
    
    def semantic_drift(self, query_results: List[Dict[str, Any]]) -> float:
        """Measure semantic drift in search results.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Drift score as a float (lower is better)
        """
        # Placeholder implementation
        # This would measure how much results drift from the original query intent
        return 0.2  # Placeholder value


class ComparisonMetrics:
    """Metrics for comparing different search approaches."""
    
    def relative_improvement(self, baseline_metric: float, new_metric: float) -> float:
        """Calculate relative improvement over baseline.
        
        Args:
            baseline_metric: Baseline metric value
            new_metric: New metric value
            
        Returns:
            Relative improvement as a percentage
        """
        if baseline_metric == 0:
            return float('inf') if new_metric > 0 else 0.0
        
        return ((new_metric - baseline_metric) / baseline_metric) * 100
    
    def statistical_significance(self, baseline_scores: List[float], new_scores: List[float]) -> Dict[str, float]:
        """Calculate statistical significance of improvement.
        
        Args:
            baseline_scores: List of baseline metric scores
            new_scores: List of new metric scores
            
        Returns:
            Dictionary with statistical test results
        """
        # Placeholder for statistical tests (t-test, Mann-Whitney U, etc.)
        # Would require scipy.stats in a full implementation
        
        if not baseline_scores or not new_scores:
            return {'p_value': 1.0, 'significant': False}
        
        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        new_mean = sum(new_scores) / len(new_scores)
        
        # Simplified significance test (placeholder)
        improvement = abs(new_mean - baseline_mean) / max(baseline_mean, 0.001)
        p_value = max(0.001, 1.0 - improvement)  # Simplified p-value calculation
        
        return {
            'baseline_mean': baseline_mean,
            'new_mean': new_mean,
            'improvement': improvement,
            'p_value': p_value,
            'significant': p_value < 0.05
        }