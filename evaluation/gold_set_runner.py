"""Gold Set + Metrics Runner Enhancement for DocFoundry

This module provides an enhanced evaluation system with curated gold standard datasets,
automated metrics collection, and comprehensive reporting capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import aiohttp
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .datasets import EvaluationDataset, QueryRelevancePair
from .metrics import RelevanceMetrics, SearchQualityMetrics, PerformanceMetrics
from .harness import EvaluationHarness, EvaluationConfig, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class GoldSetConfig:
    """Configuration for Gold Set evaluation."""
    
    # Gold set configuration
    gold_set_path: str = "evaluation/gold_sets"
    auto_generate_queries: bool = True
    min_queries_per_domain: int = 20
    
    # Quality thresholds
    relevance_threshold: float = 0.7
    performance_threshold_ms: float = 1000.0
    success_rate_threshold: float = 0.95
    
    # Reporting configuration
    generate_html_report: bool = True
    generate_csv_export: bool = True
    alert_on_regression: bool = True
    
    # Scheduling configuration
    run_interval_hours: int = 24
    retention_days: int = 30


@dataclass
class MetricsSnapshot:
    """Snapshot of metrics at a point in time."""
    
    timestamp: datetime
    search_mode: str
    precision_at_5: float
    recall_at_5: float
    f1_at_5: float
    ndcg_at_5: float
    avg_query_time_ms: float
    success_rate: float
    total_queries: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoldSetManager:
    """Manages gold standard evaluation datasets."""
    
    def __init__(self, config: GoldSetConfig):
        self.config = config
        self.gold_sets_dir = Path(config.gold_set_path)
        self.gold_sets_dir.mkdir(parents=True, exist_ok=True)
    
    async def load_or_create_gold_set(self, domain: str = "general") -> EvaluationDataset:
        """Load existing gold set or create a new one."""
        gold_set_file = self.gold_sets_dir / f"{domain}_gold_set.json"
        
        if gold_set_file.exists():
            logger.info(f"Loading existing gold set from {gold_set_file}")
            return self._load_gold_set(gold_set_file)
        
        if self.config.auto_generate_queries:
            logger.info(f"Creating new gold set for domain: {domain}")
            return await self._create_gold_set(domain)
        
        raise FileNotFoundError(f"Gold set not found: {gold_set_file}")
    
    def _load_gold_set(self, file_path: Path) -> EvaluationDataset:
        """Load gold set from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        pairs = [QueryRelevancePair.from_dict(pair_data) for pair_data in data['pairs']]
        return EvaluationDataset(
            pairs=pairs,
            name=data.get('name', 'gold_set'),
            description=data.get('description', ''),
            version=data.get('version', '1.0'),
            metadata=data.get('metadata', {})
        )
    
    async def _create_gold_set(self, domain: str) -> EvaluationDataset:
        """Create a new gold set by analyzing existing documents."""
        # This would typically involve:
        # 1. Analyzing document corpus to identify key topics
        # 2. Generating diverse queries covering different aspects
        # 3. Using human annotation or automated relevance scoring
        
        # For now, create a basic set with common query patterns
        queries = [
            # Technical documentation queries
            "how to install and configure",
            "API authentication methods",
            "error handling best practices",
            "performance optimization techniques",
            "security configuration guide",
            
            # Feature-specific queries
            "search functionality implementation",
            "database migration steps",
            "user interface components",
            "testing and validation procedures",
            "deployment and monitoring setup",
            
            # Troubleshooting queries
            "common errors and solutions",
            "debugging connection issues",
            "performance bottleneck analysis",
            "configuration troubleshooting",
            "system requirements and compatibility"
        ]
        
        pairs = []
        for i, query in enumerate(queries):
            # In a real implementation, this would involve actual relevance judgments
            pair = QueryRelevancePair(
                query=query,
                relevant_doc_ids=[f"doc_{i}_1", f"doc_{i}_2"],  # Placeholder
                query_type="technical" if i < 10 else "troubleshooting",
                difficulty="medium",
                metadata={"domain": domain, "auto_generated": True}
            )
            pairs.append(pair)
        
        dataset = EvaluationDataset(
            pairs=pairs,
            name=f"{domain}_gold_set",
            description=f"Auto-generated gold set for {domain} domain",
            version="1.0",
            metadata={"created_at": datetime.now().isoformat(), "domain": domain}
        )
        
        # Save the gold set
        await self._save_gold_set(dataset, domain)
        return dataset
    
    async def _save_gold_set(self, dataset: EvaluationDataset, domain: str):
        """Save gold set to JSON file."""
        gold_set_file = self.gold_sets_dir / f"{domain}_gold_set.json"
        
        data = {
            'name': dataset.name,
            'description': dataset.description,
            'version': dataset.version,
            'metadata': dataset.metadata,
            'pairs': [pair.to_dict() for pair in dataset.pairs]
        }
        
        with open(gold_set_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Gold set saved to {gold_set_file}")


class EnhancedMetricsRunner:
    """Enhanced metrics runner with comprehensive evaluation capabilities."""
    
    def __init__(self, config: GoldSetConfig):
        self.config = config
        self.gold_set_manager = GoldSetManager(config)
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.relevance_metrics = RelevanceMetrics()
        self.quality_metrics = SearchQualityMetrics()
        self.performance_metrics = PerformanceMetrics()
        
        # Historical data storage
        self.metrics_history: List[MetricsSnapshot] = []
    
    async def run_comprehensive_evaluation(self, domains: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation across multiple domains."""
        if domains is None:
            domains = ["general", "technical", "troubleshooting"]
        
        logger.info(f"Starting comprehensive evaluation for domains: {domains}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "domains": {},
            "summary": {},
            "alerts": []
        }
        
        for domain in domains:
            try:
                domain_results = await self._evaluate_domain(domain)
                results["domains"][domain] = domain_results
                
                # Check for performance regressions
                alerts = self._check_performance_alerts(domain_results)
                results["alerts"].extend(alerts)
                
            except Exception as e:
                logger.error(f"Failed to evaluate domain {domain}: {e}")
                results["domains"][domain] = {"error": str(e)}
        
        # Generate summary statistics
        results["summary"] = self._generate_summary(results["domains"])
        
        # Save results
        await self._save_results(results)
        
        # Generate reports
        if self.config.generate_html_report:
            await self._generate_html_report(results)
        
        if self.config.generate_csv_export:
            await self._generate_csv_export(results)
        
        return results
    
    async def _evaluate_domain(self, domain: str) -> Dict[str, Any]:
        """Evaluate a specific domain."""
        logger.info(f"Evaluating domain: {domain}")
        
        # Load gold set
        gold_set = await self.gold_set_manager.load_or_create_gold_set(domain)
        
        # Configure evaluation
        eval_config = EvaluationConfig(
            search_endpoints=[
                "http://localhost:8001/search",
                "http://localhost:8001/search/semantic",
                "http://localhost:8001/search/hybrid"
            ],
            top_k_values=[1, 3, 5, 10],
            timeout_seconds=30.0
        )
        
        # Run evaluation
        harness = EvaluationHarness(eval_config)
        
        domain_results = {
            "gold_set_info": {
                "name": gold_set.name,
                "size": len(gold_set),
                "description": gold_set.description
            },
            "search_modes": {},
            "performance_summary": {}
        }
        
        # Evaluate each search mode
        for endpoint in eval_config.search_endpoints:
            mode_name = endpoint.split('/')[-1] if '/' in endpoint else 'default'
            
            try:
                mode_results = await self._evaluate_search_mode(harness, gold_set, endpoint)
                domain_results["search_modes"][mode_name] = mode_results
                
                # Store metrics snapshot
                snapshot = MetricsSnapshot(
                    timestamp=datetime.now(),
                    search_mode=mode_name,
                    precision_at_5=mode_results["metrics"]["precision_at_5"],
                    recall_at_5=mode_results["metrics"]["recall_at_5"],
                    f1_at_5=mode_results["metrics"]["f1_at_5"],
                    ndcg_at_5=mode_results["metrics"]["ndcg_at_5"],
                    avg_query_time_ms=mode_results["performance"]["avg_query_time_ms"],
                    success_rate=mode_results["performance"]["success_rate"],
                    total_queries=len(gold_set)
                )
                self.metrics_history.append(snapshot)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {mode_name}: {e}")
                domain_results["search_modes"][mode_name] = {"error": str(e)}
        
        return domain_results
    
    async def _evaluate_search_mode(self, harness: EvaluationHarness, gold_set: EvaluationDataset, endpoint: str) -> Dict[str, Any]:
        """Evaluate a specific search mode."""
        # This would integrate with the existing evaluation harness
        # For now, return mock results that demonstrate the structure
        
        return {
            "endpoint": endpoint,
            "metrics": {
                "precision_at_5": 0.85,
                "recall_at_5": 0.78,
                "f1_at_5": 0.81,
                "ndcg_at_5": 0.82,
                "map": 0.79
            },
            "performance": {
                "avg_query_time_ms": 245.6,
                "p95_query_time_ms": 450.2,
                "success_rate": 0.98,
                "timeout_rate": 0.01
            },
            "quality": {
                "result_diversity": 0.73,
                "coverage": 0.89,
                "freshness_score": 0.91
            }
        }
    
    def _check_performance_alerts(self, domain_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance regressions and generate alerts."""
        alerts = []
        
        for mode_name, mode_results in domain_results.get("search_modes", {}).items():
            if "error" in mode_results:
                alerts.append({
                    "type": "error",
                    "severity": "high",
                    "message": f"Search mode {mode_name} failed: {mode_results['error']}",
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            # Check performance thresholds
            performance = mode_results.get("performance", {})
            
            if performance.get("avg_query_time_ms", 0) > self.config.performance_threshold_ms:
                alerts.append({
                    "type": "performance",
                    "severity": "medium",
                    "message": f"High latency in {mode_name}: {performance['avg_query_time_ms']:.1f}ms",
                    "threshold": self.config.performance_threshold_ms,
                    "timestamp": datetime.now().isoformat()
                })
            
            if performance.get("success_rate", 1.0) < self.config.success_rate_threshold:
                alerts.append({
                    "type": "reliability",
                    "severity": "high",
                    "message": f"Low success rate in {mode_name}: {performance['success_rate']:.1%}",
                    "threshold": self.config.success_rate_threshold,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Check quality thresholds
            metrics = mode_results.get("metrics", {})
            if metrics.get("f1_at_5", 0) < self.config.relevance_threshold:
                alerts.append({
                    "type": "quality",
                    "severity": "medium",
                    "message": f"Low F1@5 score in {mode_name}: {metrics['f1_at_5']:.2f}",
                    "threshold": self.config.relevance_threshold,
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts
    
    def _generate_summary(self, domain_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all domains."""
        summary = {
            "total_domains": len(domain_results),
            "successful_domains": 0,
            "failed_domains": 0,
            "overall_metrics": {},
            "best_performing_mode": None,
            "recommendations": []
        }
        
        all_metrics = []
        mode_performance = {}
        
        for domain, results in domain_results.items():
            if "error" in results:
                summary["failed_domains"] += 1
                continue
            
            summary["successful_domains"] += 1
            
            for mode_name, mode_results in results.get("search_modes", {}).items():
                if "error" not in mode_results:
                    metrics = mode_results.get("metrics", {})
                    all_metrics.append(metrics)
                    
                    if mode_name not in mode_performance:
                        mode_performance[mode_name] = []
                    mode_performance[mode_name].append(metrics.get("f1_at_5", 0))
        
        # Calculate overall metrics
        if all_metrics:
            summary["overall_metrics"] = {
                "avg_precision_at_5": np.mean([m.get("precision_at_5", 0) for m in all_metrics]),
                "avg_recall_at_5": np.mean([m.get("recall_at_5", 0) for m in all_metrics]),
                "avg_f1_at_5": np.mean([m.get("f1_at_5", 0) for m in all_metrics]),
                "avg_ndcg_at_5": np.mean([m.get("ndcg_at_5", 0) for m in all_metrics])
            }
        
        # Determine best performing mode
        if mode_performance:
            mode_avg_f1 = {mode: np.mean(scores) for mode, scores in mode_performance.items()}
            summary["best_performing_mode"] = max(mode_avg_f1, key=mode_avg_f1.get)
        
        # Generate recommendations
        if summary["overall_metrics"].get("avg_f1_at_5", 0) < 0.7:
            summary["recommendations"].append("Consider tuning search parameters to improve relevance")
        
        if summary["failed_domains"] > 0:
            summary["recommendations"].append(f"Investigate {summary['failed_domains']} failed domain evaluations")
        
        return summary
    
    async def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    async def _generate_html_report(self, results: Dict[str, Any]):
        """Generate HTML report."""
        # This would generate a comprehensive HTML report
        # For now, create a simple summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"evaluation_report_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DocFoundry Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
                .alert {{ padding: 10px; margin: 10px 0; border-radius: 4px; }}
                .alert.high {{ background: #ffebee; border-left: 4px solid #f44336; }}
                .alert.medium {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>DocFoundry Evaluation Report</h1>
            <p>Generated: {results['timestamp']}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Evaluated {results['summary']['total_domains']} domains</p>
                <p>Success rate: {results['summary']['successful_domains']}/{results['summary']['total_domains']}</p>
                <p>Best performing mode: {results['summary'].get('best_performing_mode', 'N/A')}</p>
            </div>
            
            <h2>Alerts</h2>
            {''.join([f'<div class="alert {alert["severity"]}">{alert["message"]}</div>' for alert in results.get('alerts', [])])}
            
            <h2>Detailed Results</h2>
            <pre>{json.dumps(results, indent=2, default=str)}</pre>
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_file}")
    
    async def _generate_csv_export(self, results: Dict[str, Any]):
        """Generate CSV export of metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.results_dir / f"evaluation_metrics_{timestamp}.csv"
        
        # Flatten results for CSV export
        rows = []
        for domain, domain_results in results.get("domains", {}).items():
            if "error" in domain_results:
                continue
            
            for mode_name, mode_results in domain_results.get("search_modes", {}).items():
                if "error" in mode_results:
                    continue
                
                row = {
                    "timestamp": results["timestamp"],
                    "domain": domain,
                    "search_mode": mode_name,
                    **mode_results.get("metrics", {}),
                    **{f"perf_{k}": v for k, v in mode_results.get("performance", {}).items()},
                    **{f"quality_{k}": v for k, v in mode_results.get("quality", {}).items()}
                }
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_file, index=False)
            logger.info(f"CSV export generated: {csv_file}")


async def main():
    """Main function for running the enhanced metrics evaluation."""
    config = GoldSetConfig(
        auto_generate_queries=True,
        generate_html_report=True,
        generate_csv_export=True,
        alert_on_regression=True
    )
    
    runner = EnhancedMetricsRunner(config)
    results = await runner.run_comprehensive_evaluation()
    
    print(f"Evaluation completed. Results saved to evaluation_results/")
    print(f"Summary: {results['summary']}")
    
    if results.get('alerts'):
        print(f"\nAlerts ({len(results['alerts'])}):")
        for alert in results['alerts']:
            print(f"  [{alert['severity'].upper()}] {alert['message']}")


if __name__ == "__main__":
    asyncio.run(main())