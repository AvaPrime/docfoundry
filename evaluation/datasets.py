"""Evaluation Datasets for DocFoundry Search Quality Assessment

This module provides tools for creating, loading, and managing evaluation datasets
for search quality testing.
"""

import json
import random
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryRelevancePair:
    """A query with its relevant document IDs."""
    
    query: str
    relevant_doc_ids: List[str]
    query_type: str = "general"  # general, specific, ambiguous, etc.
    difficulty: str = "medium"   # easy, medium, hard
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryRelevancePair':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvaluationDataset:
    """Collection of query-relevance pairs for evaluation."""
    
    pairs: List[QueryRelevancePair]
    name: str = "evaluation_dataset"
    description: str = ""
    version: str = "1.0"
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def add_pair(self, pair: QueryRelevancePair) -> None:
        """Add a query-relevance pair to the dataset."""
        self.pairs.append(pair)
    
    def filter_by_type(self, query_type: str) -> 'EvaluationDataset':
        """Filter dataset by query type."""
        filtered_pairs = [p for p in self.pairs if p.query_type == query_type]
        return EvaluationDataset(
            pairs=filtered_pairs,
            name=f"{self.name}_{query_type}",
            description=f"Filtered by query type: {query_type}",
            version=self.version,
            metadata=self.metadata.copy()
        )
    
    def filter_by_difficulty(self, difficulty: str) -> 'EvaluationDataset':
        """Filter dataset by difficulty level."""
        filtered_pairs = [p for p in self.pairs if p.difficulty == difficulty]
        return EvaluationDataset(
            pairs=filtered_pairs,
            name=f"{self.name}_{difficulty}",
            description=f"Filtered by difficulty: {difficulty}",
            version=self.version,
            metadata=self.metadata.copy()
        )
    
    def sample(self, n: int, random_seed: Optional[int] = None) -> 'EvaluationDataset':
        """Sample n pairs from the dataset."""
        if random_seed is not None:
            random.seed(random_seed)
        
        sampled_pairs = random.sample(self.pairs, min(n, len(self.pairs)))
        return EvaluationDataset(
            pairs=sampled_pairs,
            name=f"{self.name}_sample_{n}",
            description=f"Random sample of {len(sampled_pairs)} pairs",
            version=self.version,
            metadata=self.metadata.copy()
        )
    
    def split(self, train_ratio: float = 0.8, random_seed: Optional[int] = None) -> Tuple['EvaluationDataset', 'EvaluationDataset']:
        """Split dataset into train and test sets."""
        if random_seed is not None:
            random.seed(random_seed)
        
        shuffled_pairs = self.pairs.copy()
        random.shuffle(shuffled_pairs)
        
        split_idx = int(len(shuffled_pairs) * train_ratio)
        train_pairs = shuffled_pairs[:split_idx]
        test_pairs = shuffled_pairs[split_idx:]
        
        train_dataset = EvaluationDataset(
            pairs=train_pairs,
            name=f"{self.name}_train",
            description=f"Training split ({len(train_pairs)} pairs)",
            version=self.version,
            metadata=self.metadata.copy()
        )
        
        test_dataset = EvaluationDataset(
            pairs=test_pairs,
            name=f"{self.name}_test",
            description=f"Test split ({len(test_pairs)} pairs)",
            version=self.version,
            metadata=self.metadata.copy()
        )
        
        return train_dataset, test_dataset
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.pairs:
            return {}
        
        query_types = [p.query_type for p in self.pairs]
        difficulties = [p.difficulty for p in self.pairs]
        relevant_counts = [len(p.relevant_doc_ids) for p in self.pairs]
        
        stats = {
            'total_pairs': len(self.pairs),
            'unique_queries': len(set(p.query for p in self.pairs)),
            'query_types': {
                qt: query_types.count(qt) for qt in set(query_types)
            },
            'difficulties': {
                d: difficulties.count(d) for d in set(difficulties)
            },
            'relevant_docs': {
                'min': min(relevant_counts),
                'max': max(relevant_counts),
                'avg': sum(relevant_counts) / len(relevant_counts),
                'total_unique': len(set(
                    doc_id for pair in self.pairs for doc_id in pair.relevant_doc_ids
                ))
            }
        }
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'metadata': self.metadata,
            'pairs': [pair.to_dict() for pair in self.pairs]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationDataset':
        """Create from dictionary."""
        pairs = [QueryRelevancePair.from_dict(pair_data) for pair_data in data.get('pairs', [])]
        return cls(
            pairs=pairs,
            name=data.get('name', 'evaluation_dataset'),
            description=data.get('description', ''),
            version=data.get('version', '1.0'),
            metadata=data.get('metadata', {})
        )
    
    def save(self, file_path: str) -> None:
        """Save dataset to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Dataset saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'EvaluationDataset':
        """Load dataset from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        dataset = cls.from_dict(data)
        logger.info(f"Dataset loaded from {file_path}: {len(dataset)} pairs")
        return dataset


class DatasetGenerator:
    """Generator for creating synthetic evaluation datasets."""
    
    def __init__(self, db_path: str = "data/docfoundry.db"):
        self.db_path = db_path
        self._sample_queries = [
            "API documentation",
            "authentication methods",
            "database configuration",
            "error handling",
            "getting started guide",
            "installation instructions",
            "performance optimization",
            "security best practices",
            "troubleshooting guide",
            "user management",
            "data migration",
            "backup procedures",
            "monitoring setup",
            "deployment guide",
            "configuration options"
        ]
    
    def _get_available_documents(self) -> List[Tuple[str, str, str]]:
        """Get available documents from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT doc_id, title, content 
                FROM chunks 
                LIMIT 1000
            """)
            
            documents = cursor.fetchall()
            conn.close()
            
            return documents
            
        except Exception as e:
            logger.warning(f"Could not load documents from database: {e}")
            return self._generate_mock_documents()
    
    def _generate_mock_documents(self) -> List[Tuple[str, str, str]]:
        """Generate mock documents for testing."""
        mock_docs = []
        topics = [
            "API", "authentication", "database", "error", "guide", 
            "installation", "performance", "security", "troubleshooting", 
            "user", "migration", "backup", "monitoring", "deployment", "configuration"
        ]
        
        for i, topic in enumerate(topics):
            for j in range(3):  # 3 docs per topic
                doc_id = f"doc_{topic}_{j}"
                title = f"{topic.title()} Documentation {j+1}"
                content = f"This document covers {topic} related information. " * 10
                mock_docs.append((doc_id, title, content))
        
        return mock_docs
    
    def _find_relevant_documents(self, query: str, documents: List[Tuple[str, str, str]]) -> List[str]:
        """Find documents relevant to a query using simple keyword matching."""
        query_words = set(query.lower().split())
        relevant_docs = []
        
        for doc_id, title, content in documents:
            doc_text = (title + " " + content).lower()
            doc_words = set(doc_text.split())
            
            # Simple relevance: intersection of words
            overlap = len(query_words & doc_words)
            if overlap > 0:
                relevant_docs.append((doc_id, overlap))
        
        # Sort by relevance and return top matches
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in relevant_docs[:5]]  # Top 5 relevant docs
    
    def _generate_query_variations(self, base_query: str) -> List[str]:
        """Generate variations of a base query."""
        variations = [base_query]
        
        # Add question variations
        variations.append(f"How to {base_query}?")
        variations.append(f"What is {base_query}?")
        variations.append(f"Where to find {base_query}?")
        
        # Add specific variations
        variations.append(f"{base_query} tutorial")
        variations.append(f"{base_query} example")
        variations.append(f"{base_query} setup")
        
        return variations
    
    def generate_synthetic_dataset(self, size: int = 100) -> EvaluationDataset:
        """Generate a synthetic evaluation dataset.
        
        Args:
            size: Number of query-relevance pairs to generate
            
        Returns:
            EvaluationDataset with synthetic data
        """
        logger.info(f"Generating synthetic dataset with {size} pairs")
        
        documents = self._get_available_documents()
        if not documents:
            logger.warning("No documents available, creating minimal dataset")
            return EvaluationDataset(
                pairs=[],
                name="empty_synthetic_dataset",
                description="Empty dataset - no documents available"
            )
        
        pairs = []
        query_types = ["general", "specific", "ambiguous"]
        difficulties = ["easy", "medium", "hard"]
        
        for i in range(size):
            # Select base query
            base_query = random.choice(self._sample_queries)
            
            # Generate query variations
            query_variations = self._generate_query_variations(base_query)
            query = random.choice(query_variations)
            
            # Find relevant documents
            relevant_doc_ids = self._find_relevant_documents(query, documents)
            
            # Assign random metadata
            query_type = random.choice(query_types)
            difficulty = random.choice(difficulties)
            
            # Create pair
            pair = QueryRelevancePair(
                query=query,
                relevant_doc_ids=relevant_doc_ids,
                query_type=query_type,
                difficulty=difficulty,
                metadata={
                    'generated': True,
                    'base_query': base_query,
                    'generation_method': 'keyword_matching'
                }
            )
            
            pairs.append(pair)
        
        dataset = EvaluationDataset(
            pairs=pairs,
            name="synthetic_evaluation_dataset",
            description=f"Synthetically generated dataset with {len(pairs)} query-relevance pairs",
            version="1.0",
            metadata={
                'generation_method': 'synthetic',
                'document_count': len(documents),
                'generation_timestamp': str(Path().cwd())
            }
        )
        
        logger.info(f"Generated synthetic dataset: {dataset.get_statistics()}")
        return dataset
    
    def generate_from_search_logs(self, log_file: str) -> EvaluationDataset:
        """Generate dataset from search logs (placeholder).
        
        Args:
            log_file: Path to search log file
            
        Returns:
            EvaluationDataset created from logs
        """
        # Placeholder implementation
        # In practice, this would parse search logs and extract:
        # - Queries that were performed
        # - Documents that were clicked/viewed
        # - Session information for relevance inference
        
        logger.info(f"Generating dataset from search logs: {log_file}")
        
        # For now, return empty dataset
        return EvaluationDataset(
            pairs=[],
            name="log_based_dataset",
            description=f"Dataset generated from search logs: {log_file}",
            metadata={'source': 'search_logs', 'log_file': log_file}
        )


# Convenience functions

async def create_synthetic_dataset(size: int = 100, db_path: str = "data/docfoundry.db") -> EvaluationDataset:
    """Create a synthetic evaluation dataset.
    
    Args:
        size: Number of query-relevance pairs to generate
        db_path: Path to the DocFoundry database
        
    Returns:
        EvaluationDataset with synthetic data
    """
    generator = DatasetGenerator(db_path)
    return generator.generate_synthetic_dataset(size)


async def load_evaluation_dataset(file_path: str) -> EvaluationDataset:
    """Load an evaluation dataset from file.
    
    Args:
        file_path: Path to the dataset JSON file
        
    Returns:
        EvaluationDataset loaded from file
    """
    return EvaluationDataset.load(file_path)


def create_manual_dataset(pairs_data: List[Dict[str, Any]]) -> EvaluationDataset:
    """Create a dataset from manually curated data.
    
    Args:
        pairs_data: List of dictionaries containing query-relevance pair data
        
    Returns:
        EvaluationDataset with manual data
    """
    pairs = [QueryRelevancePair.from_dict(pair_data) for pair_data in pairs_data]
    
    return EvaluationDataset(
        pairs=pairs,
        name="manual_evaluation_dataset",
        description=f"Manually curated dataset with {len(pairs)} pairs",
        metadata={'source': 'manual_curation'}
    )


def merge_datasets(*datasets: EvaluationDataset) -> EvaluationDataset:
    """Merge multiple evaluation datasets.
    
    Args:
        *datasets: Variable number of EvaluationDataset objects
        
    Returns:
        Merged EvaluationDataset
    """
    all_pairs = []
    dataset_names = []
    
    for dataset in datasets:
        all_pairs.extend(dataset.pairs)
        dataset_names.append(dataset.name)
    
    merged_name = "merged_" + "_".join(dataset_names)
    
    return EvaluationDataset(
        pairs=all_pairs,
        name=merged_name,
        description=f"Merged dataset from {len(datasets)} sources",
        metadata={
            'source': 'merged',
            'source_datasets': dataset_names,
            'source_count': len(datasets)
        }
    )