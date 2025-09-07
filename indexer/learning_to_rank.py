"""Learning-to-Rank Module for DocFoundry

This module implements learning-to-rank improvements for search quality,
including click feedback collection, feature extraction, and ranking model training.
"""

import sqlite3
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClickEvent:
    """Represents a user click event for learning-to-rank."""
    query: str
    doc_id: str
    chunk_id: str
    position: int  # Position in search results (0-based)
    timestamp: datetime
    session_id: str
    user_id: Optional[str] = None
    dwell_time: Optional[float] = None  # Time spent on document (seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClickEvent':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class SearchFeatures:
    """Features extracted for a query-document pair."""
    # Query features
    query_length: int
    query_terms: int
    
    # Document features
    doc_length: int
    title_match: bool
    heading_match: bool
    
    # Relevance features
    bm25_score: float
    semantic_similarity: float
    rrf_score: float
    
    # Position features
    original_rank: int
    
    # Historical features
    click_rate: float = 0.0
    avg_dwell_time: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML models."""
        return np.array([
            self.query_length,
            self.query_terms,
            self.doc_length,
            float(self.title_match),
            float(self.heading_match),
            self.bm25_score,
            self.semantic_similarity,
            self.rrf_score,
            self.original_rank,
            self.click_rate,
            self.avg_dwell_time
        ])


class ClickLogger:
    """Handles click event logging and storage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_click_tables()
    
    def _init_click_tables(self):
        """Initialize click tracking tables."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS click_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    dwell_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_click_events_query ON click_events(query);
                CREATE INDEX IF NOT EXISTS idx_click_events_chunk ON click_events(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_click_events_timestamp ON click_events(timestamp);
                
                CREATE TABLE IF NOT EXISTS search_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    query_count INTEGER DEFAULT 0,
                    click_count INTEGER DEFAULT 0
                );
            """)
            conn.commit()
        finally:
            conn.close()
    
    def log_click(self, event: ClickEvent):
        """Log a click event."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO click_events 
                (query, doc_id, chunk_id, position, timestamp, session_id, user_id, dwell_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.query,
                event.doc_id,
                event.chunk_id,
                event.position,
                event.timestamp.isoformat(),
                event.session_id,
                event.user_id,
                event.dwell_time
            ))
            conn.commit()
            logger.info(f"Logged click event: query='{event.query}', chunk_id={event.chunk_id}, position={event.position}")
        finally:
            conn.close()
    
    def get_click_stats(self, query: str = None, days: int = 30) -> Dict[str, Any]:
        """Get click statistics for analysis."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            if query:
                # Query-specific stats
                stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_clicks,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        AVG(position) as avg_click_position,
                        AVG(dwell_time) as avg_dwell_time
                    FROM click_events 
                    WHERE query = ? AND timestamp >= ?
                """, (query, since_date)).fetchone()
            else:
                # Overall stats
                stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_clicks,
                        COUNT(DISTINCT query) as unique_queries,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        AVG(position) as avg_click_position,
                        AVG(dwell_time) as avg_dwell_time
                    FROM click_events 
                    WHERE timestamp >= ?
                """, (since_date,)).fetchone()
            
            return dict(stats) if stats else {}
        finally:
            conn.close()


class FeatureExtractor:
    """Extracts features for learning-to-rank."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.click_logger = ClickLogger(db_path)
    
    def extract_features(self, query: str, chunk_data: Dict[str, Any], 
                        original_rank: int, bm25_score: float = 0.0,
                        semantic_similarity: float = 0.0, rrf_score: float = 0.0) -> SearchFeatures:
        """Extract features for a query-document pair."""
        
        # Query features
        query_terms = len(query.split())
        query_length = len(query)
        
        # Document features
        doc_text = chunk_data.get('text', '')
        doc_length = len(doc_text)
        title = chunk_data.get('title', '').lower()
        heading = chunk_data.get('heading', '').lower()
        query_lower = query.lower()
        
        title_match = query_lower in title
        heading_match = query_lower in heading
        
        # Historical features
        click_rate, avg_dwell_time = self._get_historical_features(chunk_data['id'])
        
        return SearchFeatures(
            query_length=query_length,
            query_terms=query_terms,
            doc_length=doc_length,
            title_match=title_match,
            heading_match=heading_match,
            bm25_score=bm25_score,
            semantic_similarity=semantic_similarity,
            rrf_score=rrf_score,
            original_rank=original_rank,
            click_rate=click_rate,
            avg_dwell_time=avg_dwell_time
        )
    
    def _get_historical_features(self, chunk_id: str, days: int = 30) -> Tuple[float, float]:
        """Get historical click features for a chunk."""
        conn = sqlite3.connect(self.db_path)
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Get click rate and average dwell time
            result = conn.execute("""
                SELECT 
                    COUNT(*) as clicks,
                    AVG(dwell_time) as avg_dwell_time
                FROM click_events 
                WHERE chunk_id = ? AND timestamp >= ?
            """, (chunk_id, since_date)).fetchone()
            
            if result and result[0] > 0:
                # Normalize click rate (simple approach)
                click_rate = min(result[0] / 100.0, 1.0)  # Cap at 1.0
                avg_dwell_time = result[1] or 0.0
            else:
                click_rate = 0.0
                avg_dwell_time = 0.0
            
            return click_rate, avg_dwell_time
        finally:
            conn.close()


class LearningToRankReranker:
    """Learning-to-rank reranker using click feedback."""
    
    def __init__(self, db_path: str, model_path: Optional[str] = None):
        self.db_path = db_path
        self.model_path = model_path or str(Path(db_path).parent / "ltr_model.json")
        self.feature_extractor = FeatureExtractor(db_path)
        self.click_logger = ClickLogger(db_path)
        
        # Simple linear model weights (can be replaced with more sophisticated models)
        self.weights = self._load_or_init_weights()
    
    def _load_or_init_weights(self) -> np.ndarray:
        """Load model weights or initialize with defaults."""
        try:
            if Path(self.model_path).exists():
                with open(self.model_path, 'r') as f:
                    data = json.load(f)
                    return np.array(data['weights'])
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}")
        
        # Default weights (can be tuned)
        return np.array([
            0.1,   # query_length
            0.2,   # query_terms
            0.05,  # doc_length
            0.3,   # title_match
            0.2,   # heading_match
            0.4,   # bm25_score
            0.5,   # semantic_similarity
            0.6,   # rrf_score
            -0.1,  # original_rank (negative because lower rank is better)
            0.8,   # click_rate (strong signal)
            0.3    # avg_dwell_time
        ])
    
    def _save_weights(self):
        """Save model weights to disk."""
        try:
            with open(self.model_path, 'w') as f:
                json.dump({
                    'weights': self.weights.tolist(),
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save model weights: {e}")
    
    def rerank_results(self, query: str, results: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        """Rerank search results using learning-to-rank."""
        if not results:
            return results
        
        # Extract features and compute LTR scores
        ltr_results = []
        for rank, (chunk_data, original_score) in enumerate(results):
            features = self.feature_extractor.extract_features(
                query=query,
                chunk_data=chunk_data,
                original_rank=rank,
                rrf_score=original_score
            )
            
            # Compute LTR score using linear model
            ltr_score = np.dot(self.weights, features.to_array())
            
            # Combine with original score (weighted average)
            combined_score = 0.7 * original_score + 0.3 * ltr_score
            
            ltr_results.append((chunk_data, combined_score))
        
        # Sort by combined score
        ltr_results.sort(key=lambda x: x[1], reverse=True)
        return ltr_results
    
    def log_click_feedback(self, query: str, clicked_chunk_id: str, position: int, 
                          session_id: str, user_id: Optional[str] = None, 
                          dwell_time: Optional[float] = None):
        """Log click feedback for model improvement."""
        event = ClickEvent(
            query=query,
            doc_id="",  # Will be filled from chunk data if needed
            chunk_id=clicked_chunk_id,
            position=position,
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            dwell_time=dwell_time
        )
        
        self.click_logger.log_click(event)
    
    def update_model(self, learning_rate: float = 0.01):
        """Update model weights based on recent click feedback."""
        # This is a simplified update mechanism
        # In practice, you'd want more sophisticated learning algorithms
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Get recent click data for training
            since_date = (datetime.now() - timedelta(days=7)).isoformat()
            
            clicks = conn.execute("""
                SELECT query, chunk_id, position, dwell_time
                FROM click_events 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 1000
            """, (since_date,)).fetchall()
            
            if len(clicks) < 10:  # Need minimum data for training
                logger.info("Insufficient click data for model update")
                return
            
            # Simple weight adjustment based on click patterns
            # Higher weights for features that correlate with clicks
            click_positions = [click['position'] for click in clicks]
            avg_click_position = np.mean(click_positions)
            
            # Adjust weights to favor features that lead to higher-ranked clicks
            if avg_click_position < 3:  # Good performance
                self.weights[9] *= (1 + learning_rate)  # Increase click_rate weight
                self.weights[10] *= (1 + learning_rate)  # Increase dwell_time weight
            else:  # Poor performance
                self.weights[9] *= (1 - learning_rate)  # Decrease click_rate weight
            
            # Normalize weights
            self.weights = self.weights / np.linalg.norm(self.weights)
            
            self._save_weights()
            logger.info(f"Updated LTR model weights based on {len(clicks)} click events")
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
        finally:
            conn.close()
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics and performance metrics."""
        click_stats = self.click_logger.get_click_stats()
        
        return {
            'model_path': self.model_path,
            'weights': self.weights.tolist(),
            'click_stats': click_stats,
            'last_updated': datetime.now().isoformat()
        }


def main():
    """CLI for learning-to-rank operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DocFoundry Learning-to-Rank CLI")
    parser.add_argument("--db", default="data/docfoundry.db", help="Database path")
    parser.add_argument("--update-model", action="store_true", help="Update LTR model from click data")
    parser.add_argument("--stats", action="store_true", help="Show model statistics")
    parser.add_argument("--test-rerank", help="Test reranking with a query")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    reranker = LearningToRankReranker(args.db)
    
    if args.update_model:
        logger.info("Updating LTR model...")
        reranker.update_model()
        logger.info("Model update complete")
    
    if args.stats:
        stats = reranker.get_model_stats()
        print(json.dumps(stats, indent=2))
    
    if args.test_rerank:
        # This would need integration with the actual search system
        logger.info(f"Test reranking functionality would be implemented here for query: {args.test_rerank}")


if __name__ == "__main__":
    main()