# DocFoundry Embeddings Module
# Handles semantic search using sentence transformers

import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import pickle
import logging
from pathlib import Path

# Import learning-to-rank module
try:
    from .learning_to_rank import LearningToRankReranker
except ImportError:
    # Fallback for when running as standalone
    try:
        from learning_to_rank import LearningToRankReranker
    except ImportError:
        LearningToRankReranker = None
        logging.warning("Learning-to-rank module not available")

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages document embeddings for semantic search"""
    
    def __init__(self, db_path: str, model_name: str = "all-MiniLM-L6-v2", enable_ltr: bool = True):
        """
        Initialize embedding manager
        
        Args:
            db_path: Path to SQLite database
            model_name: Sentence transformer model name
            enable_ltr: Enable learning-to-rank reranking
        """
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.enable_ltr = enable_ltr and LearningToRankReranker is not None
        
        # Initialize learning-to-rank reranker if available
        self.ltr_reranker = None
        if self.enable_ltr:
            try:
                self.ltr_reranker = LearningToRankReranker(db_path)
                logger.info("Learning-to-rank reranker initialized")
            except Exception as e:
                logger.warning(f"Could not initialize LTR reranker: {e}")
                self.enable_ltr = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Clean and prepare text
        text = text.strip()
        if not text:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts efficiently"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Clean texts
        cleaned_texts = [text.strip() if text else "" for text in texts]
        
        # Generate embeddings in batch for efficiency
        embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True, show_progress_bar=True)
        return [emb for emb in embeddings]
    
    def serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize embedding for database storage"""
        return pickle.dumps(embedding.astype(np.float32))  # Use float32 to save space
    
    def deserialize_embedding(self, data: bytes) -> np.ndarray:
        """Deserialize embedding from database"""
        return pickle.loads(data)
    
    def update_chunk_embeddings(self, chunk_ids: Optional[List[int]] = None):
        """Update embeddings for chunks in the database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Get chunks that need embedding updates
            if chunk_ids:
                placeholders = ','.join('?' * len(chunk_ids))
                query = f"SELECT id, text FROM chunks WHERE id IN ({placeholders}) AND (embedding IS NULL OR embedding = '')"
                chunks = conn.execute(query, chunk_ids).fetchall()
            else:
                chunks = conn.execute("SELECT id, text FROM chunks WHERE embedding IS NULL OR embedding = ''").fetchall()
            
            if not chunks:
                logger.info("No chunks need embedding updates")
                return
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Extract texts and generate embeddings in batch
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings_batch(texts)
            
            # Update database with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                serialized_embedding = self.serialize_embedding(embedding)
                conn.execute(
                    "UPDATE chunks SET embedding = ? WHERE id = ?",
                    (serialized_embedding, chunk['id'])
                )
            
            conn.commit()
            logger.info(f"Successfully updated embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error updating embeddings: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def semantic_search(self, query: str, limit: int = 10, min_similarity: float = 0.3) -> List[Tuple[dict, float]]:
        """Perform semantic search using embeddings"""
        if not query.strip():
            return []
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Get all chunks with embeddings
            chunks = conn.execute("""
                SELECT c.id, c.text, c.heading, c.anchor, c.embedding,
                       d.path, d.title, d.source_url
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL AND c.embedding != ''
            """).fetchall()
            
            if not chunks:
                logger.warning("No chunks with embeddings found")
                return []
            
            # Calculate similarities
            results = []
            for chunk in chunks:
                try:
                    chunk_embedding = self.deserialize_embedding(chunk['embedding'])
                    similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                    
                    if similarity >= min_similarity:
                        chunk_data = {
                            'id': chunk['id'],
                            'text': chunk['text'],
                            'heading': chunk['heading'],
                            'anchor': chunk['anchor'],
                            'path': chunk['path'],
                            'title': chunk['title'],
                            'source_url': chunk['source_url']
                        }
                        results.append((chunk_data, similarity))
                except Exception as e:
                    logger.warning(f"Error processing chunk {chunk['id']}: {e}")
                    continue
            
            # Sort by similarity (descending) and limit results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
        finally:
            conn.close()
    
    def hybrid_search(self, query: str, limit: int = 10, k: int = 60, min_similarity: float = 0.3) -> List[Tuple[dict, float]]:
        """Combine FTS and semantic search using Reciprocal Rank Fusion (RRF)
        
        Args:
            query: Search query
            limit: Number of results to return
            k: RRF parameter (typically 60)
            min_similarity: Minimum similarity threshold for semantic results
        """
        if not query.strip():
            return []
        
        # Get semantic search results with rankings
        semantic_results = self.semantic_search(query, limit * 3, min_similarity)
        semantic_rankings = {result[0]['id']: rank + 1 for rank, result in enumerate(semantic_results)}
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Get FTS search results with rankings
            fts_results = []
            fts_rankings = {}
            try:
                fts_chunks = conn.execute("""
                    SELECT c.id, c.text, c.heading, c.anchor,
                           d.path, d.title, d.source_url,
                           bm25(chunks_fts) as bm25_score
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    JOIN documents d ON d.id = c.document_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25_score
                    LIMIT ?
                """, (query, limit * 3)).fetchall()
                
                for rank, chunk in enumerate(fts_chunks):
                    chunk_data = {
                        'id': chunk['id'],
                        'text': chunk['text'],
                        'heading': chunk['heading'],
                        'anchor': chunk['anchor'],
                        'path': chunk['path'],
                        'title': chunk['title'],
                        'source_url': chunk['source_url']
                    }
                    fts_results.append((chunk_data, chunk['bm25_score']))
                    fts_rankings[chunk['id']] = rank + 1
                    
            except Exception as e:
                logger.warning(f"FTS search failed, falling back to LIKE search: {e}")
                # Fallback to LIKE search if FTS fails
                like_chunks = conn.execute("""
                    SELECT c.id, c.text, c.heading, c.anchor,
                           d.path, d.title, d.source_url
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.text LIKE ? OR c.heading LIKE ?
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit * 2)).fetchall()
                
                for rank, chunk in enumerate(like_chunks):
                    chunk_data = {
                        'id': chunk['id'],
                        'text': chunk['text'],
                        'heading': chunk['heading'],
                        'anchor': chunk['anchor'],
                        'path': chunk['path'],
                        'title': chunk['title'],
                        'source_url': chunk['source_url']
                    }
                    fts_results.append((chunk_data, 1.0))  # Uniform score for LIKE results
                    fts_rankings[chunk['id']] = rank + 1
            
            # Collect all unique chunks
            all_chunks = {}
            
            # Add semantic results
            for result, similarity in semantic_results:
                chunk_id = result['id']
                all_chunks[chunk_id] = result
            
            # Add FTS results
            for result, score in fts_results:
                chunk_id = result['id']
                if chunk_id not in all_chunks:
                    all_chunks[chunk_id] = result
            
            # Calculate RRF scores
            rrf_scores = []
            for chunk_id, chunk_data in all_chunks.items():
                rrf_score = 0.0
                
                # Add semantic ranking contribution
                if chunk_id in semantic_rankings:
                    rrf_score += 1.0 / (k + semantic_rankings[chunk_id])
                
                # Add FTS ranking contribution
                if chunk_id in fts_rankings:
                    rrf_score += 1.0 / (k + fts_rankings[chunk_id])
                
                rrf_scores.append((chunk_data, rrf_score))
            
            # Sort by RRF score (descending)
            rrf_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Apply learning-to-rank reranking if enabled
            if self.enable_ltr and self.ltr_reranker:
                try:
                    rrf_scores = self.ltr_reranker.rerank_results(query, rrf_scores)
                    logger.debug(f"Applied LTR reranking for query: {query}")
                except Exception as e:
                    logger.warning(f"LTR reranking failed, using original RRF scores: {e}")
            
            return rrf_scores[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return semantic_results[:limit]  # Fallback to semantic only
        finally:
            conn.close()

def main():
    """CLI for embedding operations"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="DocFoundry Embeddings CLI")
    parser.add_argument("--db", default="data/docfoundry.db", help="Database path")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--update", action="store_true", help="Update all chunk embeddings")
    parser.add_argument("--search", help="Perform semantic search")
    parser.add_argument("--limit", type=int, default=5, help="Search result limit")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        manager = EmbeddingManager(args.db, args.model)
        
        if args.update:
            logger.info("Updating chunk embeddings...")
            manager.update_chunk_embeddings()
            logger.info("Embedding update complete")
        
        if args.search:
            logger.info(f"Searching for: {args.search}")
            results = manager.hybrid_search(args.search, args.limit)
            
            if results:
                print(f"\nFound {len(results)} results (RRF Hybrid Search):\n")
                for i, (chunk, score) in enumerate(results, 1):
                    print(f"{i}. {chunk['title']} (RRF Score: {score:.4f})")
                    if chunk['heading']:
                        print(f"   Section: {chunk['heading']}")
                    print(f"   Path: {chunk['path']}")
                    print(f"   Text: {chunk['text'][:200]}...")
                    print()
            else:
                print("No results found.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()