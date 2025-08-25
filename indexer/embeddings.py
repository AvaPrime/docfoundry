# DocFoundry Embeddings Module
# Handles semantic search using sentence transformers

import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages document embeddings for semantic search"""
    
    def __init__(self, db_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager
        
        Args:
            db_path: Path to SQLite database
            model_name: Sentence transformer model name
        """
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
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
    
    def hybrid_search(self, query: str, limit: int = 10, semantic_weight: float = 0.7) -> List[Tuple[dict, float]]:
        """Combine FTS and semantic search for better results"""
        if not query.strip():
            return []
        
        # Get semantic search results
        semantic_results = self.semantic_search(query, limit * 2)  # Get more for mixing
        semantic_scores = {result[0]['id']: result[1] for result in semantic_results}
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Get FTS search results
            fts_results = []
            try:
                fts_chunks = conn.execute("""
                    SELECT c.id, c.text, c.heading, c.anchor,
                           d.path, d.title, d.source_url,
                           rank
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    JOIN documents d ON d.id = c.document_id
                    WHERE chunks_fts MATCH ?
                    LIMIT ?
                """, (query, limit * 2)).fetchall()
                
                # Normalize FTS scores (rank is negative, lower is better)
                if fts_chunks:
                    max_rank = abs(min(chunk['rank'] for chunk in fts_chunks))
                    for chunk in fts_chunks:
                        normalized_score = abs(chunk['rank']) / max_rank if max_rank > 0 else 0
                        chunk_data = {
                            'id': chunk['id'],
                            'text': chunk['text'],
                            'heading': chunk['heading'],
                            'anchor': chunk['anchor'],
                            'path': chunk['path'],
                            'title': chunk['title'],
                            'source_url': chunk['source_url']
                        }
                        fts_results.append((chunk_data, normalized_score))
            except Exception as e:
                logger.warning(f"FTS search failed: {e}")
            
            # Combine results with weighted scoring
            combined_scores = {}
            all_chunk_ids = set()
            
            # Add semantic results
            for result, score in semantic_results:
                chunk_id = result['id']
                combined_scores[chunk_id] = {
                    'data': result,
                    'semantic_score': score,
                    'fts_score': 0.0
                }
                all_chunk_ids.add(chunk_id)
            
            # Add FTS results
            for result, score in fts_results:
                chunk_id = result['id']
                if chunk_id in combined_scores:
                    combined_scores[chunk_id]['fts_score'] = score
                else:
                    combined_scores[chunk_id] = {
                        'data': result,
                        'semantic_score': 0.0,
                        'fts_score': score
                    }
                    all_chunk_ids.add(chunk_id)
            
            # Calculate final weighted scores
            final_results = []
            for chunk_id, scores in combined_scores.items():
                final_score = (
                    semantic_weight * scores['semantic_score'] +
                    (1 - semantic_weight) * scores['fts_score']
                )
                final_results.append((scores['data'], final_score))
            
            # Sort by final score and limit
            final_results.sort(key=lambda x: x[1], reverse=True)
            return final_results[:limit]
            
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
                print(f"\nFound {len(results)} results:\n")
                for i, (chunk, score) in enumerate(results, 1):
                    print(f"{i}. {chunk['title']} (Score: {score:.3f})")
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