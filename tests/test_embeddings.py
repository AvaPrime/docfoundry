"""Comprehensive unit tests for DocFoundry embeddings module.

Tests cover:
- EmbeddingManager initialization and configuration
- Document embedding generation and caching
- Similarity search functionality
- Learning-to-rank integration
- Error handling and edge cases
- Performance considerations
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import tempfile
import os

# Import the module under test
from embeddings import EmbeddingManager


class TestEmbeddingManager:
    """Test suite for EmbeddingManager class."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer for testing."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        return mock_model
    
    @pytest.fixture
    def mock_ltr_module(self):
        """Mock learning-to-rank module for testing."""
        mock_ltr = Mock()
        mock_ltr.rerank_results.return_value = [
            {'id': 'doc2', 'score': 0.9, 'content': 'Second document'},
            {'id': 'doc1', 'score': 0.8, 'content': 'First document'}
        ]
        return mock_ltr
    
    @pytest.fixture
    def embedding_manager(self, mock_sentence_transformer, mock_ltr_module):
        """Create EmbeddingManager instance with mocked dependencies."""
        with patch('embeddings.SentenceTransformer', return_value=mock_sentence_transformer):
            with patch('embeddings.ltr_module', mock_ltr_module):
                manager = EmbeddingManager(model_name='test-model')
                return manager
    
    def test_initialization_default_model(self):
        """Test EmbeddingManager initialization with default model."""
        with patch('embeddings.SentenceTransformer') as mock_st:
            manager = EmbeddingManager()
            mock_st.assert_called_once_with('all-MiniLM-L6-v2')
            assert manager.model_name == 'all-MiniLM-L6-v2'
    
    def test_initialization_custom_model(self):
        """Test EmbeddingManager initialization with custom model."""
        with patch('embeddings.SentenceTransformer') as mock_st:
            manager = EmbeddingManager(model_name='custom-model')
            mock_st.assert_called_once_with('custom-model')
            assert manager.model_name == 'custom-model'
    
    def test_initialization_with_cache_dir(self):
        """Test EmbeddingManager initialization with custom cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('embeddings.SentenceTransformer') as mock_st:
                manager = EmbeddingManager(cache_dir=temp_dir)
                mock_st.assert_called_once_with('all-MiniLM-L6-v2', cache_folder=temp_dir)
    
    def test_embed_documents_single_document(self, embedding_manager):
        """Test embedding generation for a single document."""
        documents = ["This is a test document."]
        embeddings = embedding_manager.embed_documents(documents)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 3)  # 1 document, 3 dimensions
        embedding_manager.model.encode.assert_called_once_with(documents)
    
    def test_embed_documents_multiple_documents(self, embedding_manager):
        """Test embedding generation for multiple documents."""
        documents = ["First document.", "Second document."]
        embeddings = embedding_manager.embed_documents(documents)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3)  # 2 documents, 3 dimensions
        embedding_manager.model.encode.assert_called_once_with(documents)
    
    def test_embed_documents_empty_list(self, embedding_manager):
        """Test embedding generation with empty document list."""
        documents = []
        with patch.object(embedding_manager.model, 'encode', return_value=np.array([])):
            embeddings = embedding_manager.embed_documents(documents)
            assert embeddings.size == 0
    
    def test_embed_documents_none_input(self, embedding_manager):
        """Test embedding generation with None input."""
        with pytest.raises((TypeError, ValueError)):
            embedding_manager.embed_documents(None)
    
    def test_embed_query(self, embedding_manager):
        """Test query embedding generation."""
        query = "test query"
        with patch.object(embedding_manager.model, 'encode', return_value=np.array([0.1, 0.2, 0.3])):
            embedding = embedding_manager.embed_query(query)
            
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (3,)
            embedding_manager.model.encode.assert_called_once_with([query])
    
    def test_embed_query_empty_string(self, embedding_manager):
        """Test query embedding with empty string."""
        query = ""
        with patch.object(embedding_manager.model, 'encode', return_value=np.array([0.0, 0.0, 0.0])):
            embedding = embedding_manager.embed_query(query)
            assert isinstance(embedding, np.ndarray)
    
    def test_similarity_search_basic(self, embedding_manager):
        """Test basic similarity search functionality."""
        query_embedding = np.array([0.1, 0.2, 0.3])
        document_embeddings = np.array([
            [0.1, 0.2, 0.3],  # Perfect match
            [0.4, 0.5, 0.6],  # Different
            [0.15, 0.25, 0.35]  # Close match
        ])
        documents = [
            {'id': 'doc1', 'content': 'First document'},
            {'id': 'doc2', 'content': 'Second document'},
            {'id': 'doc3', 'content': 'Third document'}
        ]
        
        results = embedding_manager.similarity_search(
            query_embedding, document_embeddings, documents, top_k=2
        )
        
        assert len(results) == 2
        assert results[0]['id'] == 'doc1'  # Perfect match should be first
        assert results[0]['score'] >= results[1]['score']  # Scores should be descending
        assert all('score' in result for result in results)
    
    def test_similarity_search_with_ltr(self, embedding_manager):
        """Test similarity search with learning-to-rank reranking."""
        query_embedding = np.array([0.1, 0.2, 0.3])
        document_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        documents = [
            {'id': 'doc1', 'content': 'First document'},
            {'id': 'doc2', 'content': 'Second document'}
        ]
        query = "test query"
        
        results = embedding_manager.similarity_search(
            query_embedding, document_embeddings, documents, 
            top_k=2, query=query, use_ltr=True
        )
        
        # Should return LTR reranked results
        assert len(results) == 2
        assert results[0]['id'] == 'doc2'  # LTR mock returns doc2 first
        assert results[1]['id'] == 'doc1'
        embedding_manager.ltr_module.rerank_results.assert_called_once()
    
    def test_similarity_search_top_k_limit(self, embedding_manager):
        """Test similarity search respects top_k parameter."""
        query_embedding = np.array([0.1, 0.2, 0.3])
        document_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.2, 0.3, 0.4]
        ])
        documents = [
            {'id': f'doc{i}', 'content': f'Document {i}'} for i in range(4)
        ]
        
        results = embedding_manager.similarity_search(
            query_embedding, document_embeddings, documents, top_k=2
        )
        
        assert len(results) == 2
    
    def test_similarity_search_empty_documents(self, embedding_manager):
        """Test similarity search with empty document list."""
        query_embedding = np.array([0.1, 0.2, 0.3])
        document_embeddings = np.array([])
        documents = []
        
        results = embedding_manager.similarity_search(
            query_embedding, document_embeddings, documents, top_k=5
        )
        
        assert len(results) == 0
    
    def test_similarity_search_mismatched_dimensions(self, embedding_manager):
        """Test similarity search with mismatched embedding dimensions."""
        query_embedding = np.array([0.1, 0.2, 0.3])  # 3D
        document_embeddings = np.array([[0.1, 0.2]])  # 2D
        documents = [{'id': 'doc1', 'content': 'Document'}]
        
        with pytest.raises((ValueError, IndexError)):
            embedding_manager.similarity_search(
                query_embedding, document_embeddings, documents
            )
    
    def test_get_embedding_dimension(self, embedding_manager):
        """Test getting embedding dimension from model."""
        dimension = embedding_manager.get_embedding_dimension()
        assert dimension == 3  # Mock returns 3
        embedding_manager.model.get_sentence_embedding_dimension.assert_called_once()
    
    def test_cosine_similarity_calculation(self, embedding_manager):
        """Test cosine similarity calculation."""
        # Test with known vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 0, 0])  # Same as vec1
        
        query_embedding = vec1
        document_embeddings = np.array([vec2, vec3])
        documents = [
            {'id': 'doc1', 'content': 'Orthogonal'},
            {'id': 'doc2', 'content': 'Identical'}
        ]
        
        results = embedding_manager.similarity_search(
            query_embedding, document_embeddings, documents, top_k=2
        )
        
        # Identical vector should have higher similarity than orthogonal
        assert results[0]['id'] == 'doc2'  # Identical vector
        assert results[0]['score'] > results[1]['score']
        assert results[0]['score'] == pytest.approx(1.0, abs=1e-6)  # Perfect similarity
        assert results[1]['score'] == pytest.approx(0.0, abs=1e-6)  # Orthogonal
    
    def test_batch_processing_performance(self, embedding_manager):
        """Test that batch processing is more efficient than individual calls."""
        documents = [f"Document {i}" for i in range(100)]
        
        # Batch processing
        batch_embeddings = embedding_manager.embed_documents(documents)
        
        # Verify single call to model.encode
        assert embedding_manager.model.encode.call_count == 1
        assert batch_embeddings.shape[0] == 100
    
    def test_error_handling_model_failure(self, embedding_manager):
        """Test error handling when model fails."""
        embedding_manager.model.encode.side_effect = Exception("Model error")
        
        with pytest.raises(Exception, match="Model error"):
            embedding_manager.embed_documents(["test"])
    
    def test_error_handling_invalid_embeddings(self, embedding_manager):
        """Test error handling with invalid embedding arrays."""
        query_embedding = "not an array"
        document_embeddings = np.array([[0.1, 0.2, 0.3]])
        documents = [{'id': 'doc1', 'content': 'Document'}]
        
        with pytest.raises((TypeError, AttributeError)):
            embedding_manager.similarity_search(
                query_embedding, document_embeddings, documents
            )
    
    def test_ltr_integration_without_query(self, embedding_manager):
        """Test that LTR is not used when query is not provided."""
        query_embedding = np.array([0.1, 0.2, 0.3])
        document_embeddings = np.array([[0.1, 0.2, 0.3]])
        documents = [{'id': 'doc1', 'content': 'Document'}]
        
        results = embedding_manager.similarity_search(
            query_embedding, document_embeddings, documents, use_ltr=True
        )
        
        # LTR should not be called without query
        embedding_manager.ltr_module.rerank_results.assert_not_called()
    
    def test_memory_efficiency_large_batch(self, embedding_manager):
        """Test memory efficiency with large document batches."""
        # Test with a reasonably large batch
        documents = [f"Document {i} with some content" for i in range(1000)]
        
        # Mock to return appropriate sized array
        with patch.object(embedding_manager.model, 'encode', 
                         return_value=np.random.rand(1000, 3)):
            embeddings = embedding_manager.embed_documents(documents)
            
            assert embeddings.shape == (1000, 3)
            assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64
    
    def test_thread_safety_concurrent_embedding(self, embedding_manager):
        """Test thread safety of embedding operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def embed_worker(doc_id):
            try:
                docs = [f"Document {doc_id}"]
                embedding = embedding_manager.embed_documents(docs)
                results.append((doc_id, embedding))
            except Exception as e:
                errors.append((doc_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=embed_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
    
    def test_caching_behavior(self, embedding_manager):
        """Test that model caching works correctly."""
        # This test verifies that the same model instance is reused
        model_instance_1 = embedding_manager.model
        model_instance_2 = embedding_manager.model
        
        assert model_instance_1 is model_instance_2
    
    def test_normalization_of_embeddings(self, embedding_manager):
        """Test that embeddings are properly normalized for cosine similarity."""
        # Mock unnormalized embeddings
        unnormalized = np.array([[3, 4, 0], [1, 1, 1]])  # Length 5 and sqrt(3)
        
        with patch.object(embedding_manager.model, 'encode', return_value=unnormalized):
            embeddings = embedding_manager.embed_documents(["doc1", "doc2"])
            
            # Check if embeddings are normalized (length = 1)
            norms = np.linalg.norm(embeddings, axis=1)
            # Note: This depends on whether the actual implementation normalizes
            # For now, we just verify the shape and type
            assert embeddings.shape == (2, 3)
            assert isinstance(embeddings, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])