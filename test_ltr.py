#!/usr/bin/env python3
"""Test script for Learning-to-Rank functionality in DocFoundry

This script tests the LTR integration with the search system,
including click feedback logging and model updates.
"""

import sys
import json
import time
import uuid
import logging
from pathlib import Path

# Add indexer to path
sys.path.append(str(Path(__file__).parent / "indexer"))

from indexer.embeddings import EmbeddingManager
from indexer.learning_to_rank import LearningToRankReranker, ClickEvent
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ltr_integration():
    """Test learning-to-rank integration with search system."""
    
    db_path = "data/docfoundry.db"
    
    logger.info("Testing Learning-to-Rank Integration")
    logger.info("=" * 50)
    
    try:
        # Initialize components
        logger.info("1. Initializing EmbeddingManager with LTR enabled...")
        embedding_manager = EmbeddingManager(db_path, enable_ltr=True)
        
        logger.info("2. Initializing LTR Reranker...")
        ltr_reranker = LearningToRankReranker(db_path)
        
        # Test search without LTR feedback
        test_query = "python documentation"
        logger.info(f"3. Testing search for: '{test_query}'")
        
        results = embedding_manager.hybrid_search(test_query, limit=5)
        logger.info(f"   Found {len(results)} results")
        
        if results:
            logger.info("   Top 3 results:")
            for i, (chunk, score) in enumerate(results[:3]):
                logger.info(f"   {i+1}. Score: {score:.4f} - {chunk.get('title', 'No title')[:50]}...")
        
        # Simulate click feedback
        logger.info("4. Simulating click feedback...")
        session_id = str(uuid.uuid4())
        
        if results:
            # Simulate clicks on different positions
            click_scenarios = [
                (0, 5.2),  # Click on first result, good dwell time
                (2, 1.8),  # Click on third result, short dwell time
                (1, 8.5),  # Click on second result, excellent dwell time
            ]
            
            for position, dwell_time in click_scenarios:
                if position < len(results):
                    chunk_id = results[position][0]['id']
                    ltr_reranker.log_click_feedback(
                        query=test_query,
                        clicked_chunk_id=chunk_id,
                        position=position,
                        session_id=session_id,
                        dwell_time=dwell_time
                    )
                    logger.info(f"   Logged click: position={position}, dwell_time={dwell_time}s")
        
        # Test click statistics
        logger.info("5. Testing click statistics...")
        click_stats = ltr_reranker.click_logger.get_click_stats(query=test_query)
        logger.info(f"   Click stats: {json.dumps(click_stats, indent=2)}")
        
        # Test model statistics
        logger.info("6. Testing model statistics...")
        model_stats = ltr_reranker.get_model_stats()
        logger.info(f"   Model weights: {model_stats['weights'][:5]}... (showing first 5)")
        
        # Test model update
        logger.info("7. Testing model update...")
        ltr_reranker.update_model(learning_rate=0.05)
        logger.info("   Model updated successfully")
        
        # Test search with updated model
        logger.info("8. Testing search with updated LTR model...")
        updated_results = embedding_manager.hybrid_search(test_query, limit=5)
        
        if updated_results:
            logger.info("   Updated top 3 results:")
            for i, (chunk, score) in enumerate(updated_results[:3]):
                logger.info(f"   {i+1}. Score: {score:.4f} - {chunk.get('title', 'No title')[:50]}...")
        
        # Compare results
        if results and updated_results:
            logger.info("9. Comparing original vs LTR-enhanced results...")
            original_ids = [chunk[0]['id'] for chunk in results[:3]]
            updated_ids = [chunk[0]['id'] for chunk in updated_results[:3]]
            
            if original_ids != updated_ids:
                logger.info("   ✓ LTR reranking changed result order")
            else:
                logger.info("   → LTR reranking maintained same order (may need more feedback data)")
        
        logger.info("\n✅ Learning-to-Rank integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ LTR integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_click_logging():
    """Test click event logging functionality."""
    
    logger.info("\nTesting Click Event Logging")
    logger.info("=" * 30)
    
    try:
        db_path = "data/docfoundry.db"
        ltr_reranker = LearningToRankReranker(db_path)
        
        # Create test click events
        test_events = [
            {
                "query": "machine learning",
                "clicked_chunk_id": "test_chunk_1",
                "position": 0,
                "session_id": str(uuid.uuid4()),
                "dwell_time": 12.5
            },
            {
                "query": "machine learning",
                "clicked_chunk_id": "test_chunk_2",
                "position": 1,
                "session_id": str(uuid.uuid4()),
                "dwell_time": 3.2
            },
            {
                "query": "deep learning",
                "clicked_chunk_id": "test_chunk_3",
                "position": 0,
                "session_id": str(uuid.uuid4()),
                "dwell_time": 25.8
            }
        ]
        
        # Log test events
        for i, event_data in enumerate(test_events, 1):
            ltr_reranker.log_click_feedback(**event_data)
            logger.info(f"   {i}. Logged click for query: '{event_data['query']}'")
        
        # Test statistics retrieval
        overall_stats = ltr_reranker.click_logger.get_click_stats()
        ml_stats = ltr_reranker.click_logger.get_click_stats(query="machine learning")
        
        logger.info(f"   Overall stats: {json.dumps(overall_stats, indent=2)}")
        logger.info(f"   ML query stats: {json.dumps(ml_stats, indent=2)}")
        
        logger.info("✅ Click logging test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Click logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test feature extraction for learning-to-rank."""
    
    logger.info("\nTesting Feature Extraction")
    logger.info("=" * 28)
    
    try:
        db_path = "data/docfoundry.db"
        ltr_reranker = LearningToRankReranker(db_path)
        
        # Test feature extraction
        test_query = "python programming"
        test_chunk = {
            'id': 'test_chunk_feature',
            'text': 'This is a comprehensive guide to Python programming language features and best practices.',
            'title': 'Python Programming Guide',
            'heading': 'Introduction to Python'
        }
        
        features = ltr_reranker.feature_extractor.extract_features(
            query=test_query,
            chunk_data=test_chunk,
            original_rank=2,
            bm25_score=0.85,
            semantic_similarity=0.72,
            rrf_score=0.78
        )
        
        logger.info("   Extracted features:")
        logger.info(f"   - Query length: {features.query_length}")
        logger.info(f"   - Query terms: {features.query_terms}")
        logger.info(f"   - Doc length: {features.doc_length}")
        logger.info(f"   - Title match: {features.title_match}")
        logger.info(f"   - Heading match: {features.heading_match}")
        logger.info(f"   - BM25 score: {features.bm25_score}")
        logger.info(f"   - Semantic similarity: {features.semantic_similarity}")
        logger.info(f"   - RRF score: {features.rrf_score}")
        logger.info(f"   - Original rank: {features.original_rank}")
        logger.info(f"   - Click rate: {features.click_rate}")
        logger.info(f"   - Avg dwell time: {features.avg_dwell_time}")
        
        # Test feature array conversion
        feature_array = features.to_array()
        logger.info(f"   Feature array shape: {feature_array.shape}")
        logger.info(f"   Feature array: {feature_array}")
        
        logger.info("✅ Feature extraction test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all LTR tests."""
    
    logger.info("DocFoundry Learning-to-Rank Test Suite")
    logger.info("=" * 40)
    
    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("Click Logging", test_click_logging),
        ("LTR Integration", test_ltr_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        start_time = time.time()
        
        try:
            success = test_func()
            duration = time.time() - start_time
            results.append((test_name, success, duration))
            
            if success:
                logger.info(f"✅ {test_name} test passed ({duration:.2f}s)")
            else:
                logger.error(f"❌ {test_name} test failed ({duration:.2f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            results.append((test_name, False, duration))
            logger.error(f"💥 {test_name} test crashed: {e} ({duration:.2f}s)")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name} ({duration:.2f}s)")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Learning-to-Rank system is working correctly.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)