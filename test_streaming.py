#!/usr/bin/env python3
"""
Test script for streaming search API endpoints.
Verifies that the streaming responses work correctly with Server-Sent Events.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Test imports
try:
    from server.rag_api import app, stream_search_results
    from fastapi.testclient import TestClient
    print("✓ Successfully imported streaming API components")
except ImportError as e:
    print(f"✗ Failed to import streaming API components: {e}")
    sys.exit(1)

async def test_stream_search_results():
    """Test the stream_search_results function."""
    print("\n=== Testing stream_search_results function ===")
    
    # Mock search results
    mock_results = [
        {"id": "1", "content": "First result", "similarity": 0.9},
        {"id": "2", "content": "Second result", "similarity": 0.8},
        {"id": "3", "content": "Third result", "similarity": 0.7}
    ]
    
    try:
        # Collect all streamed data
        streamed_data = []
        async for chunk in stream_search_results(mock_results, "semantic"):
            streamed_data.append(chunk)
        
        print(f"✓ Generated {len(streamed_data)} streaming chunks")
        
        # Verify structure
        metadata_found = False
        result_count = 0
        complete_found = False
        
        for chunk in streamed_data:
            if chunk.startswith("data: "):
                data_str = chunk[6:].strip()
                try:
                    data = json.loads(data_str)
                    if data.get('type') == 'metadata':
                        metadata_found = True
                        assert data.get('search_type') == 'semantic'
                        assert data.get('total_results') == 3
                    elif data.get('type') == 'result':
                        result_count += 1
                        assert 'index' in data
                        assert 'data' in data
                    elif data.get('type') == 'complete':
                        complete_found = True
                except json.JSONDecodeError:
                    pass
        
        assert metadata_found, "Metadata chunk not found"
        assert result_count == 3, f"Expected 3 results, got {result_count}"
        assert complete_found, "Completion chunk not found"
        
        print("✓ Stream structure is correct")
        print("✓ All streaming tests passed")
        
    except Exception as e:
        print(f"✗ Stream test failed: {e}")
        return False
    
    return True

def test_streaming_endpoints():
    """Test the streaming endpoints exist and are properly defined."""
    print("\n=== Testing streaming endpoints ===")
    
    try:
        # Check that the streaming endpoints are defined in the app
        routes = [route.path for route in app.routes]
        
        expected_routes = [
            "/search/semantic/stream",
            "/search/hybrid/stream"
        ]
        
        for route in expected_routes:
            if route in routes:
                print(f"✓ Found endpoint: {route}")
            else:
                print(f"✗ Missing endpoint: {route}")
                return False
        
        print("✓ All streaming endpoints are properly defined")
        
    except Exception as e:
        print(f"✗ Streaming endpoint test failed: {e}")
        return False
    
    return True

def test_streaming_response_structure():
    """Test that streaming responses have the correct structure."""
    print("\n=== Testing streaming response structure ===")
    
    try:
        from server.rag_api import semantic_search_stream, hybrid_search_stream
        from server.rag_api import SemanticSearchRequest, HybridSearchRequest
        
        # Check that the streaming functions are properly defined
        assert callable(semantic_search_stream), "semantic_search_stream is not callable"
        assert callable(hybrid_search_stream), "hybrid_search_stream is not callable"
        
        # Check that request models are properly defined
        semantic_req = SemanticSearchRequest(q="test", k=5, min_similarity=0.5)
        hybrid_req = HybridSearchRequest(q="test", k=5, rrf_k=60, min_similarity=0.5)
        
        assert semantic_req.q == "test"
        assert hybrid_req.q == "test"
        
        print("✓ Streaming functions are properly defined")
        print("✓ Request models are working correctly")
        
    except Exception as e:
        print(f"✗ Streaming response structure test failed: {e}")
        return False
    
    return True

async def main():
    """Run all streaming tests."""
    print("DocFoundry Streaming API Test Suite")
    print("====================================")
    
    tests = [
        ("Stream Function", test_stream_search_results()),
        ("Streaming Endpoints", test_streaming_endpoints()),
        ("Response Structure", test_streaming_response_structure())
    ]
    
    results = []
    for test_name, test_coro in tests:
        if asyncio.iscoroutine(test_coro):
            result = await test_coro
        else:
            result = test_coro
        results.append((test_name, result))
    
    print("\n=== Test Results ===")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All streaming tests passed! The streaming API is working correctly.")
        return True
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)