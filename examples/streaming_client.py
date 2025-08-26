#!/usr/bin/env python3
"""
Example client for DocFoundry streaming search API.
Demonstrates how to consume Server-Sent Events from the streaming endpoints.
"""

import asyncio
import aiohttp
import json
import sys
from typing import AsyncGenerator

class StreamingSearchClient:
    """Client for consuming DocFoundry streaming search results."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
    
    async def semantic_search_stream(self, query: str, k: int = 5, min_similarity: float = 0.3) -> AsyncGenerator[dict, None]:
        """Stream semantic search results."""
        url = f"{self.base_url}/search/semantic/stream"
        payload = {
            "q": query,
            "k": k,
            "min_similarity": min_similarity
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Request failed with status {response.status}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            yield data
                        except json.JSONDecodeError:
                            continue
    
    async def hybrid_search_stream(self, query: str, k: int = 5, rrf_k: int = 60, min_similarity: float = 0.3) -> AsyncGenerator[dict, None]:
        """Stream hybrid search results."""
        url = f"{self.base_url}/search/hybrid/stream"
        payload = {
            "q": query,
            "k": k,
            "rrf_k": rrf_k,
            "min_similarity": min_similarity
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Request failed with status {response.status}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            yield data
                        except json.JSONDecodeError:
                            continue

async def demo_semantic_search():
    """Demonstrate semantic search streaming."""
    print("=== Semantic Search Stream Demo ===")
    
    client = StreamingSearchClient()
    query = "machine learning algorithms"
    
    print(f"Searching for: '{query}'")
    print("Streaming results...\n")
    
    try:
        result_count = 0
        async for event in client.semantic_search_stream(query, k=10):
            if event['type'] == 'metadata':
                print(f"📊 Search Type: {event['search_type']}")
                print(f"📊 Total Results: {event['total_results']}\n")
            elif event['type'] == 'result':
                result_count += 1
                result = event['data']
                print(f"Result #{event['index'] + 1}:")
                print(f"  📄 Content: {result.get('content', 'N/A')[:100]}...")
                print(f"  🎯 Similarity: {result.get('similarity', 'N/A')}")
                print(f"  🔗 Source: {result.get('source', 'N/A')}\n")
            elif event['type'] == 'complete':
                print(f"✅ {event['message']}")
                print(f"📈 Processed {result_count} results")
                break
    except Exception as e:
        print(f"❌ Error: {e}")

async def demo_hybrid_search():
    """Demonstrate hybrid search streaming."""
    print("\n=== Hybrid Search Stream Demo ===")
    
    client = StreamingSearchClient()
    query = "natural language processing"
    
    print(f"Searching for: '{query}'")
    print("Streaming results...\n")
    
    try:
        result_count = 0
        async for event in client.hybrid_search_stream(query, k=5, rrf_k=60):
            if event['type'] == 'metadata':
                print(f"📊 Search Type: {event['search_type']}")
                print(f"📊 Total Results: {event['total_results']}\n")
            elif event['type'] == 'result':
                result_count += 1
                result = event['data']
                print(f"Result #{event['index'] + 1}:")
                print(f"  📄 Content: {result.get('content', 'N/A')[:100]}...")
                print(f"  🎯 Similarity: {result.get('similarity', 'N/A')}")
                print(f"  🔗 Source: {result.get('source', 'N/A')}\n")
            elif event['type'] == 'complete':
                print(f"✅ {event['message']}")
                print(f"📈 Processed {result_count} results")
                break
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    """Run streaming search demos."""
    print("DocFoundry Streaming Search Client Demo")
    print("=======================================")
    print("\n⚠️  Make sure DocFoundry server is running on http://localhost:8001")
    print("   Start with: python -m uvicorn server.rag_api:app --host 0.0.0.0 --port 8001 --reload\n")
    
    # Run demos
    await demo_semantic_search()
    await demo_hybrid_search()
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)