#!/usr/bin/env python3
"""
Seed minimal test data for DocFoundry performance testing.
Creates a small dataset suitable for smoke tests and basic functionality validation.
"""

import os
import sys
import asyncio
from datetime import datetime
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import get_database
from indexer.embeddings import get_embedding_model
from indexer.postgres_adapter import PostgresAdapter


class TestDataSeeder:
    """Minimal test data seeder for smoke tests."""
    
    def __init__(self):
        self.db = None
        self.adapter = None
        self.embedding_model = None
    
    async def initialize(self):
        """Initialize database connections and models."""
        self.db = await get_database()
        self.adapter = PostgresAdapter(self.db)
        self.embedding_model = get_embedding_model()
        
        # Ensure tables exist
        await self.adapter.create_tables()
    
    async def seed_minimal_data(self):
        """Seed minimal test data (5 documents, 50 chunks)."""
        print("🌱 Seeding minimal test data...")
        
        # Sample documents with realistic content
        documents = [
            {
                'title': 'Machine Learning Fundamentals',
                'url': 'https://example.com/ml-fundamentals',
                'site': 'example.com',
                'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns.'
            },
            {
                'title': 'Database Optimization Guide',
                'url': 'https://docs.example.com/db-optimization',
                'site': 'docs.example.com',
                'content': 'Database optimization involves improving query performance, indexing strategies, and data structure design. Key techniques include proper indexing, query optimization, normalization, and caching strategies to reduce response times.'
            },
            {
                'title': 'API Security Best Practices',
                'url': 'https://blog.example.com/api-security',
                'site': 'blog.example.com',
                'content': 'API security requires implementing authentication, authorization, input validation, rate limiting, and encryption. Common vulnerabilities include injection attacks, broken authentication, and excessive data exposure.'
            },
            {
                'title': 'Microservices Architecture Patterns',
                'url': 'https://example.com/microservices',
                'site': 'example.com',
                'content': 'Microservices architecture breaks down applications into small, independent services that communicate over well-defined APIs. Benefits include scalability, technology diversity, and fault isolation.'
            },
            {
                'title': 'Document Processing Workflows',
                'url': 'https://docs.example.com/document-processing',
                'site': 'docs.example.com',
                'content': 'Document processing involves extracting, transforming, and analyzing text from various document formats. Modern approaches use natural language processing, OCR, and machine learning for automated content analysis.'
            }
        ]
        
        # Insert documents and create chunks
        for i, doc in enumerate(documents):
            print(f"  📄 Processing document {i+1}/5: {doc['title']}")
            
            # Create document chunks (10 chunks per document)
            chunks = self._create_chunks(doc['content'], doc['title'])
            
            for j, chunk_text in enumerate(chunks):
                # Generate embedding
                embedding = await self.embedding_model.encode(chunk_text)
                
                # Store chunk
                await self.adapter.store_chunk(
                    chunk_id=f"doc_{i+1}_chunk_{j+1}",
                    text=chunk_text,
                    embedding=embedding,
                    metadata={
                        'title': doc['title'],
                        'url': doc['url'],
                        'site': doc['site'],
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'created_at': datetime.utcnow().isoformat()
                    }
                )
        
        print("✅ Minimal test data seeded successfully")
        print(f"   📊 Total: 5 documents, 50 chunks")
    
    def _create_chunks(self, content: str, title: str) -> List[str]:
        """Create 10 chunks from document content."""
        # Simple chunking strategy for test data
        words = content.split()
        chunk_size = max(10, len(words) // 10)  # Aim for 10 chunks
        
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = f"{title}: {' '.join(chunk_words)}"
            chunks.append(chunk_text)
        
        # Ensure we have exactly 10 chunks
        while len(chunks) < 10:
            chunks.append(f"{title}: Additional context chunk {len(chunks) + 1}")
        
        return chunks[:10]
    
    async def cleanup(self):
        """Clean up database connections."""
        if self.db:
            await self.db.close()


async def main():
    """Main seeding function."""
    print("🚀 Starting minimal test data seeding...")
    
    seeder = TestDataSeeder()
    try:
        await seeder.initialize()
        await seeder.seed_minimal_data()
        print("🎉 Test data seeding completed successfully!")
    except Exception as e:
        print(f"❌ Error during seeding: {e}")
        sys.exit(1)
    finally:
        await seeder.cleanup()


if __name__ == "__main__":
    asyncio.run(main())