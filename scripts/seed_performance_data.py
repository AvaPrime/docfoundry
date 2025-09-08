#!/usr/bin/env python3
"""
Seed comprehensive performance test data for DocFoundry.
Creates a larger dataset (100 documents, 1000+ chunks) for stress testing.
"""

import os
import sys
import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import get_database
from indexer.embeddings import get_embedding_model
from indexer.postgres_adapter import PostgresAdapter


class PerformanceDataSeeder:
    """Comprehensive performance data seeder for stress testing."""
    
    def __init__(self):
        self.db = None
        self.adapter = None
        self.embedding_model = None
        
        # Content templates for diverse test data
        self.content_templates = {
            'technical': [
                "This document covers advanced {topic} concepts including {detail1}, {detail2}, and {detail3}. Implementation requires careful consideration of {consideration1} and {consideration2}.",
                "Understanding {topic} is crucial for modern software development. Key aspects include {detail1}, {detail2}, and best practices for {detail3}.",
                "The {topic} framework provides solutions for {detail1} and {detail2}. Performance optimization focuses on {detail3} and {consideration1}."
            ],
            'tutorial': [
                "Step-by-step guide to {topic}: First, understand {detail1}. Next, implement {detail2}. Finally, optimize for {detail3}.",
                "Learn {topic} fundamentals: {detail1} forms the foundation, {detail2} adds functionality, and {detail3} ensures scalability.",
                "Mastering {topic} requires practice with {detail1}, {detail2}, and advanced techniques like {detail3}."
            ],
            'reference': [
                "{topic} API reference: {detail1} handles core functionality, {detail2} manages configuration, {detail3} provides utilities.",
                "Complete {topic} documentation: {detail1} overview, {detail2} implementation details, {detail3} troubleshooting guide.",
                "{topic} specification defines {detail1}, {detail2}, and {detail3} for standardized implementation."
            ]
        }
        
        self.topics = [
            'machine learning', 'database optimization', 'API security', 'microservices',
            'cloud computing', 'data visualization', 'natural language processing',
            'distributed systems', 'performance monitoring', 'container orchestration',
            'serverless architecture', 'data pipelines', 'search algorithms',
            'caching strategies', 'load balancing', 'message queues', 'event sourcing',
            'CQRS patterns', 'domain driven design', 'test automation'
        ]
        
        self.details = [
            'implementation patterns', 'optimization techniques', 'security measures',
            'scalability solutions', 'monitoring strategies', 'error handling',
            'data structures', 'algorithm efficiency', 'resource management',
            'configuration options', 'deployment strategies', 'testing approaches',
            'integration methods', 'performance tuning', 'troubleshooting guides'
        ]
        
        self.sites = [
            'example.com', 'docs.example.com', 'blog.example.com', 'api.example.com',
            'support.example.com', 'learn.example.com', 'dev.example.com',
            'community.example.com', 'wiki.example.com', 'help.example.com'
        ]
    
    async def initialize(self):
        """Initialize database connections and models."""
        print("🔧 Initializing database connections...")
        self.db = await get_database()
        self.adapter = PostgresAdapter(self.db)
        self.embedding_model = get_embedding_model()
        
        # Ensure tables exist
        await self.adapter.create_tables()
        print("✅ Database initialized")
    
    async def seed_performance_data(self):
        """Seed comprehensive performance test data."""
        print("🌱 Seeding performance test data...")
        print("   Target: 100 documents, 1000+ chunks")
        
        total_chunks = 0
        batch_size = 10  # Process documents in batches
        
        for batch in range(0, 100, batch_size):
            print(f"  📦 Processing batch {batch//batch_size + 1}/10 (docs {batch+1}-{min(batch+batch_size, 100)})")
            
            batch_tasks = []
            for i in range(batch, min(batch + batch_size, 100)):
                task = self._process_document(i + 1)
                batch_tasks.append(task)
            
            # Process batch concurrently
            batch_results = await asyncio.gather(*batch_tasks)
            batch_chunks = sum(batch_results)
            total_chunks += batch_chunks
            
            print(f"    ✅ Batch completed: {batch_chunks} chunks")
        
        # Optimize database for performance
        await self._optimize_database()
        
        print("🎉 Performance test data seeded successfully!")
        print(f"   📊 Total: 100 documents, {total_chunks} chunks")
    
    async def _process_document(self, doc_num: int) -> int:
        """Process a single document and return chunk count."""
        # Generate document metadata
        topic = random.choice(self.topics)
        site = random.choice(self.sites)
        doc_type = random.choice(list(self.content_templates.keys()))
        
        doc_title = f"{topic.title()} {doc_type.title()} - Part {doc_num}"
        doc_url = f"https://{site}/{topic.replace(' ', '-')}-{doc_type}-{doc_num}"
        
        # Generate document content
        content_parts = []
        num_sections = random.randint(3, 8)  # Variable document length
        
        for section in range(num_sections):
            template = random.choice(self.content_templates[doc_type])
            section_content = template.format(
                topic=topic,
                detail1=random.choice(self.details),
                detail2=random.choice(self.details),
                detail3=random.choice(self.details),
                consideration1=random.choice(self.details),
                consideration2=random.choice(self.details)
            )
            content_parts.append(section_content)
        
        full_content = " ".join(content_parts)
        
        # Create chunks (10-15 chunks per document)
        chunks = self._create_realistic_chunks(full_content, doc_title)
        
        # Store chunks with embeddings
        for j, chunk_text in enumerate(chunks):
            # Generate realistic embedding
            embedding = await self.embedding_model.encode(chunk_text)
            
            # Create realistic metadata
            created_date = datetime.utcnow() - timedelta(
                days=random.randint(1, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            await self.adapter.store_chunk(
                chunk_id=f"perf_doc_{doc_num}_chunk_{j+1}",
                text=chunk_text,
                embedding=embedding,
                metadata={
                    'title': doc_title,
                    'url': doc_url,
                    'site': site,
                    'topic': topic,
                    'doc_type': doc_type,
                    'chunk_index': j,
                    'total_chunks': len(chunks),
                    'created_at': created_date.isoformat(),
                    'word_count': len(chunk_text.split()),
                    'section_id': f"section_{(j // 2) + 1}",
                    'priority': random.choice(['high', 'medium', 'low']),
                    'tags': random.sample(self.details, random.randint(2, 5))
                }
            )
        
        return len(chunks)
    
    def _create_realistic_chunks(self, content: str, title: str) -> List[str]:
        """Create realistic chunks with variable sizes."""
        words = content.split()
        chunks = []
        
        # Variable chunk sizes (50-150 words)
        i = 0
        while i < len(words):
            chunk_size = random.randint(50, 150)
            chunk_words = words[i:i + chunk_size]
            
            if len(chunk_words) < 20:  # Merge small remaining chunks
                if chunks:
                    chunks[-1] += " " + " ".join(chunk_words)
                else:
                    chunks.append(f"{title}: {' '.join(chunk_words)}")
                break
            
            chunk_text = f"{title}: {' '.join(chunk_words)}"
            chunks.append(chunk_text)
            i += chunk_size
        
        return chunks
    
    async def _optimize_database(self):
        """Optimize database for performance testing."""
        print("  🔧 Optimizing database for performance...")
        
        optimization_queries = [
            "ANALYZE;",  # Update table statistics
            "VACUUM ANALYZE;",  # Clean up and analyze
            "REINDEX DATABASE docfoundry_perf;",  # Rebuild indexes
        ]
        
        for query in optimization_queries:
            try:
                await self.db.execute(query)
                print(f"    ✅ Executed: {query}")
            except Exception as e:
                print(f"    ⚠️  Warning: {query} failed: {e}")
    
    async def cleanup(self):
        """Clean up database connections."""
        if self.db:
            await self.db.close()


async def main():
    """Main seeding function."""
    print("🚀 Starting performance test data seeding...")
    print("   This may take several minutes...")
    
    seeder = PerformanceDataSeeder()
    try:
        await seeder.initialize()
        await seeder.seed_performance_data()
    except Exception as e:
        print(f"❌ Error during seeding: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await seeder.cleanup()


if __name__ == "__main__":
    asyncio.run(main())