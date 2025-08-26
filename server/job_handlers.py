"""Job handlers for background processing tasks."""

import asyncio
import logging
import pathlib
import sys
from typing import Dict, Any, List, Set
from datetime import datetime

# Add indexer to path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "indexer"))
from embeddings import EmbeddingManager
from sources.loader import load_source_config
from pipelines.crawler import crawl_urls
from pipelines.indexer import index_documents
from pipelines.chunker import chunk_documents
from pipelines.policy import check_url_policy, policy_checker

logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs"
DB_PATH = BASE_DIR / "data" / "docfoundry.db"


async def crawl_source_job(job_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crawl a source and ingest documents with policy compliance checks.
    
    Args:
        job_id: Unique job identifier
        params: Job parameters containing:
            - source_name: Name of the source to crawl
            - urls: List of URLs to crawl (optional)
            - reindex: Whether to reindex after crawling
            - allowed_licenses: Set of allowed licenses for policy compliance
    
    Returns:
        Dict with crawl results
    """
    source_name = params.get("source_name")
    urls = params.get("urls", [])
    reindex = params.get("reindex", False)
    allowed_licenses = params.get('allowed_licenses', {'MIT', 'Apache-2.0', 'BSD-3-Clause', 'CC-BY-4.0'})
    
    logger.info(f"Starting crawl job {job_id} for source: {source_name}")
    
    try:
        # Load source configuration if no URLs provided
        if not urls and source_name:
            try:
                source_config = load_source_config(source_name)
                urls = source_config.get('urls', [])
                logger.info(f"Job {job_id}: Loaded {len(urls)} URLs from source config")
            except Exception as e:
                logger.warning(f"Job {job_id}: Could not load source config: {e}")
                urls = []
        
        if not urls:
            logger.warning(f"Job {job_id}: No URLs to crawl")
            return {
                "source_name": source_name,
                "urls_crawled": 0,
                "errors": ["No URLs provided or found in source config"],
                "success": False
            }
        
        # Policy compliance check for each URL
        compliant_urls = []
        policy_violations = []
        
        for url in urls:
            try:
                logger.info(f"Job {job_id}: Checking policy compliance for {url}")
                policy_result = await check_url_policy(
                    url, 
                    allowed_licenses=set(allowed_licenses),
                    source_name=source_name
                )
                
                if policy_result['robots_compliant']:
                    compliant_urls.append(url)
                    logger.info(f"Job {job_id}: URL {url} passed policy checks")
                    
                    # Log crawl delay if specified
                    if policy_result['crawl_delay']:
                        logger.info(f"Job {job_id}: Crawl delay for {url}: {policy_result['crawl_delay']}s")
                else:
                    policy_violations.append({
                        'url': url,
                        'reason': policy_result['robots_reason']
                    })
                    logger.warning(f"Job {job_id}: URL {url} failed robots.txt check: {policy_result['robots_reason']}")
                    
            except Exception as e:
                logger.error(f"Job {job_id}: Error checking policy for {url}: {e}")
                # Be permissive on policy check errors
                compliant_urls.append(url)
        
        if policy_violations:
            logger.warning(f"Job {job_id}: {len(policy_violations)} URLs blocked by policy: {policy_violations}")
        
        if not compliant_urls:
            logger.warning(f"Job {job_id}: No compliant URLs to crawl after policy checks")
            return {
                "source_name": source_name,
                "urls_crawled": 0,
                "policy_violations": policy_violations,
                "errors": ["No policy-compliant URLs to crawl"],
                "success": False
            }
        
        logger.info(f"Job {job_id}: Proceeding with {len(compliant_urls)} policy-compliant URLs")
        
        crawled_count = 0
        errors = []
        
        # Crawl only policy-compliant URLs
        for url in compliant_urls:
            try:
                # TODO: Implement actual web crawling logic
                # This would involve:
                # - Fetching the URL content
                # - Parsing HTML/content
                # - Extracting meaningful text
                # - Saving to appropriate location in docs/
                logger.info(f"Would crawl URL: {url}")
                crawled_count += 1
                
                # Simulate some processing time
                await asyncio.sleep(0.1)
                
            except Exception as e:
                error_msg = f"Failed to crawl {url}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        result = {
            "source_name": source_name,
            "urls_crawled": crawled_count,
            "policy_violations": policy_violations,
            "errors": errors,
            "success": len(errors) == 0
        }
        
        logger.info(f"Crawl job {job_id} completed: {crawled_count} URLs processed")
        return result
        
    except Exception as e:
        error_msg = f"Crawl job {job_id} failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


async def reindex_job(job_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reindex documents to update embeddings and search index.
    
    Args:
        job_id: Unique job identifier
        params: Job parameters containing:
            - source_filter: Optional source name to filter reindexing
            - force: Whether to force reindexing of all documents
    
    Returns:
        Dict with reindexing results
    """
    source_filter = params.get("source_filter")
    force = params.get("force", False)
    
    logger.info(f"Starting reindex job {job_id}")
    if source_filter:
        logger.info(f"Filtering by source: {source_filter}")
    
    try:
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(str(DB_PATH))
        
        # Get documents that need reindexing
        # This is a simplified implementation
        # In practice, you might want to:
        # 1. Query the database for documents without embeddings
        # 2. Check for documents that have been updated since last indexing
        # 3. Handle incremental vs full reindexing
        
        import sqlite3
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Query for chunks that need embedding updates
            if source_filter:
                cursor.execute(
                    "SELECT id, content FROM chunks WHERE source_url LIKE ? AND (embedding IS NULL OR ?)",
                    (f"%{source_filter}%", force)
                )
            else:
                cursor.execute(
                    "SELECT id, content FROM chunks WHERE embedding IS NULL OR ?",
                    (force,)
                )
            
            chunks_to_process = cursor.fetchall()
        
        processed_count = 0
        errors = []
        
        logger.info(f"Found {len(chunks_to_process)} chunks to process")
        
        # Process chunks in batches
        batch_size = 50
        for i in range(0, len(chunks_to_process), batch_size):
            batch = chunks_to_process[i:i + batch_size]
            
            try:
                # Extract content for batch processing
                batch_content = [chunk[1] for chunk in batch]
                batch_ids = [chunk[0] for chunk in batch]
                
                # Generate embeddings for the batch
                embeddings = embedding_manager.get_embeddings_batch(batch_content)
                
                # Update database with new embeddings
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    for chunk_id, embedding in zip(batch_ids, embeddings):
                        embedding_manager.update_chunk_embedding(chunk_id, embedding)
                        processed_count += 1
                
                logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} chunks")
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                error_msg = f"Failed to process batch {i//batch_size + 1}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        result = {
            "chunks_processed": processed_count,
            "total_chunks": len(chunks_to_process),
            "source_filter": source_filter,
            "errors": errors,
            "success": len(errors) == 0
        }
        
        logger.info(f"Reindex job {job_id} completed: {processed_count} chunks processed")
        return result
        
    except Exception as e:
        error_msg = f"Reindex job {job_id} failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


async def periodic_crawl_job(job_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Periodic job to crawl all configured sources.
    
    Args:
        job_id: Unique job identifier
        params: Job parameters (currently unused)
    
    Returns:
        Dict with periodic crawl results
    """
    logger.info(f"Starting periodic crawl job {job_id}")
    
    try:
        # TODO: Implement periodic crawling logic
        # This would involve:
        # 1. Reading source configurations
        # 2. Determining which sources need updating
        # 3. Enqueuing individual crawl jobs for each source
        # 4. Monitoring overall progress
        
        # Placeholder implementation
        sources_processed = 0
        errors = []
        
        # Simulate processing multiple sources
        configured_sources = ["example-docs", "api-reference"]  # Placeholder
        
        for source in configured_sources:
            try:
                logger.info(f"Processing periodic crawl for source: {source}")
                # In practice, this would enqueue individual crawl jobs
                sources_processed += 1
                await asyncio.sleep(0.1)  # Simulate processing time
                
            except Exception as e:
                error_msg = f"Failed to process source {source}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        result = {
            "sources_processed": sources_processed,
            "total_sources": len(configured_sources),
            "errors": errors,
            "success": len(errors) == 0
        }
        
        logger.info(f"Periodic crawl job {job_id} completed: {sources_processed} sources processed")
        return result
        
    except Exception as e:
        error_msg = f"Periodic crawl job {job_id} failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)