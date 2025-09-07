"""Pipelines package for DocFoundry.

Provides crawling, chunking, indexing, and policy checking functionality.
"""

from .crawler import WebCrawler, CrawlResult, CrawlStats, crawl_urls, crawl_urls_sync
from .policy import (
    PolicyViolation,
    RobotsCache,
    PolicyChecker,
    check_url_policy,
    policy_checker
)
from .chunker import DocumentChunker, DocumentChunk, chunk_documents, chunk_single_document
from .indexer import index_documents, get_document_by_id, list_documents, delete_documents_by_source

__all__ = [
    # Crawler
    'WebCrawler',
    'CrawlResult', 
    'CrawlStats',
    'crawl_urls',
    'crawl_urls_sync',
    
    # Policy
    'PolicyViolation',
    'RobotsCache',
    'PolicyChecker',
    'check_url_policy',
    'policy_checker',
    
    # Chunker
    'DocumentChunker',
    'DocumentChunk',
    'chunk_documents',
    'chunk_single_document',
    
    # Indexer
    'index_documents',
    'get_document_by_id',
    'list_documents',
    'delete_documents_by_source'
]