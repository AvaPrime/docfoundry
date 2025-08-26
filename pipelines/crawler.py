"""Web crawler pipeline for DocFoundry.

Provides crawling functionality with policy compliance checks.
"""

import asyncio
import logging
import time
import random
from typing import List, Dict, Set, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from dataclasses import dataclass
from datetime import datetime, timedelta

import aiohttp
import asyncio
from bs4 import BeautifulSoup

from .policy import check_url_policy, policy_checker, PolicyViolation
from sources.loader import load_source_config, SourceConfig

logger = logging.getLogger(__name__)

@dataclass
class CrawlResult:
    """Result of crawling a single URL."""
    url: str
    status_code: int
    content: Optional[str] = None
    content_type: Optional[str] = None
    error: Optional[str] = None
    policy_violations: List[PolicyViolation] = None
    crawl_time: Optional[datetime] = None
    response_time: Optional[float] = None
    retry_count: int = 0
    final_url: Optional[str] = None  # After redirects
    
    def __post_init__(self):
        if self.policy_violations is None:
            self.policy_violations = []
        if self.crawl_time is None:
            self.crawl_time = datetime.utcnow()

@dataclass
class CrawlStats:
    """Statistics for a crawl session."""
    total_urls: int = 0
    successful: int = 0
    failed: int = 0
    policy_blocked: int = 0
    retried: int = 0
    redirected: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.utcnow()
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
    
    def finish(self):
        """Mark crawl as finished."""
        self.end_time = datetime.utcnow()

class WebCrawler:
    """Asynchronous web crawler with policy compliance."""
    
    def __init__(self, 
                 max_concurrent: int = 10,
                 request_timeout: int = 30,
                 user_agent: str = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 max_retry_delay: float = 60.0):
        """Initialize crawler.
        
        Args:
            max_concurrent: Maximum concurrent requests
            request_timeout: Request timeout in seconds
            user_agent: User agent string (defaults to policy config)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (seconds)
            max_retry_delay: Maximum delay between retries (seconds)
        """
        self.max_concurrent = max_concurrent
        self.request_timeout = request_timeout
        self.user_agent = user_agent or policy_checker.user_agent
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        
        # Rate limiting
        self.last_request_time: Dict[str, float] = {}
        self.rate_limits: Dict[str, float] = {}
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2)
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': self.user_agent}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def close(self):
        """Close the crawler session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc
    
    async def _respect_rate_limit(self, url: str, rate_limit: float):
        """Respect rate limiting for domain."""
        domain = self._get_domain(url)
        
        if domain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[domain]
            if elapsed < rate_limit:
                sleep_time = rate_limit - elapsed
                logger.debug(f"Rate limiting {domain}: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        self.last_request_time[domain] = time.time()
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        base_delay = self.retry_delay * (2 ** attempt)
        jitter = random.uniform(0.1, 0.3) * base_delay
        delay = min(base_delay + jitter, self.max_retry_delay)
        return delay
    
    def _is_retryable_error(self, exception: Exception, status_code: int = None) -> bool:
        """Determine if an error is retryable."""
        # Retryable HTTP status codes
        retryable_status_codes = {408, 429, 500, 502, 503, 504}
        
        if status_code and status_code in retryable_status_codes:
            return True
        
        # Retryable exceptions
        if isinstance(exception, (asyncio.TimeoutError, aiohttp.ServerTimeoutError)):
            return True
        
        if isinstance(exception, aiohttp.ClientError):
            # Retry on connection errors, but not on client errors like 404
            return isinstance(exception, (aiohttp.ClientConnectionError, 
                                        aiohttp.ClientConnectorError,
                                        aiohttp.ServerDisconnectedError))
        
        return False
    
    async def _fetch_url(self, url: str, source_name: str, rate_limit: float = 0.5) -> CrawlResult:
        """Fetch a single URL with policy checks and retry logic."""
        start_time = time.time()
        
        try:
            # Check policy compliance first
            policy_result = await check_url_policy(url, source_name=source_name)
            
            if not policy_result or not policy_result.get('overall_compliant', False):
                reason = policy_result.get('reason', 'Unknown') if policy_result else 'Policy check failed'
                logger.info(f"Policy blocked URL {url}: {reason}")
                return CrawlResult(
                    url=url,
                    status_code=403,
                    error=f"Policy violation: {reason}",
                    policy_violations=policy_result.get('violations', []) if policy_result else [],
                    response_time=time.time() - start_time
                )
            
            # Initialize session if not already done
            if not self.session:
                await self.__aenter__()
            
            # Retry loop
            last_exception = None
            for attempt in range(self.max_retries + 1):
                try:
                    # Respect rate limiting
                    await self._respect_rate_limit(url, rate_limit)
                    
                    async with self.semaphore:
                        logger.debug(f"Fetching {url} (attempt {attempt + 1}/{self.max_retries + 1})")
                        
                        async with self.session.get(url, allow_redirects=True) as response:
                            content_type = response.headers.get('content-type', '')
                            
                            # Check if we should retry based on status code
                            if self._is_retryable_error(None, response.status) and attempt < self.max_retries:
                                logger.warning(f"Retryable status {response.status} for {url}, attempt {attempt + 1}/{self.max_retries + 1}")
                                delay = self._calculate_retry_delay(attempt)
                                await asyncio.sleep(delay)
                                continue
                            
                            # Only process text content
                            if not content_type.startswith('text/'):
                                return CrawlResult(
                                    url=url,
                                    status_code=response.status,
                                    error=f"Non-text content type: {content_type}",
                                    content_type=content_type,
                                    response_time=time.time() - start_time,
                                    retry_count=attempt,
                                    final_url=str(response.url)
                                )
                            
                            content = await response.text()
                            
                            # Check content policy
                            content_policy = policy_checker.check_content_policy(content, url, source_name=source_name)
                            
                            violations = content_policy.get('violations', [])
                            if violations:
                                logger.info(f"Content policy violations for {url}: {len(violations)} violations")
                            
                            return CrawlResult(
                                url=url,
                                status_code=response.status,
                                content=content,
                                content_type=content_type,
                                policy_violations=violations,
                                response_time=time.time() - start_time,
                                retry_count=attempt,
                                final_url=str(response.url)
                            )
                
                except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
                    last_exception = e
                    if attempt < self.max_retries and self._is_retryable_error(e):
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Timeout fetching {url}, retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.warning(f"Timeout fetching {url} after {attempt + 1} attempts")
                        break
                        
                except aiohttp.ClientError as e:
                    last_exception = e
                    if attempt < self.max_retries and self._is_retryable_error(e):
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Client error fetching {url}: {e}, retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.warning(f"Client error fetching {url} after {attempt + 1} attempts: {e}")
                        break
                        
                except Exception as e:
                    last_exception = e
                    logger.error(f"Unexpected error fetching {url}: {e}")
                    break
            
            # If we get here, all retries failed
            error_msg = str(last_exception) if last_exception else "Unknown error"
            status_code = 408 if isinstance(last_exception, (asyncio.TimeoutError, aiohttp.ServerTimeoutError)) else 0
            
            return CrawlResult(
                url=url,
                status_code=status_code,
                error=error_msg,
                response_time=time.time() - start_time,
                retry_count=self.max_retries
            )
        
        except Exception as e:
            logger.error(f"Unexpected error in _fetch_url for {url}: {e}")
            return CrawlResult(
                url=url,
                status_code=0,
                error=f"Unexpected error: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def _extract_links(self, content: str, base_url: str) -> Set[str]:
        """Extract links from HTML content."""
        links = set()
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if href:
                    absolute_url = urljoin(base_url, href)
                    # Remove fragment
                    parsed = urlparse(absolute_url)
                    clean_url = urlunparse(parsed._replace(fragment=''))
                    links.add(clean_url)
        
        except Exception as e:
            logger.warning(f"Failed to extract links from {base_url}: {e}")
        
        return links
    
    def _should_follow_link(self, url: str, base_domain: str, 
                           include_patterns: List[str] = None,
                           exclude_patterns: List[str] = None) -> bool:
        """Check if a link should be followed."""
        parsed = urlparse(url)
        
        # Only follow links on the same domain
        if parsed.netloc != base_domain:
            return False
        
        # Check include patterns
        if include_patterns:
            if not any(pattern in url for pattern in include_patterns):
                return False
        
        # Check exclude patterns
        if exclude_patterns:
            if any(pattern in url for pattern in exclude_patterns):
                return False
        
        return True
    
    async def crawl_urls(self, urls: List[str], source_name: str, 
                        max_depth: int = 3) -> Tuple[List[CrawlResult], CrawlStats]:
        """Crawl a list of URLs.
        
        Args:
            urls: List of URLs to crawl
            source_name: Name of the source configuration
            max_depth: Maximum crawl depth
        
        Returns:
            Tuple of (results, stats)
        """
        stats = CrawlStats()
        results = []
        
        # Load source configuration
        source_config = load_source_config(source_name)
        if not source_config:
            logger.warning(f"No source configuration found for {source_name}")
            source_config = SourceConfig(
                name=source_name,
                base_urls=urls,
                rate_limit=0.5,
                depth=max_depth
            )
        
        rate_limit = source_config.rate_limit
        crawl_depth = min(max_depth, source_config.depth)
        
        # Track URLs by depth
        urls_to_crawl = {0: set(urls)}
        crawled_urls = set()
        
        logger.info(f"Starting crawl of {len(urls)} URLs for source '{source_name}' (max_depth={crawl_depth})")
        
        for depth in range(crawl_depth + 1):
            if depth not in urls_to_crawl or not urls_to_crawl[depth]:
                continue
            
            current_urls = list(urls_to_crawl[depth] - crawled_urls)
            if not current_urls:
                continue
            
            logger.info(f"Crawling depth {depth}: {len(current_urls)} URLs")
            
            # Crawl URLs at current depth
            tasks = [
                self._fetch_url(url, source_name, rate_limit)
                for url in current_urls
            ]
            
            depth_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(depth_results):
                url = current_urls[i]
                crawled_urls.add(url)
                stats.total_urls += 1
                
                if isinstance(result, Exception):
                    logger.error(f"Exception crawling {url}: {result}")
                    result = CrawlResult(
                        url=url,
                        status_code=0,
                        error=f"Exception: {str(result)}"
                    )
                
                results.append(result)
                
                # Update stats
                if result.retry_count > 0:
                    stats.retried += 1
                
                if result.final_url and result.final_url != url:
                    stats.redirected += 1
                
                if result.status_code == 403 and "Policy violation" in (result.error or ""):
                    stats.policy_blocked += 1
                elif result.status_code == 200:
                    stats.successful += 1
                    
                    # Extract links for next depth
                    if depth < crawl_depth and result.content:
                        try:
                            base_domain = urlparse(url).netloc
                            links = self._extract_links(result.content, url)
                            
                            # Filter links
                            valid_links = {
                                link for link in links
                                if self._should_follow_link(
                                    link, base_domain,
                                    source_config.include,
                                    source_config.exclude
                                )
                            }
                            
                            if valid_links:
                                next_depth = depth + 1
                                if next_depth not in urls_to_crawl:
                                    urls_to_crawl[next_depth] = set()
                                urls_to_crawl[next_depth].update(valid_links)
                                
                                logger.debug(f"Found {len(valid_links)} links at depth {depth} for {url}")
                        
                        except Exception as e:
                            logger.warning(f"Failed to extract links from {url}: {e}")
                else:
                    stats.failed += 1
        
        stats.finish()
        
        logger.info(f"Crawl completed: {stats.successful} successful, {stats.failed} failed, "
                   f"{stats.policy_blocked} policy blocked, {stats.retried} retried, "
                   f"{stats.redirected} redirected out of {stats.total_urls} total URLs")
        
        return results, stats

# Convenience functions
async def crawl_urls(urls: List[str], source_name: str, 
                    max_depth: int = 3, max_concurrent: int = 10) -> Tuple[List[CrawlResult], CrawlStats]:
    """Convenience function to crawl URLs.
    
    Args:
        urls: List of URLs to crawl
        source_name: Name of the source configuration
        max_depth: Maximum crawl depth
        max_concurrent: Maximum concurrent requests
    
    Returns:
        Tuple of (results, stats)
    """
    async with WebCrawler(max_concurrent=max_concurrent) as crawler:
        return await crawler.crawl_urls(urls, source_name, max_depth)

def crawl_urls_sync(urls: List[str], source_name: str, 
                   max_depth: int = 3, max_concurrent: int = 10) -> Tuple[List[CrawlResult], CrawlStats]:
    """Synchronous wrapper for crawl_urls.
    
    Args:
        urls: List of URLs to crawl
        source_name: Name of the source configuration
        max_depth: Maximum crawl depth
        max_concurrent: Maximum concurrent requests
    
    Returns:
        Tuple of (results, stats)
    """
    return asyncio.run(crawl_urls(urls, source_name, max_depth, max_concurrent))