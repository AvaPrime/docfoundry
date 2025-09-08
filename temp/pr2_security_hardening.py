# ============================================================================
# PR-2: Security Hardening
# Files: Rate limiting + CORS + SSRF protection
# ============================================================================

# FILE: services/api/app.py (rate limiting and CORS updates)

import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import redis.asyncio as redis

# Initialize rate limiter
def get_limiter():
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    return Limiter(
        key_func=get_remote_address,
        storage_uri=redis_url,
        default_limits=["1000/hour"]  # Global fallback limit
    )

app = FastAPI(title="DocFoundry API", version="1.0.0")
limiter = get_limiter()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add SlowAPI middleware
app.add_middleware(SlowAPIMiddleware)

# CORS configuration with strict origins
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
if not CORS_ORIGINS or CORS_ORIGINS == [""]:
    # Development fallback - should be overridden in production
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)

# Rate limit configurations from environment
QUERY_RATE_LIMIT = os.getenv("RATE_LIMIT_QUERY", "30/minute")
INGEST_RATE_LIMIT = os.getenv("RATE_LIMIT_INGEST", "5/minute")
UPLOAD_RATE_LIMIT = os.getenv("RATE_LIMIT_UPLOAD", "10/hour")

# API key validation for sensitive endpoints
API_KEY = os.getenv("API_KEY")

def validate_api_key(request: Request):
    """Validate API key for sensitive endpoints"""
    if not API_KEY:
        return True  # Skip validation if no API key set (development)
    
    provided_key = request.headers.get("X-API-Key")
    if not provided_key or provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Enhanced rate-limited endpoints
@app.post("/query")
@limiter.limit(QUERY_RATE_LIMIT)
async def query_documents(
    request: Request,
    query_request: QueryRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """Query documents with rate limiting"""
    # Validate input
    if len(query_request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if len(query_request.query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
    
    # Your existing query logic here
    # ...
    pass

@app.post("/ingest")
@limiter.limit(INGEST_RATE_LIMIT)
async def ingest_document(
    request: Request,
    ingest_request: IngestRequest,
    _: bool = Depends(validate_api_key),  # Require API key
    db: AsyncSession = Depends(get_db_session)
):
    """Ingest document with rate limiting and API key protection"""
    # Validate URL safety before processing
    if not is_safe_url(ingest_request.url):
        raise HTTPException(
            status_code=400, 
            detail="URL not allowed (private/local addresses blocked)"
        )
    
    # Your existing ingest logic here
    # ...
    pass

@app.post("/upload")
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    _: bool = Depends(validate_api_key),  # Require API key
    db: AsyncSession = Depends(get_db_session)
):
    """Upload file with comprehensive validation"""
    # File validation
    if not validate_file_upload(file):
        raise HTTPException(status_code=400, detail="Invalid file")
    
    # Your existing upload logic here
    # ...
    pass

# ============================================================================
# FILE: services/shared/security.py (SSRF and file validation)
# ============================================================================

import ipaddress
import mimetypes
import hashlib
from urllib.parse import urlparse, unquote
from typing import Set, List
import magic
from fastapi import UploadFile, HTTPException
import os

# Blocked CIDR ranges for SSRF prevention
BLOCKED_CIDRS = [
    ipaddress.IPv4Network('10.0.0.0/8'),        # Private Class A
    ipaddress.IPv4Network('172.16.0.0/12'),     # Private Class B  
    ipaddress.IPv4Network('192.168.0.0/16'),    # Private Class C
    ipaddress.IPv4Network('127.0.0.0/8'),       # Loopback
    ipaddress.IPv4Network('169.254.0.0/16'),    # Link-local
    ipaddress.IPv4Network('224.0.0.0/4'),       # Multicast
    ipaddress.IPv4Network('240.0.0.0/4'),       # Reserved
    ipaddress.IPv6Network('::1/128'),           # IPv6 loopback
    ipaddress.IPv6Network('fe80::/10'),         # IPv6 link-local
    ipaddress.IPv6Network('fc00::/7'),          # IPv6 unique local
]

# Allowed schemes for URL validation
ALLOWED_SCHEMES = {'http', 'https'}

# Allowed file types for upload
ALLOWED_MIME_TYPES = {
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
    'text/html',
    'text/markdown',
    'application/rtf',
}

# Maximum file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

def is_safe_url(url: str) -> bool:
    """
    Check if URL is safe to crawl (no SSRF vulnerabilities)
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if URL is safe to access
    """
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ALLOWED_SCHEMES:
            return False
        
        # Check for obvious bypasses
        if not parsed.hostname:
            return False
            
        hostname = parsed.hostname.lower()
        
        # Block localhost variants
        localhost_variants = {
            'localhost', '0.0.0.0', '0', '127.0.0.1', 
            '::1', '0:0:0:0:0:0:0:1'
        }
        if hostname in localhost_variants:
            return False
        
        # Try to resolve hostname to IP
        try:
            import socket
            ip_str = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_str)
            
            # Check against blocked CIDR ranges
            for blocked_cidr in BLOCKED_CIDRS:
                if ip in blocked_cidr:
                    return False
                    
        except (socket.gaierror, ValueError, ipaddress.AddressValueError):
            # If we can't resolve, allow it (might be external)
            # The actual HTTP client will handle DNS resolution
            pass
        
        # Additional checks for URL encoding bypasses
        unquoted_hostname = unquote(hostname)
        if unquoted_hostname != hostname:
            # Recursive check for encoded bypasses
            return is_safe_url(f"{parsed.scheme}://{unquoted_hostname}{parsed.path}")
        
        return True
        
    except Exception:
        # If parsing fails, consider it unsafe
        return False

def validate_file_upload(file: UploadFile) -> bool:
    """
    Comprehensive file upload validation
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        bool: True if file is valid and safe
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large (max {MAX_FILE_SIZE // (1024*1024)}MB)"
        )
    
    # Validate filename
    if not file.filename or len(file.filename) > 255:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Check for directory traversal in filename
    if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename characters")
    
    # Read file content for validation
    content = file.file.read(8192)  # Read first 8KB for magic number check
    file.file.seek(0)  # Reset file pointer
    
    # Magic number validation (more reliable than extension)
    try:
        detected_mime = magic.from_buffer(content, mime=True)
        if detected_mime not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed: {detected_mime}"
            )
    except Exception:
        # Fallback to extension-based validation
        _, ext = os.path.splitext(file.filename.lower())
        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt', '.html', '.md', '.rtf'}
        if ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="File extension not allowed")
    
    # Additional PDF-specific validation
    if detected_mime == 'application/pdf':
        if not content.startswith(b'%PDF-'):
            raise HTTPException(status_code=400, detail="Invalid PDF file")
    
    return True

def get_file_hash(content: bytes) -> str:
    """Generate SHA256 hash of file content"""
    return hashlib.sha256(content).hexdigest()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    # Remove/replace dangerous characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:95] + ext
    return filename

# ============================================================================
# FILE: services/worker/crawler.py (SSRF-protected crawler)
# ============================================================================

import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
import time
from services.shared.security import is_safe_url

logger = logging.getLogger(__name__)

class EthicalCrawler:
    """SSRF-protected ethical web crawler with rate limiting and robots.txt respect"""
    
    def __init__(self, user_agent: str = "DocFoundry/1.0", delay: float = 1.0):
        self.user_agent = user_agent
        self.default_delay = delay
        self.robots_cache = {}
        self.last_crawl_times = {}
        
        # Session configuration for security
        self.session_config = {
            'timeout': aiohttp.ClientTimeout(total=30, connect=10),
            'headers': {'User-Agent': self.user_agent},
            'max_redirects': 5,
        }
    
    async def can_crawl(self, url: str) -> bool:
        """Check if URL can be crawled according to robots.txt"""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = urljoin(base_url, '/robots.txt')
        
        # Check robots.txt (with caching)
        if base_url not in self.robots_cache:
            try:
                async with aiohttp.ClientSession(**self.session_config) as session:
                    if is_safe_url(robots_url):
                        async with session.get(robots_url) as response:
                            if response.status == 200:
                                robots_txt = await response.text()
                                rp = RobotFileParser()
                                rp.set_url(robots_url)
                                rp.read_file(robots_txt.splitlines())
                                self.robots_cache[base_url] = rp
                            else:
                                self.robots_cache[base_url] = None
                    else:
                        logger.warning(f"Blocked robots.txt URL due to SSRF protection: {robots_url}")
                        return False
            except Exception as e:
                logger.warning(f"Failed to fetch robots.txt for {base_url}: {e}")
                self.robots_cache[base_url] = None
        
        robots = self.robots_cache.get(base_url)
        if robots and not robots.can_fetch(self.user_agent, url):
            return False
        
        return True
    
    async def respect_crawl_delay(self, url: str):
        """Implement crawl delay based on robots.txt and rate limiting"""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Get delay from robots.txt or use default
        robots = self.robots_cache.get(base_url)
        if robots:
            delay = robots.crawl_delay(self.user_agent) or self.default_delay
        else:
            delay = self.default_delay
        
        # Check last crawl time for this domain
        last_crawl = self.last_crawl_times.get(base_url, 0)
        time_since_last = time.time() - last_crawl
        
        if time_since_last < delay:
            sleep_time = delay - time_since_last
            logger.info(f"Waiting {sleep_time:.1f}s before crawling {base_url}")
            await asyncio.sleep(sleep_time)
        
        self.last_crawl_times[base_url] = time.time()
    
    async def fetch_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Safely fetch URL with SSRF protection and ethical crawling
        
        Returns:
            Dict with 'content', 'status', 'headers' or None if failed/blocked
        """
        # SSRF protection - first line of defense
        if not is_safe_url(url):
            logger.warning(f"Blocked URL due to SSRF protection: {url}")
            return None
        
        # Check robots.txt
        if not await self.can_crawl(url):
            logger.info(f"Blocked by robots.txt: {url}")
            return None
        
        # Respect crawl delay
        await self.respect_crawl_delay(url)
        
        try:
            async with aiohttp.ClientSession(**self.session_config) as session:
                async with session.get(url) as response:
                    # Check for retry-after header
                    if 'Retry-After' in response.headers:
                        retry_after = int(response.headers['Retry-After'])
                        logger.info(f"Server requested retry after {retry_after}s for {url}")
                        await asyncio.sleep(retry_after)
                        # Retry once
                        async with session.get(url) as retry_response:
                            content = await retry_response.text()
                            return {
                                'content': content,
                                'status': retry_response.status,
                                'headers': dict(retry_response.headers),
                                'url': str(retry_response.url)
                            }
                    else:
                        content = await response.text()
                        return {
                            'content': content,
                            'status': response.status,
                            'headers': dict(response.headers),
                            'url': str(response.url)
                        }
                        
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None

# Usage example
async def crawl_document(url: str) -> Optional[Dict[str, Any]]:
    """Crawl a single document with full protection"""
    crawler = EthicalCrawler()
    result = await crawler.fetch_url(url)
    return result

# ============================================================================
# FILE: requirements.txt additions for PR-2
# ============================================================================

# Add these dependencies to your existing requirements.txt:
# slowapi==0.1.9
# python-magic==0.4.27
# redis==4.6.0

# ============================================================================
# FILE: docker-compose.yml (Redis service addition if missing)
# ============================================================================

# Add this to your docker-compose.yml if Redis isn't already present:
"""
services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  redis_data:
"""

# ============================================================================
# FILE: .env.example (Security configuration template)
# ============================================================================

"""
# CORS Configuration (comma-separated origins)
CORS_ORIGINS=https://your-frontend.example.com,https://admin.example.com

# Rate Limiting
REDIS_URL=redis://redis:6379
RATE_LIMIT_QUERY=30/minute
RATE_LIMIT_INGEST=5/minute
RATE_LIMIT_UPLOAD=10/hour

# API Security
API_KEY=your-secret-api-key-here

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/docfoundry
IVFFLAT_PROBES=10

# Crawler Settings
CRAWLER_USER_AGENT=DocFoundry/1.0
CRAWLER_DEFAULT_DELAY=1.0
CRAWLER_TIMEOUT=30
"""