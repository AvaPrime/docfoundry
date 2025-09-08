"""Rate limiting middleware using Redis and slowapi for DocFoundry API security."""

import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Redis connection for rate limiting
def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client for rate limiting storage."""
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        client = redis.from_url(redis_url, decode_responses=True)
        # Test connection
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Rate limiting will use in-memory storage.")
        return None

# Initialize rate limiter
redis_client = get_redis_client()
if redis_client:
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=os.getenv("REDIS_URL", "redis://localhost:6379")
    )
else:
    # Fallback to in-memory storage
    limiter = Limiter(key_func=get_remote_address)

# Custom rate limit exceeded handler
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded responses."""
    response = JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": getattr(exc, 'retry_after', None)
        }
    )
    response.headers["Retry-After"] = str(getattr(exc, 'retry_after', 60))
    return response

def setup_rate_limiting(app: FastAPI) -> None:
    """Setup rate limiting for FastAPI application."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

# Rate limiting decorators for different endpoints
def strict_rate_limit():
    """Strict rate limiting for sensitive endpoints (10 requests per minute)."""
    return limiter.limit("10/minute")

def moderate_rate_limit():
    """Moderate rate limiting for API endpoints (100 requests per minute)."""
    return limiter.limit("100/minute")

def generous_rate_limit():
    """Generous rate limiting for search endpoints (300 requests per minute)."""
    return limiter.limit("300/minute")

def burst_rate_limit():
    """Burst rate limiting for high-frequency operations (1000 requests per hour)."""
    return limiter.limit("1000/hour")

# IP-based rate limiting
def get_client_ip(request: Request) -> str:
    """Extract client IP considering proxy headers."""
    # Check for forwarded headers (common in production behind proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"

# Advanced rate limiting with user context
def get_user_identifier(request: Request) -> str:
    """Get user identifier for rate limiting (IP + user ID if authenticated)."""
    client_ip = get_client_ip(request)
    
    # Try to get user ID from authentication context
    user_id = getattr(request.state, 'user_id', None)
    if user_id:
        return f"user:{user_id}:{client_ip}"
    
    return f"ip:{client_ip}"

# Custom limiter with user context
user_limiter = Limiter(
    key_func=get_user_identifier,
    storage_uri=os.getenv("REDIS_URL", "redis://localhost:6379") if redis_client else None
)

def user_rate_limit(rate: str):
    """Rate limiting based on user context (authenticated users get higher limits)."""
    return user_limiter.limit(rate)

# Rate limiting middleware class
class RateLimitMiddleware:
    """Middleware for applying rate limits to all requests."""
    
    def __init__(self, app: FastAPI, default_rate: str = "1000/hour"):
        self.app = app
        self.default_rate = default_rate
        setup_rate_limiting(app)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Apply default rate limiting to all requests
            try:
                await limiter.check_request(request, self.default_rate)
            except RateLimitExceeded as e:
                response = await rate_limit_handler(request, e)
                await response(scope, receive, send)
                return
        
        await self.app(scope, receive, send)

# Health check for rate limiting system
async def rate_limit_health_check() -> dict:
    """Check health of rate limiting system."""
    health = {
        "rate_limiting": "healthy",
        "redis_connected": False,
        "storage_type": "in-memory"
    }
    
    if redis_client:
        try:
            redis_client.ping()
            health["redis_connected"] = True
            health["storage_type"] = "redis"
        except Exception as e:
            health["rate_limiting"] = "degraded"
            health["error"] = str(e)
    
    return health