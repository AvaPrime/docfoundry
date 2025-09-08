"""CORS (Cross-Origin Resource Sharing) security configuration for DocFoundry API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def get_allowed_origins() -> List[str]:
    """Get allowed origins from environment variables with secure defaults."""
    # Default to localhost for development
    default_origins = [
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # FastAPI dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]
    
    # Get production origins from environment
    env_origins = os.getenv("ALLOWED_ORIGINS", "")
    if env_origins:
        # Parse comma-separated origins
        production_origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
        logger.info(f"Using production CORS origins: {production_origins}")
        return production_origins
    
    # In production, warn about using default origins
    if os.getenv("ENVIRONMENT", "development") == "production":
        logger.warning("Using default CORS origins in production. Set ALLOWED_ORIGINS environment variable.")
    
    return default_origins

def get_cors_config() -> dict:
    """Get CORS configuration based on environment."""
    is_production = os.getenv("ENVIRONMENT", "development") == "production"
    
    config = {
        "allow_origins": get_allowed_origins(),
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-API-Key",
            "X-Client-Version"
        ],
        "expose_headers": [
            "X-Total-Count",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset",
            "X-Request-ID"
        ],
        "max_age": 86400 if is_production else 600  # 24 hours in prod, 10 minutes in dev
    }
    
    if is_production:
        # Stricter settings for production
        config["allow_methods"] = ["GET", "POST", "PUT", "DELETE"]  # Remove OPTIONS
        logger.info("Using production CORS configuration")
    else:
        # More permissive for development
        config["allow_origins"].append("*")  # Allow all origins in dev (if needed)
        logger.info("Using development CORS configuration")
    
    return config

def setup_cors(app: FastAPI, custom_origins: Optional[List[str]] = None) -> None:
    """Setup CORS middleware for FastAPI application."""
    config = get_cors_config()
    
    # Override origins if provided
    if custom_origins:
        config["allow_origins"] = custom_origins
        logger.info(f"Using custom CORS origins: {custom_origins}")
    
    app.add_middleware(
        CORSMiddleware,
        **config
    )
    
    logger.info(f"CORS configured with origins: {config['allow_origins']}")

def validate_origin(origin: str, allowed_origins: List[str]) -> bool:
    """Validate if an origin is allowed."""
    if "*" in allowed_origins:
        return True
    
    # Exact match
    if origin in allowed_origins:
        return True
    
    # Check for wildcard subdomains (e.g., *.example.com)
    for allowed in allowed_origins:
        if allowed.startswith("*."):
            domain = allowed[2:]  # Remove *.
            if origin.endswith(f".{domain}") or origin == domain:
                return True
    
    return False

class StrictCORSMiddleware:
    """Custom CORS middleware with additional security checks."""
    
    def __init__(self, app: FastAPI, allowed_origins: List[str]):
        self.app = app
        self.allowed_origins = allowed_origins
        
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            origin = headers.get(b"origin", b"").decode("utf-8")
            
            # Check origin if present
            if origin and not validate_origin(origin, self.allowed_origins):
                # Reject requests from unauthorized origins
                response = {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"content-length", b"45"]
                    ]
                }
                await send(response)
                await send({
                    "type": "http.response.body",
                    "body": b'{"error": "Origin not allowed", "code": 403}'
                })
                return
        
        await self.app(scope, receive, send)

# Security headers middleware
class SecurityHeadersMiddleware:
    """Add security headers to all responses."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                
                # Add security headers
                security_headers = {
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"x-xss-protection": b"1; mode=block",
                    b"strict-transport-security": b"max-age=31536000; includeSubDomains",
                    b"referrer-policy": b"strict-origin-when-cross-origin",
                    b"permissions-policy": b"geolocation=(), microphone=(), camera=()"
                }
                
                # Only add HSTS in production with HTTPS
                if os.getenv("ENVIRONMENT") != "production":
                    del security_headers[b"strict-transport-security"]
                
                headers.update(security_headers)
                message["headers"] = list(headers.items())
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

def setup_security_headers(app: FastAPI) -> None:
    """Setup security headers middleware."""
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security headers middleware configured")

# Complete security setup function
def setup_api_security(app: FastAPI, custom_origins: Optional[List[str]] = None) -> None:
    """Setup complete API security including CORS and security headers."""
    setup_cors(app, custom_origins)
    setup_security_headers(app)
    logger.info("API security configuration complete")