"""Security package for DocFoundry API."""

from .net import is_safe_url, PRIVATE_NETS, METADATA_HOSTS
from .rate_limiting import (
    limiter,
    setup_rate_limiting,
    strict_rate_limit,
    moderate_rate_limit,
    generous_rate_limit,
    burst_rate_limit,
    user_rate_limit,
    RateLimitMiddleware,
    rate_limit_health_check
)
from .cors import (
    setup_cors,
    setup_security_headers,
    setup_api_security,
    get_allowed_origins,
    validate_origin,
    StrictCORSMiddleware,
    SecurityHeadersMiddleware
)

__all__ = [
    # Network security
    "is_safe_url",
    "PRIVATE_NETS",
    "METADATA_HOSTS",
    # Rate limiting
    "limiter",
    "setup_rate_limiting",
    "strict_rate_limit",
    "moderate_rate_limit",
    "generous_rate_limit",
    "burst_rate_limit",
    "user_rate_limit",
    "RateLimitMiddleware",
    "rate_limit_health_check",
    # CORS and security headers
    "setup_cors",
    "setup_security_headers",
    "setup_api_security",
    "get_allowed_origins",
    "validate_origin",
    "StrictCORSMiddleware",
    "SecurityHeadersMiddleware"
]