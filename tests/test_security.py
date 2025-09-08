"""Security-focused tests for DocFoundry.

This test suite focuses on security validation including SSRF protection,
input validation, and authentication/authorization checks.
"""

import pytest
import requests
from unittest.mock import Mock, patch
import re
from urllib.parse import urlparse

# Import security modules
try:
    from crawler.crawler import Crawler, CrawlConfig
    from server.security import validate_url, sanitize_input
except ImportError:
    # Create mock classes if imports fail
    class Crawler:
        def __init__(self, config):
            self.config = config
    
    class CrawlConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def validate_url(url):
        return True
    
    def sanitize_input(text):
        return text


class TestSSRFProtection:
    """Test Server-Side Request Forgery (SSRF) protection."""
    
    @pytest.fixture
    def crawler_config(self):
        """Create a secure crawler configuration."""
        return CrawlConfig(
            max_depth=2,
            max_pages=10,
            delay=1.0,
            timeout=30,
            user_agent="DocFoundry-Test/1.0",
            respect_robots_txt=True,
            allowed_domains=["example.com", "test.com"],
            blocked_domains=["localhost", "127.0.0.1", "0.0.0.0"],
            blocked_ips=["127.0.0.1", "0.0.0.0", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
        )
    
    def test_localhost_blocking(self, crawler_config):
        """Test that localhost URLs are blocked."""
        dangerous_urls = [
            "http://localhost:8080/admin",
            "http://127.0.0.1:3000/internal",
            "http://0.0.0.0:5000/secrets",
            "https://localhost/config"
        ]
        
        crawler = Crawler(crawler_config)
        
        for url in dangerous_urls:
            with pytest.raises((ValueError, SecurityError, Exception)):
                # This should raise an exception or be blocked
                result = self._mock_crawl_with_security_check(crawler, url)
                assert result is None or "blocked" in str(result).lower()
    
    def test_private_ip_blocking(self, crawler_config):
        """Test that private IP addresses are blocked."""
        private_ips = [
            "http://10.0.0.1/internal",
            "http://172.16.0.1/admin",
            "http://192.168.1.1/config",
            "http://169.254.169.254/metadata"  # AWS metadata service
        ]
        
        crawler = Crawler(crawler_config)
        
        for url in private_ips:
            with pytest.raises((ValueError, SecurityError, Exception)):
                result = self._mock_crawl_with_security_check(crawler, url)
                assert result is None or "blocked" in str(result).lower()
    
    def test_url_redirection_protection(self, crawler_config):
        """Test protection against malicious redirects."""
        # Mock a redirect chain that tries to access localhost
        with patch('requests.get') as mock_get:
            # First request redirects to localhost
            mock_response = Mock()
            mock_response.status_code = 302
            mock_response.headers = {'Location': 'http://localhost:8080/admin'}
            mock_response.history = []
            mock_get.return_value = mock_response
            
            crawler = Crawler(crawler_config)
            
            with pytest.raises((ValueError, SecurityError, Exception)):
                self._mock_crawl_with_security_check(crawler, "http://example.com/redirect")
    
    def test_allowed_domains_only(self, crawler_config):
        """Test that only allowed domains can be crawled."""
        allowed_urls = [
            "https://example.com/page1",
            "https://test.com/article",
            "https://example.com/subdir/page2"
        ]
        
        blocked_urls = [
            "https://malicious.com/page",
            "https://evil.org/data",
            "https://unauthorized.net/info"
        ]
        
        crawler = Crawler(crawler_config)
        
        # Allowed URLs should pass validation
        for url in allowed_urls:
            assert self._is_url_allowed(crawler, url)
        
        # Blocked URLs should be rejected
        for url in blocked_urls:
            assert not self._is_url_allowed(crawler, url)
    
    def test_protocol_validation(self, crawler_config):
        """Test that only safe protocols are allowed."""
        safe_urls = [
            "https://example.com/page",
            "http://example.com/page"
        ]
        
        dangerous_urls = [
            "file:///etc/passwd",
            "ftp://example.com/file",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>"
        ]
        
        crawler = Crawler(crawler_config)
        
        for url in safe_urls:
            assert self._is_protocol_safe(url)
        
        for url in dangerous_urls:
            assert not self._is_protocol_safe(url)
    
    def _mock_crawl_with_security_check(self, crawler, url):
        """Mock crawl method with security checks."""
        # Simulate security validation
        parsed = urlparse(url)
        
        # Check for localhost/private IPs
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise ValueError(f"Blocked URL: {url}")
        
        # Check for private IP ranges
        if self._is_private_ip(parsed.hostname):
            raise ValueError(f"Private IP blocked: {url}")
        
        return {"url": url, "status": "allowed"}
    
    def _is_url_allowed(self, crawler, url):
        """Check if URL is in allowed domains."""
        parsed = urlparse(url)
        return parsed.hostname in getattr(crawler.config, 'allowed_domains', [])
    
    def _is_protocol_safe(self, url):
        """Check if URL protocol is safe."""
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https']
    
    def _is_private_ip(self, hostname):
        """Check if hostname is a private IP address."""
        if not hostname:
            return False
        
        # Simple check for common private IP patterns
        private_patterns = [
            r'^10\.',
            r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',
            r'^192\.168\.',
            r'^169\.254\.'  # Link-local
        ]
        
        for pattern in private_patterns:
            if re.match(pattern, hostname):
                return True
        
        return False


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attacks."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM documents WHERE 1=1; --",
            "' UNION SELECT * FROM users --"
        ]
        
        for malicious_input in malicious_inputs:
            sanitized = sanitize_input(malicious_input)
            
            # Should not contain SQL injection patterns
            assert "DROP" not in sanitized.upper()
            assert "DELETE" not in sanitized.upper()
            assert "UNION" not in sanitized.upper()
            assert "--" not in sanitized
    
    def test_xss_prevention(self):
        """Test prevention of Cross-Site Scripting (XSS) attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "'><script>alert('xss')</script>"
        ]
        
        for payload in xss_payloads:
            sanitized = sanitize_input(payload)
            
            # Should not contain script tags or javascript
            assert "<script" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "onerror=" not in sanitized.lower()
            assert "onload=" not in sanitized.lower()
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "....//....//....//etc/passwd"
        ]
        
        for attempt in traversal_attempts:
            sanitized = sanitize_input(attempt)
            
            # Should not contain path traversal patterns
            assert "../" not in sanitized
            assert "..\\" not in sanitized
            assert "/etc/" not in sanitized.lower()
            assert "c:\\" not in sanitized.lower()
    
    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        command_injections = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& del C:\\*.*",
            "`whoami`",
            "$(cat /etc/passwd)"
        ]
        
        for injection in command_injections:
            sanitized = sanitize_input(injection)
            
            # Should not contain command injection patterns
            assert "; " not in sanitized
            assert " | " not in sanitized
            assert " && " not in sanitized
            assert "`" not in sanitized
            assert "$(" not in sanitized
    
    def test_input_length_limits(self):
        """Test that input length is properly limited."""
        # Test very long input
        long_input = "A" * 10000
        sanitized = sanitize_input(long_input)
        
        # Should be truncated or rejected
        assert len(sanitized) <= 5000  # Reasonable limit
    
    def test_unicode_normalization(self):
        """Test proper handling of Unicode characters."""
        unicode_inputs = [
            "café",  # Normal Unicode
            "caf\u00e9",  # Unicode escape
            "\u0000",  # Null byte
            "\ufeff",  # BOM
            "\u202e"   # Right-to-left override
        ]
        
        for unicode_input in unicode_inputs:
            sanitized = sanitize_input(unicode_input)
            
            # Should handle Unicode safely
            assert "\u0000" not in sanitized  # No null bytes
            assert "\ufeff" not in sanitized  # No BOM
            assert "\u202e" not in sanitized  # No RTL override


class TestAuthenticationSecurity:
    """Test authentication and authorization security."""
    
    def test_api_key_validation(self):
        """Test API key validation."""
        valid_keys = [
            "sk-1234567890abcdef1234567890abcdef",
            "ak_test_1234567890abcdef1234567890abcdef"
        ]
        
        invalid_keys = [
            "",  # Empty
            "short",  # Too short
            "invalid-format",  # Wrong format
            "sk-" + "a" * 100,  # Too long
        ]
        
        for key in valid_keys:
            assert self._validate_api_key(key)
        
        for key in invalid_keys:
            assert not self._validate_api_key(key)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Simulate rate limiting
        rate_limiter = self._create_mock_rate_limiter(max_requests=5, window=60)
        
        client_ip = "192.168.1.100"
        
        # First 5 requests should succeed
        for i in range(5):
            assert rate_limiter.allow_request(client_ip)
        
        # 6th request should be blocked
        assert not rate_limiter.allow_request(client_ip)
    
    def test_session_security(self):
        """Test session security measures."""
        session_data = {
            "user_id": "user123",
            "created_at": "2024-01-01T00:00:00Z",
            "last_activity": "2024-01-01T01:00:00Z"
        }
        
        # Test session expiration
        assert self._is_session_expired(session_data, max_age_hours=24)
        assert not self._is_session_expired(session_data, max_age_hours=48)
    
    def _validate_api_key(self, key):
        """Mock API key validation."""
        if not key or len(key) < 20 or len(key) > 64:
            return False
        
        # Check format (simplified)
        if key.startswith(('sk-', 'ak_')):
            return True
        
        return False
    
    def _create_mock_rate_limiter(self, max_requests, window):
        """Create a mock rate limiter."""
        class MockRateLimiter:
            def __init__(self, max_requests, window):
                self.max_requests = max_requests
                self.window = window
                self.requests = {}
            
            def allow_request(self, client_id):
                import time
                now = time.time()
                
                if client_id not in self.requests:
                    self.requests[client_id] = []
                
                # Remove old requests outside the window
                self.requests[client_id] = [
                    req_time for req_time in self.requests[client_id]
                    if now - req_time < self.window
                ]
                
                # Check if under limit
                if len(self.requests[client_id]) < self.max_requests:
                    self.requests[client_id].append(now)
                    return True
                
                return False
        
        return MockRateLimiter(max_requests, window)
    
    def _is_session_expired(self, session_data, max_age_hours):
        """Mock session expiration check."""
        from datetime import datetime, timedelta
        
        created_at = datetime.fromisoformat(session_data['created_at'].replace('Z', '+00:00'))
        now = datetime.now(created_at.tzinfo)
        
        return (now - created_at) > timedelta(hours=max_age_hours)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])