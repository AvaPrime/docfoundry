#!/usr/bin/env python3
"""
Comprehensive tests for SSRF protection functionality

This module tests the security measures implemented to prevent
Server-Side Request Forgery (SSRF) attacks in the DocFoundry crawler.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from pipelines.security import check_url_ssrf, SSRFError, get_safe_connector
from pipelines.crawler import WebCrawler
import aiohttp

class TestSSRFProtection:
    """Test suite for SSRF protection mechanisms"""
    
    def test_private_ip_ranges_blocked(self):
        """Test that private IP ranges are blocked"""
        private_ips = [
            "http://127.0.0.1/test",
            "http://localhost/test", 
            "http://10.0.0.1/test",
            "http://172.16.0.1/test",
            "http://192.168.1.1/test",
            "http://169.254.1.1/test",  # Link-local
            "http://[::1]/test",  # IPv6 localhost
            "http://[fc00::1]/test",  # IPv6 private
        ]
        
        for url in private_ips:
            with pytest.raises(SSRFError, match="Private IP address detected"):
                check_url_ssrf(url)
    
    def test_local_file_access_blocked(self):
        """Test that local file access is blocked"""
        file_urls = [
            "file:///etc/passwd",
            "file://C:\\Windows\\System32\\config\\SAM",
            "file:///home/user/.ssh/id_rsa",
            "file://localhost/etc/hosts",
        ]
        
        for url in file_urls:
            with pytest.raises(SSRFError, match="File protocol not allowed"):
                check_url_ssrf(url)
    
    def test_internal_services_blocked(self):
        """Test that internal service ports are blocked"""
        internal_services = [
            "http://127.0.0.1:22/test",  # SSH
            "http://localhost:3306/test",  # MySQL
            "http://10.0.0.1:5432/test",  # PostgreSQL
            "http://192.168.1.1:6379/test",  # Redis
            "http://172.16.0.1:27017/test",  # MongoDB
            "http://127.0.0.1:9200/test",  # Elasticsearch
        ]
        
        for url in internal_services:
            with pytest.raises(SSRFError, match="Private IP address detected"):
                check_url_ssrf(url)
    
    def test_valid_public_urls_allowed(self):
        """Test that valid public URLs are allowed"""
        public_urls = [
            "https://example.com/docs",
            "http://docs.python.org/3/",
            "https://github.com/user/repo",
            "https://8.8.8.8/test",  # Google DNS
            "https://1.1.1.1/test",  # Cloudflare DNS
        ]
        
        for url in public_urls:
            # Should not raise any exception
            check_url_ssrf(url)
    
    def test_malformed_urls_handled(self):
        """Test that malformed URLs are handled gracefully"""
        malformed_urls = [
            "not-a-url",
            "http://",
            "://missing-scheme",
            "http://[invalid-ipv6",
            "ftp://example.com",  # Unsupported scheme
        ]
        
        for url in malformed_urls:
            with pytest.raises(SSRFError):
                check_url_ssrf(url)
    
    def test_url_redirection_protection(self):
        """Test protection against URL redirection attacks"""
        # Test URLs that might redirect to private IPs
        suspicious_urls = [
            "http://example.com@127.0.0.1/",
            "http://127.0.0.1#example.com",
            "http://example.com/redirect?url=http://localhost",
        ]
        
        # The first two should be caught by URL parsing
        with pytest.raises(SSRFError):
            check_url_ssrf(suspicious_urls[0])
        
        # The third one should pass initial validation but be caught during crawling
        check_url_ssrf(suspicious_urls[2])  # Should pass initial check
    
    def test_safe_connector_configuration(self):
        """Test that the safe connector is properly configured"""
        connector = get_safe_connector()
        
        # Verify connector settings
        assert isinstance(connector, aiohttp.TCPConnector)
        assert connector._limit == 100  # Connection limit
        assert connector._limit_per_host == 10  # Per-host limit
    
    @pytest.mark.asyncio
    async def test_crawler_ssrf_integration(self):
        """Test SSRF protection integration in the crawler"""
        crawler = WebCrawler(max_concurrent=1)
        
        # Test that crawler rejects private IPs
        result = await crawler._fetch_url("http://127.0.0.1/test")
        assert result.status_code == 0
        assert "SSRF" in result.error or "Private IP" in result.error
        
        await crawler.close()
    
    def test_ipv6_private_ranges(self):
        """Test IPv6 private address detection"""
        ipv6_private = [
            "http://[::1]/test",  # Loopback
            "http://[fe80::1]/test",  # Link-local
            "http://[fc00::1]/test",  # Unique local
            "http://[fd00::1]/test",  # Unique local
        ]
        
        for url in ipv6_private:
            with pytest.raises(SSRFError, match="Private IP address detected"):
                check_url_ssrf(url)
    
    def test_dns_rebinding_protection(self):
        """Test protection against DNS rebinding attacks"""
        # Test domains that might resolve to private IPs
        suspicious_domains = [
            "http://localtest.me/",  # Known to resolve to 127.0.0.1
            "http://vcap.me/",      # Another service that resolves to localhost
        ]
        
        # These should pass URL validation but be caught during actual resolution
        for url in suspicious_domains:
            check_url_ssrf(url)  # Should pass initial validation
    
    def test_port_scanning_prevention(self):
        """Test prevention of port scanning via SSRF"""
        # Test various ports on localhost
        ports_to_test = [22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 5432, 6379, 9200]
        
        for port in ports_to_test:
            url = f"http://127.0.0.1:{port}/"
            with pytest.raises(SSRFError, match="Private IP address detected"):
                check_url_ssrf(url)
    
    def test_cloud_metadata_endpoints_blocked(self):
        """Test that cloud metadata endpoints are blocked"""
        metadata_endpoints = [
            "http://169.254.169.254/latest/meta-data/",  # AWS
            "http://metadata.google.internal/",           # GCP
            "http://169.254.169.254/metadata/instance",   # Azure
        ]
        
        for url in metadata_endpoints:
            with pytest.raises(SSRFError):
                check_url_ssrf(url)
    
    def test_unicode_domain_handling(self):
        """Test handling of unicode domains that might bypass filters"""
        unicode_domains = [
            "http://localhost\u2024com/",  # Unicode dot
            "http://127\u00010\u00010\u00011/",  # Unicode in IP
        ]
        
        for url in unicode_domains:
            with pytest.raises(SSRFError):
                check_url_ssrf(url)
    
    @pytest.mark.asyncio
    async def test_timeout_protection(self):
        """Test that requests have appropriate timeouts"""
        crawler = WebCrawler(request_timeout=1)  # 1 second timeout
        
        # Mock a slow response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = asyncio.coroutine(lambda: "<html></html>")
            mock_response.url = "https://example.com"
            
            # Simulate timeout
            mock_get.side_effect = asyncio.TimeoutError()
            
            result = await crawler._fetch_url("https://example.com")
            assert "timeout" in result.error.lower() or result.status_code == 0
        
        await crawler.close()
    
    def test_scheme_validation(self):
        """Test that only allowed URL schemes are accepted"""
        invalid_schemes = [
            "ftp://example.com/file.txt",
            "gopher://example.com/",
            "ldap://example.com/",
            "dict://example.com/",
            "sftp://example.com/",
        ]
        
        for url in invalid_schemes:
            with pytest.raises(SSRFError, match="Unsupported URL scheme"):
                check_url_ssrf(url)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        edge_cases = [
            "http://0.0.0.0/",           # All zeros
            "http://255.255.255.255/",   # Broadcast
            "http://127.000.000.001/",   # Leading zeros
            "http://2130706433/",        # Decimal IP (127.0.0.1)
            "http://0x7f000001/",        # Hex IP (127.0.0.1)
        ]
        
        for url in edge_cases:
            with pytest.raises(SSRFError):
                check_url_ssrf(url)

class TestSSRFIntegration:
    """Integration tests for SSRF protection in the full system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_protection(self):
        """Test end-to-end SSRF protection in crawling workflow"""
        crawler = WebCrawler(enable_incremental=False)
        
        # Test crawling a list with mixed valid and invalid URLs
        urls = [
            "https://httpbin.org/html",  # Valid public URL
            "http://127.0.0.1/admin",    # Should be blocked
            "https://example.com/docs",  # Valid public URL
        ]
        
        results = []
        for url in urls:
            result = await crawler._fetch_url(url)
            results.append(result)
        
        # First URL should succeed (or fail with network error, not SSRF)
        assert results[0].status_code != 0 or "SSRF" not in (results[0].error or "")
        
        # Second URL should be blocked by SSRF protection
        assert results[1].status_code == 0
        assert "SSRF" in results[1].error or "Private IP" in results[1].error
        
        # Third URL should succeed (or fail with network error, not SSRF)
        assert results[2].status_code != 0 or "SSRF" not in (results[2].error or "")
        
        await crawler.close()
    
    def test_configuration_validation(self):
        """Test that SSRF protection configuration is validated"""
        # Test that crawler initializes with SSRF protection enabled by default
        crawler = WebCrawler()
        assert hasattr(crawler, 'session')
        
        # Test safe connector creation
        connector = get_safe_connector()
        assert connector is not None
        assert isinstance(connector, aiohttp.TCPConnector)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])