"""Security utilities for DocFoundry pipelines.

Provides SSRF protection and URL validation to prevent requests to private IP ranges and local files.
"""

import ipaddress
import socket
from typing import Optional, Set, Tuple
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

# Private IP ranges as defined by RFC 1918, RFC 4193, and others
PRIVATE_IP_RANGES = [
    ipaddress.ip_network('10.0.0.0/8'),        # RFC 1918
    ipaddress.ip_network('172.16.0.0/12'),     # RFC 1918
    ipaddress.ip_network('192.168.0.0/16'),    # RFC 1918
    ipaddress.ip_network('127.0.0.0/8'),       # Loopback
    ipaddress.ip_network('169.254.0.0/16'),    # Link-local
    ipaddress.ip_network('::1/128'),           # IPv6 loopback
    ipaddress.ip_network('fc00::/7'),          # IPv6 unique local
    ipaddress.ip_network('fe80::/10'),         # IPv6 link-local
    ipaddress.ip_network('0.0.0.0/8'),         # "This" network
    ipaddress.ip_network('224.0.0.0/4'),       # Multicast
    ipaddress.ip_network('240.0.0.0/4'),       # Reserved
]

# Blocked ports (common internal services)
BLOCKED_PORTS = {
    22,    # SSH
    23,    # Telnet
    25,    # SMTP
    53,    # DNS
    110,   # POP3
    143,   # IMAP
    993,   # IMAPS
    995,   # POP3S
    1433,  # SQL Server
    1521,  # Oracle
    3306,  # MySQL
    3389,  # RDP
    5432,  # PostgreSQL
    5984,  # CouchDB
    6379,  # Redis
    8086,  # InfluxDB
    9200,  # Elasticsearch
    27017, # MongoDB
}

# Allowed schemes
ALLOWED_SCHEMES = {'http', 'https'}

class SSRFError(Exception):
    """Exception raised when SSRF protection blocks a request."""
    pass

def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private range.
    
    Args:
        ip_str: IP address as string
        
    Returns:
        True if IP is in private range, False otherwise
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in network for network in PRIVATE_IP_RANGES)
    except ValueError:
        # Invalid IP address
        return True  # Err on the side of caution

def resolve_hostname(hostname: str) -> Set[str]:
    """Resolve hostname to IP addresses.
    
    Args:
        hostname: Hostname to resolve
        
    Returns:
        Set of IP addresses
        
    Raises:
        SSRFError: If hostname resolution fails or returns private IPs
    """
    try:
        # Get all IP addresses for the hostname
        addr_info = socket.getaddrinfo(hostname, None)
        ips = {info[4][0] for info in addr_info}
        
        # Check if any resolved IP is private
        private_ips = [ip for ip in ips if is_private_ip(ip)]
        if private_ips:
            raise SSRFError(f"Hostname {hostname} resolves to private IP(s): {private_ips}")
        
        return ips
        
    except socket.gaierror as e:
        raise SSRFError(f"Failed to resolve hostname {hostname}: {e}")
    except Exception as e:
        raise SSRFError(f"Error resolving hostname {hostname}: {e}")

def validate_url_security(url: str) -> Tuple[bool, Optional[str]]:
    """Validate URL for SSRF protection.
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_safe, error_message)
    """
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            return False, f"Scheme '{parsed.scheme}' not allowed. Only {ALLOWED_SCHEMES} are permitted."
        
        # Check for file:// or other local schemes
        if parsed.scheme.lower() in {'file', 'ftp', 'ftps', 'gopher', 'ldap', 'ldaps'}:
            return False, f"Local/internal scheme '{parsed.scheme}' is blocked for security."
        
        # Check hostname
        hostname = parsed.hostname
        if not hostname:
            return False, "URL must have a valid hostname."
        
        # Check for localhost variations
        localhost_variants = {'localhost', '0.0.0.0', '0', 'local'}
        if hostname.lower() in localhost_variants:
            return False, f"Localhost hostname '{hostname}' is blocked for security."
        
        # Check port
        port = parsed.port
        if port and port in BLOCKED_PORTS:
            return False, f"Port {port} is blocked for security (internal service port)."
        
        # Check if hostname is an IP address
        try:
            ip = ipaddress.ip_address(hostname)
            if is_private_ip(str(ip)):
                return False, f"Private IP address '{hostname}' is blocked for security."
        except ValueError:
            # Not an IP address, need to resolve hostname
            try:
                resolve_hostname(hostname)
            except SSRFError as e:
                return False, str(e)
        
        # Check for suspicious patterns
        suspicious_patterns = [
            'metadata.google.internal',
            '169.254.169.254',  # AWS metadata service
            'metadata.azure.com',
            'metadata.packet.net',
        ]
        
        for pattern in suspicious_patterns:
            if pattern in hostname.lower():
                return False, f"Suspicious hostname pattern '{pattern}' detected."
        
        return True, None
        
    except Exception as e:
        return False, f"URL validation error: {e}"

def check_url_ssrf(url: str) -> None:
    """Check URL for SSRF vulnerabilities and raise exception if unsafe.
    
    Args:
        url: URL to check
        
    Raises:
        SSRFError: If URL is deemed unsafe
    """
    is_safe, error_msg = validate_url_security(url)
    if not is_safe:
        logger.warning(f"SSRF protection blocked URL: {url} - {error_msg}")
        raise SSRFError(f"URL blocked by SSRF protection: {error_msg}")

def get_safe_connector():
    """Get aiohttp connector with SSRF protection.
    
    Returns:
        aiohttp.TCPConnector configured for security
    """
    import aiohttp
    
    class SSRFProtectedConnector(aiohttp.TCPConnector):
        """Custom connector that validates connections for SSRF protection."""
        
        async def _resolve_host(self, host, port, traces=None):
            """Override host resolution to add SSRF checks."""
            # Validate the host before resolving
            try:
                # Check if host is an IP
                ip = ipaddress.ip_address(host)
                if is_private_ip(str(ip)):
                    raise SSRFError(f"Private IP address '{host}' is blocked")
            except ValueError:
                # Not an IP, validate hostname
                try:
                    resolve_hostname(host)
                except SSRFError:
                    raise
            
            # Check port
            if port in BLOCKED_PORTS:
                raise SSRFError(f"Port {port} is blocked for security")
            
            # Proceed with normal resolution
            return await super()._resolve_host(host, port, traces)
    
    return SSRFProtectedConnector()