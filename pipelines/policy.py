"""Policy compliance module for robots.txt parsing and content filtering."""

import logging
import re
import urllib.robotparser
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add config to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.policy_loader import policy_config

logger = logging.getLogger(__name__)


@dataclass
class PolicyViolation:
    """Represents a policy violation."""
    url: str
    violation_type: str
    reason: str
    severity: str  # 'error', 'warning', 'info'
    timestamp: datetime


@dataclass
class RobotsCache:
    """Cache entry for robots.txt data."""
    robots_parser: urllib.robotparser.RobotFileParser
    fetched_at: datetime
    ttl_hours: int = 24
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.now() - self.fetched_at > timedelta(hours=self.ttl_hours)


class PolicyChecker:
    """Handles robots.txt parsing and content filtering for compliance."""
    
    def __init__(self, user_agent: str = None):
        self.user_agent = user_agent or policy_config.get_user_agent()
        self.robots_cache: Dict[str, RobotsCache] = {}
        self.violations: List[PolicyViolation] = []
        
        # Load content filtering patterns from config
        base_noai_patterns = [
            r'<meta\s+name=["\']robots["\']\s+content=["\'][^"\'>]*noai[^"\'>]*["\']',
            r'<meta\s+name=["\']googlebot["\']\s+content=["\'][^"\'>]*noai[^"\'>]*["\']',
            r'<!--\s*noai\s*-->',
            r'data-noai=["\']true["\']',
        ]
        custom_patterns = policy_config.get_custom_noai_patterns()
        self.noai_patterns = base_noai_patterns + custom_patterns
        
        # License detection patterns (basic SPDX matching)
        self.license_patterns = {
            'MIT': r'MIT\s+License|Permission\s+is\s+hereby\s+granted.*MIT',
            'Apache-2.0': r'Apache\s+License.*Version\s+2\.0',
            'GPL-3.0': r'GNU\s+General\s+Public\s+License.*version\s+3',
            'BSD-3-Clause': r'BSD\s+3-Clause|Redistribution\s+and\s+use.*BSD',
            'CC-BY-4.0': r'Creative\s+Commons.*Attribution\s+4\.0',
            'CC-BY-SA-4.0': r'Creative\s+Commons.*Attribution-ShareAlike\s+4\.0',
        }
        
        # Load URL patterns from config
        self.url_whitelist = [re.compile(pattern, re.IGNORECASE) for pattern in policy_config.get_url_whitelist()]
        self.url_blacklist = [re.compile(pattern, re.IGNORECASE) for pattern in policy_config.get_url_blacklist()]
    
    def get_robots_txt_url(self, url: str) -> str:
        """Get the robots.txt URL for a given URL."""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        return urljoin(base_url, '/robots.txt')
    
    def get_domain_key(self, url: str) -> str:
        """Extract domain key for caching."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def is_url_whitelisted(self, url: str) -> bool:
        """Check if URL matches whitelist patterns."""
        for pattern in self.url_whitelist:
            if pattern.search(url):
                logger.debug(f"URL {url} matches whitelist pattern: {pattern.pattern}")
                return True
        return False
    
    def is_url_blacklisted(self, url: str) -> bool:
        """Check if URL matches blacklist patterns."""
        for pattern in self.url_blacklist:
            if pattern.search(url):
                logger.debug(f"URL {url} matches blacklist pattern: {pattern.pattern}")
                return True
        return False
    
    async def fetch_robots_txt(self, url: str) -> Optional[urllib.robotparser.RobotFileParser]:
        """Fetch and parse robots.txt for a given URL."""
        domain_key = self.get_domain_key(url)
        
        # Check cache first
        if domain_key in self.robots_cache:
            cache_entry = self.robots_cache[domain_key]
            if not cache_entry.is_expired():
                logger.debug(f"Using cached robots.txt for {domain_key}")
                return cache_entry.robots_parser
        
        robots_url = self.get_robots_txt_url(url)
        
        try:
            logger.info(f"Fetching robots.txt from {robots_url}")
            
            # Use requests for better control
            response = requests.get(
                robots_url,
                headers={'User-Agent': self.user_agent},
                timeout=10,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                # Create and configure robots parser
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                
                # Parse the content manually since we fetched it with requests
                lines = response.text.splitlines()
                rp.set_url(robots_url)
                
                # Store in cache
                self.robots_cache[domain_key] = RobotsCache(
                    robots_parser=rp,
                    fetched_at=datetime.now()
                )
                
                logger.info(f"Successfully fetched and cached robots.txt for {domain_key}")
                return rp
            
            elif response.status_code == 404:
                logger.info(f"No robots.txt found for {domain_key} (404)")
                # Create empty robots parser (allows everything)
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                
                # Cache the "no robots.txt" result
                self.robots_cache[domain_key] = RobotsCache(
                    robots_parser=rp,
                    fetched_at=datetime.now()
                )
                return rp
            
            else:
                logger.warning(f"Failed to fetch robots.txt from {robots_url}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching robots.txt from {robots_url}: {e}")
            return None
    
    async def can_fetch(self, url: str, user_agent: str = None, source_name: str = None) -> Tuple[bool, Optional[str]]:
        """Check if a URL can be fetched according to robots.txt and URL patterns.
        
        Returns:
            Tuple of (can_fetch: bool, reason: Optional[str])
        """
        if user_agent is None:
            user_agent = self.user_agent
        
        # Check blacklist first
        if self.is_url_blacklisted(url):
            reason = "URL matches blacklist pattern"
            self._add_violation(
                url=url,
                violation_type="url_blacklisted",
                reason=reason,
                severity="error"
            )
            return False, reason
        
        # Check whitelist - if whitelisted, skip robots.txt check
        if self.is_url_whitelisted(url):
            logger.info(f"URL {url} is whitelisted, skipping robots.txt check")
            return True, None
        
        robots_parser = await self.fetch_robots_txt(url)
        
        if robots_parser is None:
            # If we can't fetch robots.txt, assume we can fetch (be permissive)
            return True, None
        
        try:
            can_fetch = robots_parser.can_fetch(user_agent, url)
            
            if not can_fetch:
                reason = f"Disallowed by robots.txt for user-agent '{user_agent}'"
                self._add_violation(
                    url=url,
                    violation_type="robots_disallow",
                    reason=reason,
                    severity="error"
                )
                return False, reason
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            # If there's an error, be permissive
            return True, None
    
    def check_noai_compliance(self, content: str, url: str, source_name: str = None) -> bool:
        """Check if content contains noai directives.
        
        Returns:
            True if content is compliant (no noai restrictions), False otherwise
        """
        # Check if we should respect noai directives for this source
        if not policy_config.should_respect_noai(source_name):
            logger.debug(f"NoAI compliance disabled for source {source_name}")
            return True
        
        for pattern in self.noai_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                reason = f"Content contains noai directive matching pattern: {pattern}"
                severity = "error" if policy_config.is_strict_mode(source_name) else "warning"
                self._add_violation(
                    url=url,
                    violation_type="noai_directive",
                    reason=reason,
                    severity=severity
                )
                logger.warning(f"NoAI directive found in {url}: {reason}")
                
                # In strict mode, noai violations block content
                if policy_config.is_strict_mode(source_name):
                    return False
        
        return True
    
    def detect_license(self, content: str) -> Optional[str]:
        """Detect license in content using SPDX patterns.
        
        Returns:
            License identifier if detected, None otherwise
        """
        for license_id, pattern in self.license_patterns.items():
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                logger.info(f"Detected license: {license_id}")
                return license_id
        
        return None
    
    def is_license_compatible(self, license_id: str, allowed_licenses: Set[str]) -> bool:
        """Check if a license is in the allowed set.
        
        Args:
            license_id: SPDX license identifier
            allowed_licenses: Set of allowed license identifiers
        
        Returns:
            True if license is allowed, False otherwise
        """
        if not license_id:
            return True  # Unknown license - let it through with warning
        
        return license_id in allowed_licenses
    
    def check_content_policy(self, content: str, url: str, 
                           allowed_licenses: Optional[Set[str]] = None,
                           source_name: str = None) -> Dict[str, any]:
        """Comprehensive content policy check.
        
        Args:
            content: HTML/text content to check
            url: URL of the content
            allowed_licenses: Set of allowed SPDX license identifiers
            source_name: Source name for configuration overrides
        
        Returns:
            Dict with policy check results
        """
        results = {
            'compliant': True,
            'violations': [],
            'license': None,
            'license_compatible': True,
            'noai_compliant': True
        }
        
        # Use source-specific allowed licenses if not provided
        if allowed_licenses is None:
            allowed_licenses = policy_config.get_allowed_licenses(source_name)
        
        # Check noai compliance
        results['noai_compliant'] = self.check_noai_compliance(content, url, source_name)
        if not results['noai_compliant'] and policy_config.is_strict_mode(source_name):
            results['compliant'] = False
        
        # Detect license if license checking is enabled
        if policy_config.should_check_licenses():
            license_id = self.detect_license(content)
            results['license'] = license_id
            
            # Check license compatibility
            if allowed_licenses and license_id:
                results['license_compatible'] = self.is_license_compatible(license_id, allowed_licenses)
                if not results['license_compatible']:
                    reason = f"License {license_id} not in allowed set: {allowed_licenses}"
                    severity = "error" if policy_config.is_strict_mode(source_name) else "warning"
                    self._add_violation(
                        url=url,
                        violation_type="license_incompatible",
                        reason=reason,
                        severity=severity
                    )
                    if policy_config.is_strict_mode(source_name):
                        results['compliant'] = False
        
        # Add current violations to results
        results['violations'] = [v for v in self.violations if v.url == url]
        
        return results
    
    def _add_violation(self, url: str, violation_type: str, reason: str, severity: str):
        """Add a policy violation to the list."""
        violation = PolicyViolation(
            url=url,
            violation_type=violation_type,
            reason=reason,
            severity=severity,
            timestamp=datetime.now()
        )
        self.violations.append(violation)
    
    def get_violations(self, url: Optional[str] = None) -> List[PolicyViolation]:
        """Get policy violations, optionally filtered by URL."""
        if url:
            return [v for v in self.violations if v.url == url]
        return self.violations.copy()
    
    def clear_violations(self, url: Optional[str] = None):
        """Clear policy violations, optionally filtered by URL."""
        if url:
            self.violations = [v for v in self.violations if v.url != url]
        else:
            self.violations.clear()
    
    def get_crawl_delay(self, url: str, user_agent: str = None) -> Optional[float]:
        """Get crawl delay from robots.txt.
        
        Returns:
            Crawl delay in seconds, or None if not specified
        """
        if user_agent is None:
            user_agent = self.user_agent
        
        domain_key = self.get_domain_key(url)
        
        if domain_key in self.robots_cache:
            cache_entry = self.robots_cache[domain_key]
            if not cache_entry.is_expired():
                try:
                    return cache_entry.robots_parser.crawl_delay(user_agent)
                except Exception as e:
                    logger.error(f"Error getting crawl delay for {url}: {e}")
        
        return None


# Global policy checker instance
policy_checker = PolicyChecker()


async def check_url_policy(url: str, content: str = None, 
                          allowed_licenses: Optional[Set[str]] = None,
                          source_name: str = None) -> Dict[str, any]:
    """Convenience function to check URL policy compliance.
    
    Args:
        url: URL to check
        content: Optional content to analyze
        allowed_licenses: Set of allowed SPDX license identifiers
        source_name: Source name for configuration overrides
    
    Returns:
        Dict with comprehensive policy check results
    """
    results = {
        'url': url,
        'source_name': source_name,
        'robots_compliant': True,
        'robots_reason': None,
        'content_policy': None,
        'crawl_delay': None,
        'overall_compliant': True,
        'whitelisted': False,
        'blacklisted': False
    }
    
    # Check URL patterns
    results['whitelisted'] = policy_checker.is_url_whitelisted(url)
    results['blacklisted'] = policy_checker.is_url_blacklisted(url)
    
    # Check robots.txt compliance
    can_fetch, reason = await policy_checker.can_fetch(url, source_name=source_name)
    results['robots_compliant'] = can_fetch
    results['robots_reason'] = reason
    
    if not can_fetch:
        results['overall_compliant'] = False
    
    # Get crawl delay
    results['crawl_delay'] = policy_checker.get_crawl_delay(url)
    
    # Check content policy if content is provided
    if content:
        content_results = policy_checker.check_content_policy(content, url, allowed_licenses, source_name)
        results['content_policy'] = content_results
        
        if not content_results['compliant']:
            results['overall_compliant'] = False
    
    return results