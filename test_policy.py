#!/usr/bin/env python3
"""
Test script for policy implementation.
Tests robots.txt parsing, content filtering, and license detection.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.policy import check_url_policy, policy_checker
from config.policy_loader import policy_config

async def test_robots_compliance():
    """Test robots.txt compliance checking."""
    print("\n=== Testing Robots.txt Compliance ===")
    
    # Test URLs with known robots.txt behavior
    test_urls = [
        "https://www.google.com/search",  # Should be disallowed
        "https://www.google.com/",        # Should be allowed
        "https://httpbin.org/robots.txt", # Test endpoint
        "https://example.com/",           # Simple test
    ]
    
    for url in test_urls:
        try:
            result = await check_url_policy(url)
            print(f"URL: {url}")
            print(f"  Robots compliant: {result['robots_compliant']}")
            print(f"  Reason: {result['robots_reason']}")
            print(f"  Crawl delay: {result['crawl_delay']}")
            print(f"  Whitelisted: {result['whitelisted']}")
            print(f"  Blacklisted: {result['blacklisted']}")
            print()
        except Exception as e:
            print(f"Error testing {url}: {e}")

def test_content_filtering():
    """Test content filtering and license detection."""
    print("\n=== Testing Content Filtering ===")
    
    # Test content with noai directive
    noai_content = """
    <html>
    <head>
        <meta name="robots" content="noai, noindex">
        <title>Test Page</title>
    </head>
    <body>
        <p>This content should not be used for AI training.</p>
    </body>
    </html>
    """
    
    # Test content with MIT license
    mit_content = """
    MIT License
    
    Copyright (c) 2024 Test Project
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files...
    """
    
    # Test content with GPL license
    gpl_content = """
    GNU GENERAL PUBLIC LICENSE
    Version 3, 29 June 2007
    
    Copyright (C) 2024 Test Project
    This program is free software...
    """
    
    test_cases = [
        ("NoAI Content", noai_content, "https://example.com/noai"),
        ("MIT Licensed Content", mit_content, "https://example.com/mit"),
        ("GPL Licensed Content", gpl_content, "https://example.com/gpl"),
    ]
    
    for name, content, url in test_cases:
        print(f"\nTesting: {name}")
        try:
            result = policy_checker.check_content_policy(content, url)
            print(f"  Overall compliant: {result['compliant']}")
            print(f"  NoAI compliant: {result['noai_compliant']}")
            print(f"  License detected: {result['license']}")
            print(f"  License compatible: {result['license_compatible']}")
            
            if result['violations']:
                print(f"  Violations:")
                for violation in result['violations']:
                    print(f"    - {violation.violation_type}: {violation.reason} ({violation.severity})")
        except Exception as e:
            print(f"  Error: {e}")

def test_configuration():
    """Test policy configuration loading."""
    print("\n=== Testing Configuration ===")
    
    print(f"User Agent: {policy_config.get_user_agent()}")
    print(f"Allowed Licenses: {policy_config.get_allowed_licenses()}")
    print(f"Should Check Licenses: {policy_config.should_check_licenses()}")
    print(f"Should Respect NoAI: {policy_config.should_respect_noai()}")
    print(f"Strict Mode: {policy_config.is_strict_mode()}")
    print(f"Custom NoAI Patterns: {policy_config.get_custom_noai_patterns()}")
    print(f"URL Patterns: {policy_config.get_url_patterns()}")
    
    # Test source-specific overrides
    print("\nTesting source-specific settings:")
    test_source = "test-source"
    print(f"Allowed Licenses for '{test_source}': {policy_config.get_allowed_licenses(test_source)}")
    print(f"Strict Mode for '{test_source}': {policy_config.is_strict_mode(test_source)}")

async def main():
    """Run all policy tests."""
    print("DocFoundry Policy Implementation Test")
    print("====================================")
    
    # Test configuration loading
    test_configuration()
    
    # Test content filtering
    test_content_filtering()
    
    # Test robots.txt compliance (requires network)
    try:
        await test_robots_compliance()
    except Exception as e:
        print(f"Robots.txt testing failed (network required): {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(main())