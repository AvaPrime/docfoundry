#!/usr/bin/env python3
"""
Simple verification script for SSRF protection functionality
"""

import sys
import traceback
from pipelines.security import check_url_ssrf, SSRFError

def test_ssrf_protection():
    """Test SSRF protection with various URL patterns"""
    print("Testing SSRF Protection...")
    
    # Test cases: (url, should_be_blocked, description)
    test_cases = [
        ("http://127.0.0.1/test", True, "localhost IP"),
        ("http://localhost/test", True, "localhost domain"),
        ("http://10.0.0.1/test", True, "private IP 10.x"),
        ("http://192.168.1.1/test", True, "private IP 192.168.x"),
        ("file:///etc/passwd", True, "file protocol"),
        ("https://example.com/docs", False, "valid public URL"),
        ("https://github.com/user/repo", False, "valid GitHub URL"),
        ("ftp://example.com/file", True, "unsupported FTP scheme"),
    ]
    
    passed = 0
    failed = 0
    
    for url, should_be_blocked, description in test_cases:
        try:
            check_url_ssrf(url)
            # If we reach here, URL was allowed
            if should_be_blocked:
                print(f"❌ FAIL: {description} - URL should be blocked but was allowed: {url}")
                failed += 1
            else:
                print(f"✅ PASS: {description} - URL correctly allowed: {url}")
                passed += 1
        except SSRFError as e:
            # URL was blocked
            if should_be_blocked:
                print(f"✅ PASS: {description} - URL correctly blocked: {url} ({e})")
                passed += 1
            else:
                print(f"❌ FAIL: {description} - URL should be allowed but was blocked: {url} ({e})")
                failed += 1
        except Exception as e:
            print(f"❌ ERROR: {description} - Unexpected error for {url}: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

def test_crawler_integration():
    """Test SSRF protection integration with crawler"""
    print("\nTesting Crawler Integration...")
    
    try:
        from pipelines.crawler import WebCrawler
        print("✅ WebCrawler import successful")
        
        # Test crawler initialization
        crawler = WebCrawler()
        print("✅ WebCrawler initialization successful")
        
        # Test that crawler has session attribute
        if hasattr(crawler, 'session'):
            print("✅ WebCrawler has session attribute")
        else:
            print("❌ WebCrawler missing session attribute")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Crawler integration test failed: {e}")
        traceback.print_exc()
        return False

def test_security_module():
    """Test security module components"""
    print("\nTesting Security Module...")
    
    try:
        from pipelines.security import PRIVATE_IP_RANGES, BLOCKED_PORTS, get_safe_connector
        print(f"✅ Security constants loaded: {len(PRIVATE_IP_RANGES)} IP ranges, {len(BLOCKED_PORTS)} blocked ports")
        
        # Test safe connector
        connector = get_safe_connector()
        print(f"✅ Safe connector created: {type(connector).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Security module test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    print("DocFoundry SSRF Protection Verification")
    print("=" * 40)
    
    all_passed = True
    
    # Test SSRF protection
    if not test_ssrf_protection():
        all_passed = False
    
    # Test security module
    if not test_security_module():
        all_passed = False
    
    # Test crawler integration
    if not test_crawler_integration():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! SSRF protection is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())