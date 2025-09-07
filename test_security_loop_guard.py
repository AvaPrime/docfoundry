#!/usr/bin/env python3
"""Test script to verify the loop guard mechanism in security.py works correctly."""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.security import get_safe_connector

def test_loop_guard_from_sync_context():
    """Test that get_safe_connector works when called from synchronous context."""
    print("Testing loop guard mechanism...")
    
    try:
        # This should work without raising RuntimeError about event loop
        connector = get_safe_connector()
        print("✓ Successfully created connector from synchronous context")
        
        # Test with kwargs
        connector_with_kwargs = get_safe_connector(limit=100, ttl_dns_cache=300)
        print("✓ Successfully created connector with kwargs from synchronous context")
        
        # Clean up
        if hasattr(connector, 'close'):
            # Note: We can't actually close it here since it requires async context
            pass
            
        print("✓ All tests passed - loop guard mechanism working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_loop_guard_from_sync_context()
    sys.exit(0 if success else 1)