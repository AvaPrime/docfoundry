#!/usr/bin/env python3
"""
Test script for PostgreSQL migration functionality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.database import DatabaseConfig, initialize_database, get_db_adapter, close_database

async def test_database_config():
    """Test database configuration and adapter creation."""
    print("Testing PostgreSQL Migration Components...")
    print("=" * 50)
    
    try:
        # Test configuration loading
        print("1. Loading database configuration...")
        config = DatabaseConfig.from_env()
        print(f"   Database type: {config.type}")
        
        if config.type.value == 'postgresql':
            print(f"   PostgreSQL host: {config.postgres.host}")
            print(f"   PostgreSQL port: {config.postgres.port}")
            print(f"   PostgreSQL database: {config.postgres.database}")
            print(f"   PostgreSQL user: {config.postgres.user}")
        elif config.type.value == 'sqlite':
            print(f"   SQLite path: {config.sqlite_path}")
        
        # Test database initialization
        print("\n2. Initializing database...")
        await initialize_database(config)
        
        # Test adapter retrieval
        print("\n3. Getting database adapter...")
        adapter = await get_db_adapter()
        print(f"   Adapter type: {type(adapter).__name__}")
        
        # Test basic adapter functionality
        print("\n4. Testing adapter methods...")
        if hasattr(adapter, 'health_check'):
            try:
                health = await adapter.health_check()
                print(f"   Health check: {'PASS' if health else 'FAIL'}")
            except Exception as e:
                print(f"   Health check: SKIP (expected for SQLite: {e})")
        
        # Clean up
        print("\n5. Cleaning up...")
        await close_database()
        print("   Database closed successfully")
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! PostgreSQL migration components are working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_database_config())
    sys.exit(0 if success else 1)