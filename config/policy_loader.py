"""Configuration loader for policy settings."""

import os
import yaml
from typing import Dict, Any, Optional, List, Set
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    'user_agent': 'DocFoundry/1.0',
    'robots_cache': {
        'ttl_hours': 24,
        'max_entries': 1000
    },
    'allowed_licenses': {
        'MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 
        'CC-BY-4.0', 'CC-BY-SA-4.0', 'ISC', 'Unlicense', '0BSD'
    },
    'content_filtering': {
        'respect_noai': True,
        'check_licenses': True,
        'strict_mode': False,
        'custom_noai_patterns': []
    },
    'crawl_settings': {
        'default_crawl_delay': 1.0,
        'max_crawl_delay': 30.0,
        'respect_crawl_delay': True,
        'robots_timeout': 10
    },
    'violation_handling': {
        'log_level': 'WARNING',
        'store_violations': True,
        'max_violations_per_url': 10
    },
    'source_overrides': {},
    'url_patterns': {
        'whitelist': [],
        'blacklist': []
    },
    'monitoring': {
        'collect_metrics': True,
        'metrics_retention_days': 30
    }
}


class PolicyConfig:
    """Policy configuration manager."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Look for config file in multiple locations
        possible_paths = [
            os.environ.get('DOCFOUNDRY_POLICY_CONFIG'),
            os.path.join(os.getcwd(), 'config', 'policy_config.yaml'),
            os.path.join(Path(__file__).parent, 'policy_config.yaml'),
            os.path.join(os.path.expanduser('~'), '.docfoundry', 'policy_config.yaml')
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        # Return the expected path even if it doesn't exist
        return os.path.join(Path(__file__).parent, 'policy_config.yaml')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        config = DEFAULT_CONFIG.copy()
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                
                # Deep merge configuration
                config = self._deep_merge(config, file_config)
                
            except Exception as e:
                print(f"Warning: Failed to load policy config from {self.config_path}: {e}")
                print("Using default configuration")
        else:
            print(f"Policy config file not found at {self.config_path}, using defaults")
        
        # Convert allowed_licenses to set if it's a list
        if isinstance(config.get('allowed_licenses'), list):
            config['allowed_licenses'] = set(config['allowed_licenses'])
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_user_agent(self) -> str:
        """Get the user agent string."""
        return self.get('user_agent', 'DocFoundry/1.0')
    
    def get_allowed_licenses(self, source_name: str = None) -> Set[str]:
        """Get allowed licenses, with optional source-specific overrides."""
        # Check for source-specific overrides
        if source_name:
            override_licenses = self.get(f'source_overrides.{source_name}.allowed_licenses')
            if override_licenses:
                return set(override_licenses) if isinstance(override_licenses, list) else override_licenses
        
        # Return default allowed licenses
        licenses = self.get('allowed_licenses', set())
        return licenses if isinstance(licenses, set) else set(licenses)
    
    def get_crawl_delay_settings(self) -> Dict[str, float]:
        """Get crawl delay settings."""
        return {
            'default': self.get('crawl_settings.default_crawl_delay', 1.0),
            'max': self.get('crawl_settings.max_crawl_delay', 30.0),
            'respect': self.get('crawl_settings.respect_crawl_delay', True)
        }
    
    def get_robots_timeout(self) -> int:
        """Get robots.txt request timeout."""
        return self.get('crawl_settings.robots_timeout', 10)
    
    def get_cache_settings(self) -> Dict[str, int]:
        """Get cache settings."""
        return {
            'ttl_hours': self.get('robots_cache.ttl_hours', 24),
            'max_entries': self.get('robots_cache.max_entries', 1000)
        }
    
    def should_respect_noai(self, source_name: str = None) -> bool:
        """Check if noai directives should be respected."""
        if source_name:
            override = self.get(f'source_overrides.{source_name}.respect_noai')
            if override is not None:
                return override
        
        return self.get('content_filtering.respect_noai', True)
    
    def should_check_licenses(self) -> bool:
        """Check if license compatibility should be checked."""
        return self.get('content_filtering.check_licenses', True)
    
    def is_strict_mode(self, source_name: str = None) -> bool:
        """Check if strict mode is enabled."""
        if source_name:
            override = self.get(f'source_overrides.{source_name}.strict_mode')
            if override is not None:
                return override
        
        return self.get('content_filtering.strict_mode', False)
    
    def get_custom_noai_patterns(self) -> List[str]:
        """Get custom noai patterns."""
        return self.get('content_filtering.custom_noai_patterns', [])
    
    def get_url_whitelist(self) -> List[str]:
        """Get URL whitelist patterns."""
        return self.get('url_patterns.whitelist', [])
    
    def get_url_blacklist(self) -> List[str]:
        """Get URL blacklist patterns."""
        return self.get('url_patterns.blacklist', [])
    
    def should_collect_metrics(self) -> bool:
        """Check if metrics collection is enabled."""
        return self.get('monitoring.collect_metrics', True)
    
    def get_metrics_retention_days(self) -> int:
        """Get metrics retention period in days."""
        return self.get('monitoring.metrics_retention_days', 30)
    
    def get_crawl_settings(self, source_name: str = None) -> Dict[str, Any]:
        """Get crawl settings with source-specific overrides."""
        base_settings = self.get('crawl_settings', {})
        
        if source_name:
            override_settings = self.get(f'source_overrides.{source_name}.crawl_settings')
            if override_settings:
                return {**base_settings, **override_settings}
        
        return base_settings
    
    def get_url_patterns(self) -> Dict[str, List[str]]:
        """Get URL whitelist and blacklist patterns."""
        return self.get('url_patterns', {'whitelist': [], 'blacklist': []})
    
    def should_log_violations(self) -> bool:
        """Check if violations should be logged."""
        return self.get('violation_handling.log_violations', True)
    
    def should_store_violations(self) -> bool:
        """Check if violations should be stored."""
        return self.get('violation_handling.store_violations', True)
    
    def get_allowed_licenses(self, source_name: str = None) -> Set[str]:
        """Get allowed licenses with source-specific overrides."""
        base_licenses = set(self.get('allowed_licenses', []))
        
        if source_name:
            override_licenses = self.get(f'source_overrides.{source_name}.allowed_licenses')
            if override_licenses is not None:
                return set(override_licenses)
        
        return base_licenses
    
    def should_respect_noai(self, source_name: str = None) -> bool:
        """Check if noai directives should be respected for a source."""
        base_setting = self.get('content_filtering.respect_noai', True)
        
        if source_name:
            override = self.get(f'source_overrides.{source_name}.content_filtering.respect_noai')
            if override is not None:
                return override
        
        return base_setting
    
    def is_strict_mode(self, source_name: str = None) -> bool:
        """Check if strict mode is enabled for a source."""
        base_setting = self.get('content_filtering.strict_mode', False)
        
        if source_name:
            override = self.get(f'source_overrides.{source_name}.content_filtering.strict_mode')
            if override is not None:
                return override
        
        return base_setting
    
    def should_check_licenses(self) -> bool:
        """Check if license checking is enabled."""
        return self.get('content_filtering.check_licenses', True)
    
    def get_custom_noai_patterns(self) -> List[str]:
        """Get custom noai patterns."""
        return self.get('content_filtering.custom_noai_patterns', [])
    
    def reload(self):
        """Reload configuration from file."""
        self._config = self._load_config()


# Global configuration instance
policy_config = PolicyConfig()