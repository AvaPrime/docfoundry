"""Source configuration loader for DocFoundry.

Loads and validates source configurations from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SourceConfig:
    """Configuration for a document source."""
    name: str
    base_urls: List[str]
    sitemaps: Optional[List[str]] = None
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    rate_limit: float = 0.5
    backoff: Optional[Dict[str, float]] = None
    depth: int = 4
    priority: str = "normal"
    auth: Optional[Dict[str, Any]] = None
    license_hint: Optional[str] = None
    enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Source name cannot be empty")
        
        if not self.base_urls:
            raise ValueError("Source must have at least one base URL")
        
        if self.priority not in ["low", "normal", "high"]:
            raise ValueError(f"Invalid priority: {self.priority}")
        
        if self.rate_limit <= 0:
            raise ValueError("Rate limit must be positive")
        
        if self.depth < 0 or self.depth > 10:
            raise ValueError("Depth must be between 0 and 10")
        
        # Set default backoff if not provided
        if self.backoff is None:
            self.backoff = {"base": 0.25, "max": 8}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceConfig':
        """Create SourceConfig from dictionary."""
        return cls(
            name=data['name'],
            base_urls=data['base_urls'],
            sitemaps=data.get('sitemaps'),
            include=data.get('include'),
            exclude=data.get('exclude'),
            rate_limit=data.get('rate_limit', 0.5),
            backoff=data.get('backoff'),
            depth=data.get('depth', 4),
            priority=data.get('priority', 'normal'),
            auth=data.get('auth'),
            license_hint=data.get('license_hint'),
            enabled=data.get('enabled', True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'name': self.name,
            'base_urls': self.base_urls,
            'rate_limit': self.rate_limit,
            'depth': self.depth,
            'priority': self.priority,
            'enabled': self.enabled
        }
        
        if self.sitemaps:
            result['sitemaps'] = self.sitemaps
        if self.include:
            result['include'] = self.include
        if self.exclude:
            result['exclude'] = self.exclude
        if self.backoff:
            result['backoff'] = self.backoff
        if self.auth:
            result['auth'] = self.auth
        if self.license_hint:
            result['license_hint'] = self.license_hint
        
        return result

class SourceLoader:
    """Loads source configurations from YAML files."""
    
    def __init__(self, sources_dir: Optional[Path] = None):
        """Initialize source loader.
        
        Args:
            sources_dir: Directory containing source YAML files.
                        Defaults to 'sources' directory relative to this file.
        """
        if sources_dir is None:
            sources_dir = Path(__file__).parent
        
        self.sources_dir = Path(sources_dir)
        self._cache: Dict[str, SourceConfig] = {}
        self._last_modified: Dict[str, float] = {}
    
    def load_source_config(self, source_name: str) -> Optional[SourceConfig]:
        """Load configuration for a specific source.
        
        Args:
            source_name: Name of the source (without .yaml extension)
        
        Returns:
            SourceConfig if found and valid, None otherwise
        """
        yaml_file = self.sources_dir / f"{source_name}.yaml"
        
        if not yaml_file.exists():
            logger.warning(f"Source configuration not found: {yaml_file}")
            return None
        
        # Check if we need to reload from cache
        current_mtime = yaml_file.stat().st_mtime
        if (source_name in self._cache and 
            source_name in self._last_modified and
            self._last_modified[source_name] >= current_mtime):
            return self._cache[source_name]
        
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                logger.error(f"Empty or invalid YAML file: {yaml_file}")
                return None
            
            # Ensure name matches filename
            if 'name' not in data:
                data['name'] = source_name
            elif data['name'] != source_name:
                logger.warning(f"Source name mismatch in {yaml_file}: {data['name']} != {source_name}")
                data['name'] = source_name
            
            config = SourceConfig.from_dict(data)
            
            # Cache the configuration
            self._cache[source_name] = config
            self._last_modified[source_name] = current_mtime
            
            logger.info(f"Loaded source configuration: {source_name}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {yaml_file}: {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid source configuration in {yaml_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {yaml_file}: {e}")
            return None
    
    def load_all_sources(self) -> Dict[str, SourceConfig]:
        """Load all source configurations from the sources directory.
        
        Returns:
            Dictionary mapping source names to SourceConfig objects
        """
        sources = {}
        
        if not self.sources_dir.exists():
            logger.warning(f"Sources directory not found: {self.sources_dir}")
            return sources
        
        for yaml_file in self.sources_dir.glob("*.yaml"):
            source_name = yaml_file.stem
            config = self.load_source_config(source_name)
            if config:
                sources[source_name] = config
        
        logger.info(f"Loaded {len(sources)} source configurations")
        return sources
    
    def get_enabled_sources(self) -> Dict[str, SourceConfig]:
        """Get all enabled source configurations.
        
        Returns:
            Dictionary mapping source names to enabled SourceConfig objects
        """
        all_sources = self.load_all_sources()
        return {name: config for name, config in all_sources.items() if config.enabled}
    
    def reload_cache(self):
        """Clear cache to force reload of all configurations."""
        self._cache.clear()
        self._last_modified.clear()
        logger.info("Source configuration cache cleared")

# Global source loader instance
_source_loader = SourceLoader()

def load_source_config(source_name: str) -> Optional[SourceConfig]:
    """Convenience function to load a source configuration.
    
    Args:
        source_name: Name of the source
    
    Returns:
        SourceConfig if found, None otherwise
    """
    return _source_loader.load_source_config(source_name)

def load_all_sources() -> Dict[str, SourceConfig]:
    """Convenience function to load all source configurations.
    
    Returns:
        Dictionary mapping source names to SourceConfig objects
    """
    return _source_loader.load_all_sources()

def get_enabled_sources() -> Dict[str, SourceConfig]:
    """Convenience function to get enabled source configurations.
    
    Returns:
        Dictionary mapping source names to enabled SourceConfig objects
    """
    return _source_loader.get_enabled_sources()

def reload_source_cache():
    """Convenience function to reload source configuration cache."""
    _source_loader.reload_cache()