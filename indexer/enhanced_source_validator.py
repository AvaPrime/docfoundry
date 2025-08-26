"""Enhanced Source Schema Validation for DocFoundry

This module provides comprehensive validation for source configurations with
advanced error reporting, schema evolution support, and integration testing.
"""

import json
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Detailed validation error with context."""
    
    field: str
    message: str
    severity: str = "error"  # error, warning, info
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggestion: Optional[str] = None
    error_code: Optional[str] = None
    
    def __str__(self) -> str:
        location = ""
        if self.line_number is not None:
            location = f" (line {self.line_number}"
            if self.column_number is not None:
                location += f", col {self.column_number}"
            location += ")"
        
        result = f"[{self.severity.upper()}] {self.field}: {self.message}{location}"
        if self.suggestion:
            result += f"\n  Suggestion: {self.suggestion}"
        return result


@dataclass
class ValidationResult:
    """Result of source validation."""
    
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: ValidationError):
        """Add a validation error."""
        if error.severity == "error":
            self.errors.append(error)
            self.is_valid = False
        elif error.severity == "warning":
            self.warnings.append(error)
        else:
            self.info.append(error)
    
    def get_all_issues(self) -> List[ValidationError]:
        """Get all validation issues sorted by severity."""
        return self.errors + self.warnings + self.info
    
    def format_report(self) -> str:
        """Format a human-readable validation report."""
        lines = []
        
        if self.is_valid:
            lines.append("✅ Validation PASSED")
        else:
            lines.append("❌ Validation FAILED")
        
        lines.append(f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}, Info: {len(self.info)}")
        lines.append("")
        
        for issue in self.get_all_issues():
            lines.append(str(issue))
        
        return "\n".join(lines)


class EnhancedSourceValidator:
    """Enhanced source configuration validator with comprehensive checks."""
    
    # Schema version compatibility
    SUPPORTED_SCHEMA_VERSIONS = ["1.0", "1.1", "1.2"]
    CURRENT_SCHEMA_VERSION = "1.2"
    
    # URL validation patterns
    URL_PATTERN = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
        r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # host...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    # Common crawling patterns to validate
    ROBOTS_TXT_PATHS = ["/robots.txt", "/robot.txt"]
    SITEMAP_PATTERNS = ["sitemap.xml", "sitemap_index.xml", "sitemap"]
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules for different schema versions."""
        return {
            "1.0": {
                "required_fields": ["name", "base_urls"],
                "optional_fields": ["sitemaps", "include", "exclude", "rate_limit", "depth", "priority", "auth", "license_hint"],
                "field_types": {
                    "name": str,
                    "base_urls": list,
                    "sitemaps": list,
                    "include": list,
                    "exclude": list,
                    "rate_limit": (int, float),
                    "depth": int,
                    "priority": int,
                    "auth": dict,
                    "license_hint": str
                }
            },
            "1.1": {
                "required_fields": ["name", "base_urls"],
                "optional_fields": ["sitemaps", "include", "exclude", "rate_limit", "depth", "priority", "auth", "license_hint", "headers", "timeout"],
                "field_types": {
                    "name": str,
                    "base_urls": list,
                    "sitemaps": list,
                    "include": list,
                    "exclude": list,
                    "rate_limit": (int, float),
                    "depth": int,
                    "priority": int,
                    "auth": dict,
                    "license_hint": str,
                    "headers": dict,
                    "timeout": (int, float)
                }
            },
            "1.2": {
                "required_fields": ["name", "base_urls"],
                "optional_fields": ["sitemaps", "include", "exclude", "rate_limit", "depth", "priority", "auth", "license_hint", "headers", "timeout", "retry_config", "content_filters"],
                "field_types": {
                    "name": str,
                    "base_urls": list,
                    "sitemaps": list,
                    "include": list,
                    "exclude": list,
                    "rate_limit": (int, float),
                    "depth": int,
                    "priority": int,
                    "auth": dict,
                    "license_hint": str,
                    "headers": dict,
                    "timeout": (int, float),
                    "retry_config": dict,
                    "content_filters": dict
                }
            }
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """Validate a source configuration file."""
        file_path = Path(file_path)
        result = ValidationResult(is_valid=True)
        
        # Check file existence
        if not file_path.exists():
            result.add_error(ValidationError(
                field="file",
                message=f"Source file does not exist: {file_path}",
                error_code="FILE_NOT_FOUND"
            ))
            return result
        
        # Check file extension
        if file_path.suffix.lower() not in [".yaml", ".yml"]:
            result.add_error(ValidationError(
                field="file",
                message=f"Unsupported file extension: {file_path.suffix}. Expected .yaml or .yml",
                severity="warning",
                suggestion="Rename file with .yaml or .yml extension",
                error_code="INVALID_EXTENSION"
            ))
        
        try:
            # Load and parse YAML
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = yaml.safe_load(content)
            
            if data is None:
                result.add_error(ValidationError(
                    field="content",
                    message="File is empty or contains only comments",
                    error_code="EMPTY_FILE"
                ))
                return result
            
            # Validate the loaded data
            self._validate_source_config(data, result, file_path)
            
        except yaml.YAMLError as e:
            result.add_error(ValidationError(
                field="yaml",
                message=f"YAML parsing error: {str(e)}",
                line_number=getattr(e, 'problem_mark', {}).get('line'),
                column_number=getattr(e, 'problem_mark', {}).get('column'),
                error_code="YAML_PARSE_ERROR"
            ))
        except Exception as e:
            result.add_error(ValidationError(
                field="file",
                message=f"Unexpected error reading file: {str(e)}",
                error_code="FILE_READ_ERROR"
            ))
        
        return result
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate a source configuration dictionary."""
        result = ValidationResult(is_valid=True)
        self._validate_source_config(config, result)
        return result
    
    def _validate_source_config(self, config: Dict[str, Any], result: ValidationResult, file_path: Optional[Path] = None):
        """Validate source configuration data."""
        # Determine schema version
        schema_version = config.get("schema_version", "1.0")
        
        if schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            result.add_error(ValidationError(
                field="schema_version",
                message=f"Unsupported schema version: {schema_version}. Supported versions: {', '.join(self.SUPPORTED_SCHEMA_VERSIONS)}",
                suggestion=f"Update to current version: {self.CURRENT_SCHEMA_VERSION}",
                error_code="UNSUPPORTED_SCHEMA"
            ))
            schema_version = "1.0"  # Fallback to basic validation
        
        rules = self.validation_rules[schema_version]
        
        # Check required fields
        for field in rules["required_fields"]:
            if field not in config:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    suggestion=f"Add '{field}' field to the configuration",
                    error_code="MISSING_REQUIRED_FIELD"
                ))
        
        # Validate field types and values
        for field, value in config.items():
            if field in rules["field_types"]:
                expected_type = rules["field_types"][field]
                if not isinstance(value, expected_type):
                    result.add_error(ValidationError(
                        field=field,
                        message=f"Field '{field}' has incorrect type. Expected {expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)}, got {type(value).__name__}",
                        error_code="INVALID_FIELD_TYPE"
                    ))
                    continue
                
                # Field-specific validation
                self._validate_field_value(field, value, result)
            elif field not in ["schema_version"] + rules["required_fields"] + rules["optional_fields"]:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Unknown field '{field}' for schema version {schema_version}",
                    severity="warning",
                    suggestion="Remove unknown field or check schema version",
                    error_code="UNKNOWN_FIELD"
                ))
        
        # Cross-field validation
        self._validate_cross_field_constraints(config, result)
        
        # Performance and security recommendations
        self._validate_performance_settings(config, result)
        self._validate_security_settings(config, result)
    
    def _validate_field_value(self, field: str, value: Any, result: ValidationResult):
        """Validate specific field values."""
        if field == "name":
            if not value or not value.strip():
                result.add_error(ValidationError(
                    field=field,
                    message="Source name cannot be empty",
                    error_code="EMPTY_NAME"
                ))
            elif len(value) > 100:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Source name too long ({len(value)} chars). Maximum 100 characters",
                    severity="warning",
                    error_code="NAME_TOO_LONG"
                ))
        
        elif field == "base_urls":
            if not value:
                result.add_error(ValidationError(
                    field=field,
                    message="At least one base URL is required",
                    error_code="EMPTY_BASE_URLS"
                ))
            else:
                for i, url in enumerate(value):
                    if not isinstance(url, str):
                        result.add_error(ValidationError(
                            field=f"{field}[{i}]",
                            message=f"URL must be a string, got {type(url).__name__}",
                            error_code="INVALID_URL_TYPE"
                        ))
                        continue
                    
                    if not self.URL_PATTERN.match(url):
                        result.add_error(ValidationError(
                            field=f"{field}[{i}]",
                            message=f"Invalid URL format: {url}",
                            suggestion="Ensure URL starts with http:// or https://",
                            error_code="INVALID_URL_FORMAT"
                        ))
                    
                    # Check for common URL issues
                    parsed = urlparse(url)
                    if not parsed.netloc:
                        result.add_error(ValidationError(
                            field=f"{field}[{i}]",
                            message=f"URL missing domain: {url}",
                            error_code="MISSING_DOMAIN"
                        ))
        
        elif field == "sitemaps":
            for i, sitemap_url in enumerate(value):
                if not isinstance(sitemap_url, str):
                    result.add_error(ValidationError(
                        field=f"{field}[{i}]",
                        message=f"Sitemap URL must be a string, got {type(sitemap_url).__name__}",
                        error_code="INVALID_SITEMAP_TYPE"
                    ))
                    continue
                
                if not self.URL_PATTERN.match(sitemap_url):
                    result.add_error(ValidationError(
                        field=f"{field}[{i}]",
                        message=f"Invalid sitemap URL format: {sitemap_url}",
                        error_code="INVALID_SITEMAP_URL"
                    ))
        
        elif field in ["include", "exclude"]:
            for i, pattern in enumerate(value):
                if not isinstance(pattern, str):
                    result.add_error(ValidationError(
                        field=f"{field}[{i}]",
                        message=f"Pattern must be a string, got {type(pattern).__name__}",
                        error_code="INVALID_PATTERN_TYPE"
                    ))
                    continue
                
                # Validate regex patterns
                try:
                    re.compile(pattern)
                except re.error as e:
                    result.add_error(ValidationError(
                        field=f"{field}[{i}]",
                        message=f"Invalid regex pattern '{pattern}': {str(e)}",
                        suggestion="Check regex syntax",
                        error_code="INVALID_REGEX"
                    ))
        
        elif field == "rate_limit":
            if value <= 0:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Rate limit must be positive, got {value}",
                    error_code="INVALID_RATE_LIMIT"
                ))
            elif value > 100:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Rate limit very high ({value} req/s). Consider reducing to be respectful to servers",
                    severity="warning",
                    suggestion="Use rate_limit <= 10 for most sites",
                    error_code="HIGH_RATE_LIMIT"
                ))
        
        elif field == "depth":
            if value < 0:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Depth cannot be negative, got {value}",
                    error_code="NEGATIVE_DEPTH"
                ))
            elif value > 10:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Very deep crawl depth ({value}). This may take a long time",
                    severity="warning",
                    suggestion="Consider limiting depth to 5 or less",
                    error_code="DEEP_CRAWL"
                ))
        
        elif field == "priority":
            if not 1 <= value <= 10:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Priority must be between 1-10, got {value}",
                    error_code="INVALID_PRIORITY"
                ))
        
        elif field == "timeout":
            if value <= 0:
                result.add_error(ValidationError(
                    field=field,
                    message=f"Timeout must be positive, got {value}",
                    error_code="INVALID_TIMEOUT"
                ))
            elif value > 300:  # 5 minutes
                result.add_error(ValidationError(
                    field=field,
                    message=f"Very long timeout ({value}s). Consider reducing",
                    severity="warning",
                    suggestion="Use timeout <= 60s for most cases",
                    error_code="LONG_TIMEOUT"
                ))
    
    def _validate_cross_field_constraints(self, config: Dict[str, Any], result: ValidationResult):
        """Validate constraints that span multiple fields."""
        # Check for conflicting include/exclude patterns
        include_patterns = config.get("include", [])
        exclude_patterns = config.get("exclude", [])
        
        if include_patterns and exclude_patterns:
            # Check for patterns that might conflict
            for inc_pattern in include_patterns:
                for exc_pattern in exclude_patterns:
                    if inc_pattern == exc_pattern:
                        result.add_error(ValidationError(
                            field="include/exclude",
                            message=f"Same pattern in both include and exclude: '{inc_pattern}'",
                            severity="warning",
                            suggestion="Remove duplicate pattern from one of the lists",
                            error_code="CONFLICTING_PATTERNS"
                        ))
        
        # Validate auth configuration
        auth = config.get("auth")
        if auth:
            if "type" not in auth:
                result.add_error(ValidationError(
                    field="auth.type",
                    message="Authentication type is required",
                    suggestion="Specify auth type (e.g., 'basic', 'bearer', 'api_key')",
                    error_code="MISSING_AUTH_TYPE"
                ))
            
            auth_type = auth.get("type")
            if auth_type == "basic":
                if "username" not in auth or "password" not in auth:
                    result.add_error(ValidationError(
                        field="auth",
                        message="Basic auth requires 'username' and 'password' fields",
                        error_code="INCOMPLETE_BASIC_AUTH"
                    ))
            elif auth_type == "bearer":
                if "token" not in auth:
                    result.add_error(ValidationError(
                        field="auth",
                        message="Bearer auth requires 'token' field",
                        error_code="MISSING_BEARER_TOKEN"
                    ))
            elif auth_type == "api_key":
                if "key" not in auth or "header" not in auth:
                    result.add_error(ValidationError(
                        field="auth",
                        message="API key auth requires 'key' and 'header' fields",
                        error_code="INCOMPLETE_API_KEY_AUTH"
                    ))
    
    def _validate_performance_settings(self, config: Dict[str, Any], result: ValidationResult):
        """Validate performance-related settings."""
        rate_limit = config.get("rate_limit", 1.0)
        depth = config.get("depth", 3)
        timeout = config.get("timeout", 30)
        
        # Estimate crawl time
        base_urls = config.get("base_urls", [])
        estimated_pages = len(base_urls) * (10 ** min(depth, 3))  # Rough estimate
        estimated_time_hours = (estimated_pages / rate_limit) / 3600
        
        if estimated_time_hours > 24:
            result.add_error(ValidationError(
                field="performance",
                message=f"Estimated crawl time: {estimated_time_hours:.1f} hours. Consider reducing depth or increasing rate_limit",
                severity="warning",
                suggestion="Reduce depth or increase rate_limit for faster crawling",
                error_code="LONG_CRAWL_TIME"
            ))
        
        # Check for resource-intensive configurations
        if depth > 5 and rate_limit > 10:
            result.add_error(ValidationError(
                field="performance",
                message="High depth + high rate limit may overwhelm target servers",
                severity="warning",
                suggestion="Consider reducing either depth or rate_limit",
                error_code="RESOURCE_INTENSIVE"
            ))
    
    def _validate_security_settings(self, config: Dict[str, Any], result: ValidationResult):
        """Validate security-related settings."""
        # Check for insecure URLs
        base_urls = config.get("base_urls", [])
        for i, url in enumerate(base_urls):
            if isinstance(url, str) and url.startswith("http://"):
                result.add_error(ValidationError(
                    field=f"base_urls[{i}]",
                    message=f"Insecure HTTP URL: {url}",
                    severity="warning",
                    suggestion="Use HTTPS for better security",
                    error_code="INSECURE_URL"
                ))
        
        # Check auth configuration security
        auth = config.get("auth")
        if auth and "password" in auth:
            result.add_error(ValidationError(
                field="auth.password",
                message="Passwords should not be stored in plain text in configuration files",
                severity="warning",
                suggestion="Use environment variables or secure credential storage",
                error_code="PLAINTEXT_PASSWORD"
            ))
    
    def validate_directory(self, directory_path: Union[str, Path]) -> Dict[str, ValidationResult]:
        """Validate all source files in a directory."""
        directory_path = Path(directory_path)
        results = {}
        
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return results
        
        # Find all YAML files
        yaml_files = list(directory_path.glob("*.yaml")) + list(directory_path.glob("*.yml"))
        
        for yaml_file in yaml_files:
            try:
                result = self.validate_file(yaml_file)
                results[str(yaml_file)] = result
            except Exception as e:
                logger.error(f"Error validating {yaml_file}: {e}")
                results[str(yaml_file)] = ValidationResult(
                    is_valid=False,
                    errors=[ValidationError(
                        field="validation",
                        message=f"Validation failed: {str(e)}",
                        error_code="VALIDATION_ERROR"
                    )]
                )
        
        return results
    
    def generate_summary_report(self, results: Dict[str, ValidationResult]) -> str:
        """Generate a summary report for multiple validation results."""
        total_files = len(results)
        valid_files = sum(1 for r in results.values() if r.is_valid)
        total_errors = sum(len(r.errors) for r in results.values())
        total_warnings = sum(len(r.warnings) for r in results.values())
        
        lines = [
            "📊 Source Validation Summary",
            "=" * 30,
            f"Total files: {total_files}",
            f"Valid files: {valid_files}",
            f"Invalid files: {total_files - valid_files}",
            f"Total errors: {total_errors}",
            f"Total warnings: {total_warnings}",
            ""
        ]
        
        if total_files - valid_files > 0:
            lines.append("❌ Files with errors:")
            for file_path, result in results.items():
                if not result.is_valid:
                    lines.append(f"  - {Path(file_path).name}: {len(result.errors)} errors")
            lines.append("")
        
        if total_warnings > 0:
            lines.append("⚠️  Files with warnings:")
            for file_path, result in results.items():
                if result.warnings:
                    lines.append(f"  - {Path(file_path).name}: {len(result.warnings)} warnings")
            lines.append("")
        
        if valid_files == total_files and total_warnings == 0:
            lines.append("✅ All files are valid with no warnings!")
        elif valid_files == total_files:
            lines.append("✅ All files are valid (with warnings)")
        
        return "\n".join(lines)


def main():
    """CLI entry point for source validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate DocFoundry source configurations")
    parser.add_argument("path", help="Path to source file or directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    validator = EnhancedSourceValidator()
    path = Path(args.path)
    
    if path.is_file():
        result = validator.validate_file(path)
        if args.format == "json":
            print(json.dumps({
                "file": str(path),
                "is_valid": result.is_valid,
                "errors": [str(e) for e in result.errors],
                "warnings": [str(w) for w in result.warnings]
            }, indent=2))
        else:
            print(f"Validating: {path}")
            print(result.format_report())
        
        exit(0 if result.is_valid else 1)
    
    elif path.is_dir():
        results = validator.validate_directory(path)
        
        if args.format == "json":
            json_results = {}
            for file_path, result in results.items():
                json_results[file_path] = {
                    "is_valid": result.is_valid,
                    "errors": [str(e) for e in result.errors],
                    "warnings": [str(w) for w in result.warnings]
                }
            print(json.dumps(json_results, indent=2))
        else:
            print(validator.generate_summary_report(results))
            
            if args.verbose:
                print("\n" + "=" * 50)
                for file_path, result in results.items():
                    print(f"\n📄 {Path(file_path).name}")
                    print("-" * 30)
                    print(result.format_report())
        
        # Exit with error code if any files are invalid
        exit(0 if all(r.is_valid for r in results.values()) else 1)
    
    else:
        print(f"Error: Path does not exist: {path}")
        exit(1)


if __name__ == "__main__":
    main()