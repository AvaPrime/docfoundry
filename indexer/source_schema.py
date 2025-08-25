from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import re
from enum import Enum

class SourceType(Enum):
    """Supported source types for document ingestion."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    TEXT = "text"
    DOCX = "docx"
    CONFLUENCE = "confluence"
    NOTION = "notion"
    GITHUB = "github"
    WEB = "web"

class ValidationError(Exception):
    """Raised when source validation fails."""
    pass

@dataclass
class SourceMetadata:
    """Metadata for a document source."""
    # Required fields
    doc_id: str
    title: str
    source_type: SourceType
    url: Optional[str] = None
    
    # Content metadata
    content_hash: Optional[str] = None
    content_length: Optional[int] = None
    language: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    retrieved_at: Optional[datetime] = None
    
    # Source-specific metadata
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Technical metadata
    encoding: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    
    # Custom metadata
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, SourceType):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceMetadata':
        """Create from dictionary."""
        # Convert string timestamps back to datetime
        for field_name in ['created_at', 'updated_at', 'retrieved_at']:
            if field_name in data and isinstance(data[field_name], str):
                try:
                    data[field_name] = datetime.fromisoformat(data[field_name])
                except ValueError:
                    data[field_name] = None
        
        # Convert source_type string to enum
        if 'source_type' in data and isinstance(data['source_type'], str):
            data['source_type'] = SourceType(data['source_type'])
        
        return cls(**data)

@dataclass
class SourceDocument:
    """Complete source document with metadata and content."""
    metadata: SourceMetadata
    content: str
    raw_content: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': self.metadata.to_dict(),
            'content': self.content,
            'raw_content': self.raw_content.hex() if self.raw_content else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceDocument':
        """Create from dictionary."""
        metadata = SourceMetadata.from_dict(data['metadata'])
        content = data['content']
        raw_content = bytes.fromhex(data['raw_content']) if data.get('raw_content') else None
        return cls(metadata=metadata, content=content, raw_content=raw_content)

class SourceValidator:
    """Validates source documents against schema requirements."""
    
    def __init__(self):
        self.url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
            r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # host...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    def validate_doc_id(self, doc_id: str) -> None:
        """Validate document ID format."""
        if not doc_id or not isinstance(doc_id, str):
            raise ValidationError("doc_id must be a non-empty string")
        
        if len(doc_id) > 255:
            raise ValidationError("doc_id must be 255 characters or less")
        
        # Check for valid characters (alphanumeric, hyphens, underscores, dots)
        if not re.match(r'^[a-zA-Z0-9._-]+$', doc_id):
            raise ValidationError("doc_id contains invalid characters")
    
    def validate_title(self, title: str) -> None:
        """Validate document title."""
        if not title or not isinstance(title, str):
            raise ValidationError("title must be a non-empty string")
        
        if len(title) > 500:
            raise ValidationError("title must be 500 characters or less")
    
    def validate_url(self, url: Optional[str]) -> None:
        """Validate URL format if provided."""
        if url is None:
            return
        
        if not isinstance(url, str):
            raise ValidationError("url must be a string")
        
        if len(url) > 2048:
            raise ValidationError("url must be 2048 characters or less")
        
        if not self.url_pattern.match(url):
            raise ValidationError("url format is invalid")
    
    def validate_content(self, content: str, source_type: SourceType) -> None:
        """Validate document content."""
        if not isinstance(content, str):
            raise ValidationError("content must be a string")
        
        if len(content) == 0:
            raise ValidationError("content cannot be empty")
        
        # Content length limits by type
        max_lengths = {
            SourceType.MARKDOWN: 10_000_000,  # 10MB
            SourceType.HTML: 10_000_000,
            SourceType.TEXT: 10_000_000,
            SourceType.PDF: 50_000_000,  # 50MB
            SourceType.DOCX: 10_000_000,
            SourceType.CONFLUENCE: 5_000_000,
            SourceType.NOTION: 5_000_000,
            SourceType.GITHUB: 1_000_000,
            SourceType.WEB: 5_000_000,
        }
        
        max_length = max_lengths.get(source_type, 1_000_000)
        if len(content) > max_length:
            raise ValidationError(f"content exceeds maximum length for {source_type.value}: {max_length}")
    
    def validate_tags(self, tags: List[str]) -> None:
        """Validate tags list."""
        if not isinstance(tags, list):
            raise ValidationError("tags must be a list")
        
        if len(tags) > 50:
            raise ValidationError("too many tags (maximum 50)")
        
        for tag in tags:
            if not isinstance(tag, str):
                raise ValidationError("all tags must be strings")
            if len(tag) > 100:
                raise ValidationError("tag length exceeds 100 characters")
            if not re.match(r'^[a-zA-Z0-9._-]+$', tag):
                raise ValidationError(f"tag '{tag}' contains invalid characters")
    
    def validate_categories(self, categories: List[str]) -> None:
        """Validate categories list."""
        if not isinstance(categories, list):
            raise ValidationError("categories must be a list")
        
        if len(categories) > 20:
            raise ValidationError("too many categories (maximum 20)")
        
        for category in categories:
            if not isinstance(category, str):
                raise ValidationError("all categories must be strings")
            if len(category) > 100:
                raise ValidationError("category length exceeds 100 characters")
    
    def validate_metadata(self, metadata: SourceMetadata) -> None:
        """Validate complete metadata object."""
        self.validate_doc_id(metadata.doc_id)
        self.validate_title(metadata.title)
        self.validate_url(metadata.url)
        self.validate_tags(metadata.tags)
        self.validate_categories(metadata.categories)
        
        # Validate source type
        if not isinstance(metadata.source_type, SourceType):
            raise ValidationError("source_type must be a valid SourceType")
        
        # Validate optional string fields
        for field_name in ['author', 'language', 'encoding', 'mime_type']:
            value = getattr(metadata, field_name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValidationError(f"{field_name} must be a string")
                if len(value) > 255:
                    raise ValidationError(f"{field_name} exceeds 255 characters")
        
        # Validate numeric fields
        for field_name in ['content_length', 'file_size']:
            value = getattr(metadata, field_name)
            if value is not None:
                if not isinstance(value, int) or value < 0:
                    raise ValidationError(f"{field_name} must be a non-negative integer")
        
        # Validate datetime fields
        for field_name in ['created_at', 'updated_at', 'retrieved_at']:
            value = getattr(metadata, field_name)
            if value is not None and not isinstance(value, datetime):
                raise ValidationError(f"{field_name} must be a datetime object")
        
        # Validate custom metadata
        if not isinstance(metadata.custom, dict):
            raise ValidationError("custom metadata must be a dictionary")
    
    def validate_document(self, document: SourceDocument) -> None:
        """Validate complete source document."""
        self.validate_metadata(document.metadata)
        self.validate_content(document.content, document.metadata.source_type)
        
        # Validate raw content if provided
        if document.raw_content is not None and not isinstance(document.raw_content, bytes):
            raise ValidationError("raw_content must be bytes")
    
    def validate_json_schema(self, data: Dict[str, Any]) -> SourceDocument:
        """Validate and parse JSON data into SourceDocument."""
        try:
            document = SourceDocument.from_dict(data)
            self.validate_document(document)
            return document
        except (KeyError, TypeError, ValueError) as e:
            raise ValidationError(f"Invalid JSON schema: {e}")

# Utility functions
def create_source_metadata(
    doc_id: str,
    title: str,
    source_type: Union[str, SourceType],
    **kwargs
) -> SourceMetadata:
    """Helper function to create SourceMetadata with validation."""
    if isinstance(source_type, str):
        source_type = SourceType(source_type)
    
    metadata = SourceMetadata(
        doc_id=doc_id,
        title=title,
        source_type=source_type,
        **kwargs
    )
    
    validator = SourceValidator()
    validator.validate_metadata(metadata)
    return metadata

def validate_source_json(json_str: str) -> SourceDocument:
    """Validate JSON string and return SourceDocument."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {e}")
    
    validator = SourceValidator()
    return validator.validate_json_schema(data)

# Example usage
if __name__ == "__main__":
    # Create sample metadata
    metadata = create_source_metadata(
        doc_id="example-doc-001",
        title="Example Document",
        source_type=SourceType.MARKDOWN,
        url="https://example.com/doc",
        author="John Doe",
        tags=["example", "documentation"],
        categories=["tutorials"]
    )
    
    # Create sample document
    document = SourceDocument(
        metadata=metadata,
        content="# Example\n\nThis is an example document."
    )
    
    # Validate
    validator = SourceValidator()
    try:
        validator.validate_document(document)
        print("Document validation passed!")
        
        # Test JSON serialization
        json_data = document.to_dict()
        json_str = json.dumps(json_data, indent=2)
        print("\nJSON representation:")
        print(json_str[:200] + "...")
        
        # Test JSON validation
        validated_doc = validate_source_json(json_str)
        print("\nJSON validation passed!")
        
    except ValidationError as e:
        print(f"Validation failed: {e}")