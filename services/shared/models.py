"""Shared database models with lineage tracking."""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

class Document(Base):
    """Document model with processing tracking."""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    url = Column(String(2048), nullable=False, unique=True)
    content_hash = Column(String(64), nullable=False)
    source_type = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Processing tracking fields
    last_processed_at = Column(DateTime(timezone=True), nullable=True)
    processing_version = Column(String(50), nullable=True)
    etag = Column(String(100), nullable=True)
    last_modified = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_content_hash', 'content_hash'),
        Index('idx_documents_last_processed', 'last_processed_at'),
        Index('idx_documents_etag', 'etag'),
        Index('idx_documents_url', 'url'),
    )
    
    def needs_reprocessing(self, current_version: str, current_model: str) -> bool:
        """Check if document needs reprocessing based on version changes."""
        if not self.last_processed_at:
            return True
        if self.processing_version != current_version:
            return True
        # Check if any chunks have different embedding model
        for chunk in self.chunks:
            if chunk.embedding_model != current_model:
                return True
        return False

class Chunk(Base):
    """Chunk model with lineage tracking."""
    __tablename__ = 'chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(JSONB, nullable=True)  # Store as JSONB for PostgreSQL
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Lineage tracking fields
    chunker_version = Column(String(50), nullable=True)
    chunker_config = Column(JSONB, nullable=True)
    embedding_model = Column(String(100), nullable=True)
    embedding_version = Column(String(50), nullable=True)
    embedding_dimensions = Column(Integer, nullable=True)
    processing_timestamp = Column(DateTime(timezone=True), nullable=True, default=datetime.utcnow)
    lineage_id = Column(String(100), nullable=True)
    processing_metadata = Column(JSONB, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_chunks_document_id', 'document_id'),
        Index('idx_chunks_chunker_version', 'chunker_version'),
        Index('idx_chunks_embedding_model', 'embedding_model'),
        Index('idx_chunks_lineage_id', 'lineage_id'),
        Index('idx_chunks_processing_timestamp', 'processing_timestamp'),
    )
    
    def get_lineage_info(self) -> Dict[str, Any]:
        """Get lineage information for this chunk."""
        return {
            'chunker_version': self.chunker_version,
            'chunker_config': self.chunker_config,
            'embedding_model': self.embedding_model,
            'embedding_version': self.embedding_version,
            'embedding_dimensions': self.embedding_dimensions,
            'processing_timestamp': self.processing_timestamp.isoformat() if self.processing_timestamp else None,
            'lineage_id': self.lineage_id,
            'processing_metadata': self.processing_metadata
        }
    
    def update_lineage(self, 
                      chunker_version: str,
                      embedding_model: str,
                      embedding_version: str,
                      embedding_dimensions: int,
                      chunker_config: Optional[Dict] = None,
                      processing_metadata: Optional[Dict] = None,
                      lineage_id: Optional[str] = None):
        """Update lineage information for this chunk."""
        self.chunker_version = chunker_version
        self.embedding_model = embedding_model
        self.embedding_version = embedding_version
        self.embedding_dimensions = embedding_dimensions
        self.chunker_config = chunker_config
        self.processing_metadata = processing_metadata
        self.lineage_id = lineage_id
        self.processing_timestamp = datetime.utcnow()