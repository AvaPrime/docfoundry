"""Incremental processing utilities for DocFoundry."""
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
import requests
from sqlalchemy.orm import Session
from .models import Document, Chunk

logger = logging.getLogger(__name__)

class IncrementalProcessor:
    """Handles incremental document processing and change detection."""
    
    def __init__(self, db_session: Session, current_version: str = "1.0"):
        self.db = db_session
        self.current_version = current_version
        self.current_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Default
        self.current_chunker_version = "1.0"
    
    def compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for change detection."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def check_document_changed(self, url: str, new_content: str, 
                             etag: Optional[str] = None,
                             last_modified: Optional[str] = None) -> Tuple[bool, Optional[Document]]:
        """Check if document has changed since last processing.
        
        Returns:
            Tuple of (has_changed, existing_document)
        """
        existing_doc = self.db.query(Document).filter(Document.url == url).first()
        
        if not existing_doc:
            return True, None
        
        # Check content hash first (most reliable)
        new_hash = self.compute_content_hash(new_content)
        if existing_doc.content_hash != new_hash:
            logger.info(f"Document content changed for {url}")
            return True, existing_doc
        
        # Check HTTP conditional headers if available
        if etag and existing_doc.etag and existing_doc.etag != etag:
            logger.info(f"Document ETag changed for {url}")
            return True, existing_doc
        
        if last_modified and existing_doc.last_modified:
            try:
                new_modified = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                if new_modified > existing_doc.last_modified:
                    logger.info(f"Document last-modified changed for {url}")
                    return True, existing_doc
            except ValueError:
                logger.warning(f"Invalid last-modified header for {url}: {last_modified}")
        
        return False, existing_doc
    
    def check_http_conditional(self, url: str, existing_doc: Document) -> bool:
        """Use HTTP conditional requests to check if document changed.
        
        Returns True if document has changed, False otherwise.
        """
        try:
            headers = {}
            if existing_doc.etag:
                headers['If-None-Match'] = existing_doc.etag
            if existing_doc.last_modified:
                headers['If-Modified-Since'] = existing_doc.last_modified.strftime('%a, %d %b %Y %H:%M:%S GMT')
            
            response = requests.head(url, headers=headers, timeout=10)
            
            # 304 Not Modified means no change
            if response.status_code == 304:
                return False
            
            # Any other successful response means changed
            if response.status_code == 200:
                return True
            
            # For other status codes, assume changed to be safe
            logger.warning(f"Unexpected status code {response.status_code} for {url}")
            return True
            
        except requests.RequestException as e:
            logger.warning(f"HTTP conditional check failed for {url}: {e}")
            # If we can't check, assume changed to be safe
            return True
    
    def needs_reprocessing(self, document: Document) -> bool:
        """Check if document needs reprocessing due to pipeline changes.
        
        Returns True if:
        - Processing version has changed
        - Embedding model has changed
        - Chunker version has changed
        """
        if not document.last_processed_at:
            return True
        
        if document.processing_version != self.current_version:
            logger.info(f"Processing version changed for document {document.id}: {document.processing_version} -> {self.current_version}")
            return True
        
        # Check if any chunks have outdated embedding model
        for chunk in document.chunks:
            if chunk.embedding_model != self.current_embedding_model:
                logger.info(f"Embedding model changed for document {document.id}: {chunk.embedding_model} -> {self.current_embedding_model}")
                return True
            if chunk.chunker_version != self.current_chunker_version:
                logger.info(f"Chunker version changed for document {document.id}: {chunk.chunker_version} -> {self.current_chunker_version}")
                return True
        
        return False
    
    def process_document_full(self, url: str, title: str, content: str, 
                            source_type: str = "web",
                            etag: Optional[str] = None,
                            last_modified: Optional[str] = None) -> Document:
        """Fully process a document (update existing or create new).
        
        This method:
        1. Updates or creates the document record
        2. Deletes old chunks
        3. Creates new chunks with lineage tracking
        4. Updates processing metadata
        """
        content_hash = self.compute_content_hash(content)
        
        # Check if document exists
        existing_doc = self.db.query(Document).filter(Document.url == url).first()
        
        if existing_doc:
            # Update existing document
            existing_doc.title = title
            existing_doc.content = content
            existing_doc.content_hash = content_hash
            existing_doc.updated_at = datetime.utcnow()
            existing_doc.last_processed_at = datetime.utcnow()
            existing_doc.processing_version = self.current_version
            existing_doc.etag = etag
            if last_modified:
                try:
                    existing_doc.last_modified = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                except ValueError:
                    pass
            
            # Delete old chunks
            self.db.query(Chunk).filter(Chunk.document_id == existing_doc.id).delete()
            document = existing_doc
        else:
            # Create new document
            document = Document(
                title=title,
                content=content,
                url=url,
                content_hash=content_hash,
                source_type=source_type,
                last_processed_at=datetime.utcnow(),
                processing_version=self.current_version,
                etag=etag
            )
            if last_modified:
                try:
                    document.last_modified = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                except ValueError:
                    pass
            
            self.db.add(document)
            self.db.flush()  # Get the ID
        
        # Create new chunks
        chunks = self._create_chunks(document, content)
        
        self.db.commit()
        logger.info(f"Processed document {document.id} with {len(chunks)} chunks")
        
        return document
    
    def _create_chunks(self, document: Document, content: str) -> List[Chunk]:
        """Create chunks for a document with lineage tracking."""
        # Simple chunking strategy - split by paragraphs with max size
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_index = 0
        max_chunk_size = 1000  # characters
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                # Create chunk
                chunk = self._create_chunk_with_lineage(
                    document=document,
                    content=current_chunk.strip(),
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk if any content remains
        if current_chunk.strip():
            chunk = self._create_chunk_with_lineage(
                document=document,
                content=current_chunk.strip(),
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_with_lineage(self, document: Document, content: str, chunk_index: int) -> Chunk:
        """Create a chunk with full lineage tracking."""
        # Generate lineage ID
        lineage_data = f"{document.url}:{chunk_index}:{self.current_version}:{self.current_chunker_version}"
        lineage_id = hashlib.md5(lineage_data.encode()).hexdigest()
        
        # Generate embedding (placeholder - replace with actual embedding generation)
        embedding = self._generate_embedding(content)
        
        chunk = Chunk(
            document_id=document.id,
            content=content,
            chunk_index=chunk_index,
            embedding=embedding,
            chunker_version=self.current_chunker_version,
            chunker_config={
                "max_chunk_size": 1000,
                "split_method": "paragraph",
                "overlap": 0
            },
            embedding_model=self.current_embedding_model,
            embedding_version="1.0",
            embedding_dimensions=384 if embedding else None,
            processing_timestamp=datetime.utcnow(),
            lineage_id=lineage_id,
            processing_metadata={
                "processed_by": "IncrementalProcessor",
                "processing_version": self.current_version,
                "content_length": len(content),
                "word_count": len(content.split())
            }
        )
        
        self.db.add(chunk)
        return chunk
    
    def _generate_embedding(self, content: str) -> Optional[List[float]]:
        """Generate embedding for content.
        
        This is a placeholder implementation. In production, this would:
        1. Use the configured embedding model
        2. Handle batching for efficiency
        3. Include proper error handling
        """
        # Placeholder: return None for now
        # In production, integrate with sentence-transformers or other embedding service
        return None
    
    def count_chunks(self, document_id: int) -> int:
        """Count chunks for a document."""
        return self.db.query(Chunk).filter(Chunk.document_id == document_id).count()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_docs = self.db.query(Document).count()
        processed_docs = self.db.query(Document).filter(Document.last_processed_at.isnot(None)).count()
        total_chunks = self.db.query(Chunk).count()
        
        return {
            "total_documents": total_docs,
            "processed_documents": processed_docs,
            "unprocessed_documents": total_docs - processed_docs,
            "total_chunks": total_chunks,
            "current_version": self.current_version,
            "current_embedding_model": self.current_embedding_model,
            "current_chunker_version": self.current_chunker_version
        }