"""Document chunking pipeline for DocFoundry.

Provides functionality to chunk documents into smaller pieces for better retrieval.
"""

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    content_hash: str
    token_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "content_hash": self.content_hash,
            "token_count": self.token_count,
            "metadata": self.metadata
        }

class DocumentChunker:
    """Chunks documents into smaller pieces for better retrieval."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100):
        """Initialize chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be kept
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: ~4 characters per token on average
        return len(text) // 4
    
    def _clean_html(self, content: str) -> str:
        """Clean HTML content and extract text."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean up whitespace
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.warning(f"Failed to parse HTML content: {e}")
            return content
    
    def _extract_headings(self, content: str) -> List[Tuple[str, int]]:
        """Extract headings from HTML content.
        
        Returns:
            List of (heading_text, level) tuples
        """
        headings = []
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            for i in range(1, 7):  # h1 to h6
                for heading in soup.find_all(f'h{i}'):
                    text = heading.get_text().strip()
                    if text:
                        headings.append((text, i))
            
        except Exception as e:
            logger.warning(f"Failed to extract headings: {e}")
        
        return headings
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be improved with more sophisticated NLP
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _create_chunks_by_size(self, text: str) -> List[str]:
        """Create chunks by target size with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(start + self.chunk_size - 200, start)
                sentence_end = -1
                
                for match in re.finditer(r'[.!?]\s+', text[search_start:end]):
                    sentence_end = search_start + match.end()
                
                if sentence_end > start:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
            
            if start >= len(text):
                break
        
        return chunks
    
    def _create_heading_aware_chunks(self, content: str) -> List[Dict[str, Any]]:
        """Create chunks that respect document structure (headings)."""
        chunks = []
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find all headings and content sections
            elements = []
            current_heading = None
            current_content = []
            
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section']):
                if element.name.startswith('h'):
                    # Save previous section
                    if current_content:
                        content_text = ' '.join(current_content).strip()
                        if content_text:
                            elements.append({
                                'type': 'section',
                                'heading': current_heading,
                                'content': content_text
                            })
                    
                    # Start new section
                    current_heading = {
                        'text': element.get_text().strip(),
                        'level': int(element.name[1])
                    }
                    current_content = []
                else:
                    text = element.get_text().strip()
                    if text:
                        current_content.append(text)
            
            # Add final section
            if current_content:
                content_text = ' '.join(current_content).strip()
                if content_text:
                    elements.append({
                        'type': 'section',
                        'heading': current_heading,
                        'content': content_text
                    })
            
            # Create chunks from sections
            for element in elements:
                section_text = element['content']
                heading = element.get('heading')
                
                # Add heading to content if present
                if heading:
                    section_text = f"{heading['text']}\n\n{section_text}"
                
                # Split large sections into smaller chunks
                if len(section_text) > self.chunk_size:
                    section_chunks = self._create_chunks_by_size(section_text)
                    for i, chunk_text in enumerate(section_chunks):
                        chunks.append({
                            'content': chunk_text,
                            'metadata': {
                                'heading': heading['text'] if heading else None,
                                'heading_level': heading['level'] if heading else None,
                                'section_part': i + 1,
                                'total_parts': len(section_chunks)
                            }
                        })
                else:
                    chunks.append({
                        'content': section_text,
                        'metadata': {
                            'heading': heading['text'] if heading else None,
                            'heading_level': heading['level'] if heading else None
                        }
                    })
            
        except Exception as e:
            logger.warning(f"Failed to create heading-aware chunks: {e}")
            # Fallback to simple chunking
            text = self._clean_html(content)
            simple_chunks = self._create_chunks_by_size(text)
            chunks = [{'content': chunk, 'metadata': {}} for chunk in simple_chunks]
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any], 
                      use_heading_aware: bool = True) -> List[DocumentChunk]:
        """Chunk a single document.
        
        Args:
            document: Document dictionary with 'content', 'url', etc.
            use_heading_aware: Whether to use heading-aware chunking for HTML
        
        Returns:
            List of DocumentChunk objects
        """
        content = document.get('content', '')
        if not content:
            return []
        
        # Generate document ID
        doc_id = hashlib.sha256(document['url'].encode()).hexdigest()[:16]
        
        # Determine if content is HTML
        is_html = ('<html' in content.lower() or 
                  '<body' in content.lower() or 
                  bool(re.search(r'<[^>]+>', content)))
        
        chunks_data = []
        
        if is_html and use_heading_aware:
            # Use heading-aware chunking for HTML
            chunks_data = self._create_heading_aware_chunks(content)
        else:
            # Clean content and use simple chunking
            if is_html:
                clean_content = self._clean_html(content)
            else:
                clean_content = content
            
            simple_chunks = self._create_chunks_by_size(clean_content)
            chunks_data = [{'content': chunk, 'metadata': {}} for chunk in simple_chunks]
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_data in enumerate(chunks_data):
            chunk_content = chunk_data['content']
            
            if len(chunk_content) < self.min_chunk_size:
                continue
            
            # Generate chunk ID
            chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()
            chunk_id = f"{doc_id}_{i:04d}"
            
            # Combine document metadata with chunk metadata
            metadata = {
                'url': document['url'],
                'title': document.get('title', ''),
                'source_name': document.get('source_name', ''),
                **chunk_data.get('metadata', {})
            }
            
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=doc_id,
                chunk_index=i,
                content=chunk_content,
                content_hash=chunk_hash,
                token_count=self._estimate_tokens(chunk_content),
                metadata=metadata
            )
            
            document_chunks.append(chunk)
        
        logger.debug(f"Created {len(document_chunks)} chunks for document {document['url']}")
        return document_chunks
    
    def chunk_documents(self, documents: List[Dict[str, Any]], 
                       use_heading_aware: bool = True) -> List[DocumentChunk]:
        """Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries
            use_heading_aware: Whether to use heading-aware chunking for HTML
        
        Returns:
            List of DocumentChunk objects
        """
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.chunk_document(doc, use_heading_aware)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk document {doc.get('url', 'unknown')}: {e}")
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

# Convenience functions
def chunk_documents(documents: List[Dict[str, Any]], 
                   chunk_size: int = 1000,
                   chunk_overlap: int = 200,
                   min_chunk_size: int = 100,
                   use_heading_aware: bool = True) -> List[DocumentChunk]:
    """Convenience function to chunk documents.
    
    Args:
        documents: List of document dictionaries
        chunk_size: Target size for each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for a chunk to be kept
        use_heading_aware: Whether to use heading-aware chunking for HTML
    
    Returns:
        List of DocumentChunk objects
    """
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size
    )
    
    return chunker.chunk_documents(documents, use_heading_aware)

def chunk_single_document(document: Dict[str, Any], 
                         chunk_size: int = 1000,
                         chunk_overlap: int = 200,
                         min_chunk_size: int = 100,
                         use_heading_aware: bool = True) -> List[DocumentChunk]:
    """Convenience function to chunk a single document.
    
    Args:
        document: Document dictionary
        chunk_size: Target size for each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for a chunk to be kept
        use_heading_aware: Whether to use heading-aware chunking for HTML
    
    Returns:
        List of DocumentChunk objects
    """
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size
    )
    
    return chunker.chunk_document(document, use_heading_aware)