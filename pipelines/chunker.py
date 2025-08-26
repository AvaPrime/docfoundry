"""Document chunking pipeline for DocFoundry.

Provides functionality to chunk documents into smaller pieces for better retrieval.
"""

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Iterable
from dataclasses import dataclass
from bs4 import BeautifulSoup
from collections import defaultdict

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
                 min_chunk_size: int = 100,
                 max_tokens: int = 512,
                 overlap_tokens: int = 64):
        """Initialize chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be kept
            max_tokens: Maximum tokens per chunk (for advanced chunking)
            overlap_tokens: Token overlap for advanced chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # Simple token estimation: ~4 chars per token for English
        self.chars_per_token = 4
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: ~4 characters per token on average
        return len(text) // self.chars_per_token
    
    def _extract_headings(self, md: str) -> List[Tuple[int, str, int]]:
        """Extract headings with their levels, text, and positions.
        Returns: [(level, text, start_pos), ...]
        """
        headings = []
        lines = md.split('\n')
        pos = 0
        
        for line in lines:
            # ATX headings (# ## ###)
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append((level, text, pos))
            pos += len(line) + 1  # +1 for newline
        
        return headings
    
    def _build_heading_hierarchy(self, headings: List[Tuple[int, str, int]]) -> Dict[int, List[str]]:
        """Build heading path for each position in the document.
        Returns: {position: [h1, h2, h3, ...], ...}
        """
        hierarchy = defaultdict(list)
        current_path = []
        
        for i, (level, text, pos) in enumerate(headings):
            # Adjust current path based on heading level
            if level == 1:
                current_path = [text]
            elif level <= len(current_path):
                current_path = current_path[:level-1] + [text]
            else:
                # Fill missing levels with empty strings
                while len(current_path) < level - 1:
                    current_path.append("")
                current_path.append(text)
            
            # Set hierarchy for this position and all positions until next heading
            end_pos = headings[i+1][2] if i+1 < len(headings) else float('inf')
            hierarchy[pos] = current_path.copy()
            
        return hierarchy
    
    def _get_heading_path_for_position(self, pos: int, hierarchy: Dict[int, List[str]]) -> List[str]:
        """Get the heading path for a given position in the document."""
        best_pos = 0
        best_path = []
        
        for h_pos, h_path in hierarchy.items():
            if h_pos <= pos and h_pos > best_pos:
                best_pos = h_pos
                best_path = h_path
        
        return best_path
    
    def _generate_stable_chunk_id(self, doc_id: str, h_path: List[str], offset: int) -> str:
        """Generate deterministic chunk ID based on document, heading path, and offset."""
        path_str = "|".join(h_path) if h_path else "root"
        content = f"{doc_id}#{path_str}#{offset}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_content_hash(self, text: str) -> str:
        """Generate hash of chunk content for change detection."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _split_by_sentences(self, text: str, max_chars: int, overlap_chars: int) -> List[str]:
        """Split text by sentences with overlap, respecting sentence boundaries."""
        # Simple sentence splitting (can be enhanced with proper NLP)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max_chars, start new chunk
            if current_chunk and len(current_chunk) + len(sentence) > max_chars:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                if overlap_chars > 0 and len(current_chunk) > overlap_chars:
                    overlap_text = current_chunk[-overlap_chars:]
                    # Find sentence boundary for clean overlap
                    overlap_sentences = re.split(r'(?<=[.!?])\s+', overlap_text)
                    if len(overlap_sentences) > 1:
                        current_chunk = ' '.join(overlap_sentences[1:]) + ' ' + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_markdown_chunks(self, md: str, doc_id: str, url: str = None) -> List[Dict[str, Any]]:
        """Create heading-aware chunks from markdown content with stable IDs."""
        if not md.strip():
            return []
        
        # Extract headings and build hierarchy
        headings = self._extract_headings(md)
        hierarchy = self._build_heading_hierarchy(headings)
        
        # Convert token limits to character limits (rough estimation)
        max_chars = self.max_tokens * self.chars_per_token
        overlap_chars = self.overlap_tokens * self.chars_per_token
        
        # Split by sections (between headings) first
        sections = []
        if headings:
            for i, (level, text, pos) in enumerate(headings):
                end_pos = headings[i+1][2] if i+1 < len(headings) else len(md)
                section_text = md[pos:end_pos].strip()
                if section_text:
                    sections.append((pos, section_text))
        else:
            # No headings, treat entire document as one section
            sections = [(0, md)]
        
        chunks = []
        chunk_offset = 0
        
        for section_pos, section_text in sections:
            h_path = self._get_heading_path_for_position(section_pos, hierarchy)
            
            # If section is small enough, keep as single chunk
            if len(section_text) <= max_chars:
                chunk_id = self._generate_stable_chunk_id(doc_id, h_path, chunk_offset)
                content_hash = self._generate_content_hash(section_text)
                token_len = self._estimate_tokens(section_text)
                
                chunks.append({
                    'content': section_text,
                    'metadata': {
                        'chunk_id': chunk_id,
                        'h_path': h_path,
                        'heading': h_path[-1] if h_path else None,
                        'heading_level': len(h_path) if h_path else None,
                        'content_hash': content_hash,
                        'token_count': token_len,
                        'section_offset': chunk_offset
                    }
                })
                chunk_offset += 1
            else:
                # Split large section by sentences with overlap
                sub_chunks = self._split_by_sentences(section_text, max_chars, overlap_chars)
                
                for sub_chunk in sub_chunks:
                    if sub_chunk.strip():
                        chunk_id = self._generate_stable_chunk_id(doc_id, h_path, chunk_offset)
                        content_hash = self._generate_content_hash(sub_chunk)
                        token_len = self._estimate_tokens(sub_chunk)
                        
                        chunks.append({
                            'content': sub_chunk,
                            'metadata': {
                                'chunk_id': chunk_id,
                                'h_path': h_path,
                                'heading': h_path[-1] if h_path else None,
                                'heading_level': len(h_path) if h_path else None,
                                'content_hash': content_hash,
                                'token_count': token_len,
                                'section_offset': chunk_offset
                            }
                        })
                        chunk_offset += 1
        
        return chunks
    
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
    
    def _extract_html_headings(self, content: str) -> List[Tuple[str, int]]:
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
            use_heading_aware: Whether to use heading-aware chunking
        
        Returns:
            List of DocumentChunk objects
        """
        content = document.get('content', '')
        if not content:
            return []
        
        # Generate document ID
        doc_id = hashlib.sha256(document['url'].encode()).hexdigest()[:16]
        
        # Determine content type
        is_html = ('<html' in content.lower() or 
                  '<body' in content.lower() or 
                  bool(re.search(r'<[^>]+>', content)))
        
        # Check if content looks like markdown (has markdown headings)
        is_markdown = bool(re.search(r'^#{1,6}\s+.+$', content, re.MULTILINE))
        
        chunks_data = []
        
        if use_heading_aware and is_markdown:
            # Use advanced markdown chunking with stable IDs and hierarchy
            chunks_data = self._create_markdown_chunks(content, doc_id, document['url'])
        elif is_html and use_heading_aware:
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
            
            chunk_metadata = chunk_data.get('metadata', {})
            
            # Use stable chunk ID if available, otherwise generate one
            if 'chunk_id' in chunk_metadata:
                chunk_id = chunk_metadata['chunk_id']
            else:
                chunk_id = f"{doc_id}_{i:04d}"
            
            # Use content hash if available, otherwise generate one
            if 'content_hash' in chunk_metadata:
                chunk_hash = chunk_metadata['content_hash']
            else:
                chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()
            
            # Use token count if available, otherwise estimate
            if 'token_count' in chunk_metadata:
                token_count = chunk_metadata['token_count']
            else:
                token_count = self._estimate_tokens(chunk_content)
            
            # Combine document metadata with chunk metadata
            metadata = {
                'url': document['url'],
                'title': document.get('title', ''),
                'source_name': document.get('source_name', ''),
                **chunk_metadata
            }
            
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=doc_id,
                chunk_index=i,
                content=chunk_content,
                content_hash=chunk_hash,
                token_count=token_count,
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