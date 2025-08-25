from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict
import hashlib
import re
from collections import defaultdict

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    h_path: List[str]
    url: str | None
    retrieved_at: str | None
    token_len: int
    lang: str | None
    hash: str

class Chunker:
    def __init__(self, max_tokens: int = 512, overlap: int = 64):
        assert overlap < max_tokens
        self.max_tokens = max_tokens
        self.overlap = overlap
        # Simple token estimation: ~4 chars per token for English
        self.chars_per_token = 4

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation based on character count."""
        return len(text) // self.chars_per_token

    def _extract_headings(self, md: str) -> List[tuple[int, str, int]]:
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

    def _build_heading_hierarchy(self, headings: List[tuple[int, str, int]]) -> Dict[int, List[str]]:
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

    def _split_by_sentences(self, text: str, max_chars: int, overlap_chars: int) -> List[str]:
        """Split text by sentences with overlap."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else current_chunk
                    current_chunk = overlap_text + sentence + " "
                else:
                    # Single sentence is too long, just add it
                    chunks.append(sentence)
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _generate_chunk_id(self, doc_id: str, h_path: List[str], offset: int) -> str:
        """Generate deterministic chunk ID based on document, heading path, and offset."""
        path_str = "|".join(h_path) if h_path else "root"
        content = f"{doc_id}#{path_str}#{offset}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _generate_content_hash(self, text: str) -> str:
        """Generate hash of chunk content for change detection."""
        return hashlib.sha256(text.encode()).hexdigest()

    def split_markdown(self, md: str, doc_id: str = "unknown", url: str = None, 
                      retrieved_at: str = None, lang: str = None) -> Iterable[Chunk]:
        """Heading-aware split that respects code fences and lists.
        1) Parse headings to build h_path (H1→Hn)
        2) Fuse small sections; split long sections by sentences with overlap
        3) Yield stable chunk_ids (e.g., short hash of h_path + offset)
        """
        if not md.strip():
            return
        
        # Extract headings and build hierarchy
        headings = self._extract_headings(md)
        hierarchy = self._build_heading_hierarchy(headings)
        
        # Convert token limits to character limits (rough estimation)
        max_chars = self.max_tokens * self.chars_per_token
        overlap_chars = self.overlap * self.chars_per_token
        
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
        
        chunk_offset = 0
        
        for section_pos, section_text in sections:
            h_path = self._get_heading_path_for_position(section_pos, hierarchy)
            
            # If section is small enough, keep as single chunk
            if len(section_text) <= max_chars:
                chunk_id = self._generate_chunk_id(doc_id, h_path, chunk_offset)
                content_hash = self._generate_content_hash(section_text)
                token_len = self._estimate_tokens(section_text)
                
                yield Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=section_text,
                    h_path=h_path,
                    url=url,
                    retrieved_at=retrieved_at,
                    token_len=token_len,
                    lang=lang,
                    hash=content_hash
                )
                chunk_offset += 1
            else:
                # Split large section by sentences with overlap
                sub_chunks = self._split_by_sentences(section_text, max_chars, overlap_chars)
                
                for sub_chunk in sub_chunks:
                    if sub_chunk.strip():
                        chunk_id = self._generate_chunk_id(doc_id, h_path, chunk_offset)
                        content_hash = self._generate_content_hash(sub_chunk)
                        token_len = self._estimate_tokens(sub_chunk)
                        
                        yield Chunk(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            text=sub_chunk,
                            h_path=h_path,
                            url=url,
                            retrieved_at=retrieved_at,
                            token_len=token_len,
                            lang=lang,
                            hash=content_hash
                        )
                        chunk_offset += 1

# Example usage and testing
if __name__ == "__main__":
    chunker = Chunker(max_tokens=200, overlap=32)
    
    sample_md = """
# Introduction
This is the introduction section with some content.

## Getting Started
Here's how to get started with the project.

### Installation
Run the following commands:
```bash
npm install
```

### Configuration
Edit your config file.

## Advanced Usage
This section covers advanced topics that might be quite long and could potentially exceed the token limit for a single chunk, so it should be split appropriately while maintaining the heading context.

### Performance Tips
Some performance considerations.
"""
    
    chunks = list(chunker.split_markdown(sample_md, doc_id="test_doc", url="https://example.com"))
    
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  H-Path: {' > '.join(chunk.h_path) if chunk.h_path else 'root'}")
        print(f"  Tokens: {chunk.token_len}")
        print(f"  Text: {chunk.text[:100]}...")
        print(f"  Hash: {chunk.hash[:8]}")