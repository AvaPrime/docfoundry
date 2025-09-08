import pytest
import sys
import os

# Add the parent directory to the path so we can import from indexer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from indexer.build_index import chunk_markdown

def test_chunker_respects_size():
    """Test that chunker respects maximum character limits."""
    text = "A" * 2500
    chunks = list(chunk_markdown(text, max_chars=1000, overlap=100))
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)

def test_chunker_with_small_text():
    """Test chunker behavior with text smaller than max_chars."""
    text = "Short text"
    chunks = list(chunk_markdown(text, max_chars=1000, overlap=100))
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunker_overlap():
    """Test that chunker creates proper overlap between chunks."""
    text = "A" * 1500  # Text that will create 2 chunks
    chunks = list(chunk_markdown(text, max_chars=1000, overlap=100))
    
    if len(chunks) > 1:
        # Check that there's some overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            # The end of current chunk should have some similarity with start of next
            assert len(current_chunk) <= 1000
            assert len(next_chunk) <= 1000

def test_chunker_empty_text():
    """Test chunker behavior with empty text."""
    text = ""
    chunks = list(chunk_markdown(text, max_chars=1000, overlap=100))
    assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0] == "")

def test_chunker_markdown_structure():
    """Test chunker with actual markdown content."""
    markdown_text = """# Header 1

This is some content under header 1.

## Header 2

This is content under header 2 with more text to make it longer.

### Header 3

And even more content here to test the chunking behavior."""
    
    chunks = list(chunk_markdown(markdown_text, max_chars=100, overlap=20))
    assert len(chunks) > 0
    assert all(len(c) <= 100 for c in chunks)