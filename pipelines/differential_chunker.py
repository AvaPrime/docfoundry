#!/usr/bin/env python3
"""
Differential Chunking System for Incremental Updates

This module provides functionality to detect changes in documents and
only re-process the sections that have changed, improving efficiency
for incremental crawling and indexing.
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import difflib
import re

logger = logging.getLogger(__name__)

@dataclass
class ChunkDiff:
    """Represents a difference in a document chunk"""
    chunk_id: str
    action: str  # 'added', 'modified', 'deleted', 'unchanged'
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    similarity_score: float = 0.0
    section_path: Optional[str] = None

@dataclass
class DocumentSection:
    """Represents a section of a document with metadata"""
    section_id: str
    heading: str
    content: str
    content_hash: str
    level: int
    path: str  # Hierarchical path like "Introduction > Getting Started"
    start_pos: int
    end_pos: int

class DifferentialChunker:
    """Handles differential chunking for incremental document updates"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def extract_sections(self, content: str, url: str = "") -> List[DocumentSection]:
        """Extract hierarchical sections from document content"""
        sections = []
        
        # Split content by headings (markdown-style)
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = {
            'heading': 'Document Root',
            'content': '',
            'level': 0,
            'start_pos': 0,
            'path': []
        }
        
        heading_stack = []
        pos = 0
        
        for i, line in enumerate(lines):
            line_length = len(line) + 1  # +1 for newline
            
            heading_match = re.match(heading_pattern, line, re.MULTILINE)
            if heading_match:
                # Save previous section if it has content
                if current_section['content'].strip():
                    section_id = self._generate_section_id(
                        current_section['heading'], 
                        current_section['content'], 
                        url
                    )
                    
                    sections.append(DocumentSection(
                        section_id=section_id,
                        heading=current_section['heading'],
                        content=current_section['content'].strip(),
                        content_hash=hashlib.sha256(current_section['content'].encode()).hexdigest(),
                        level=current_section['level'],
                        path=' > '.join(current_section['path']),
                        start_pos=current_section['start_pos'],
                        end_pos=pos
                    ))
                
                # Start new section
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                
                # Update heading stack
                heading_stack = heading_stack[:level-1]
                heading_stack.append(heading)
                
                current_section = {
                    'heading': heading,
                    'content': '',
                    'level': level,
                    'start_pos': pos + line_length,
                    'path': heading_stack.copy()
                }
            else:
                current_section['content'] += line + '\n'
            
            pos += line_length
        
        # Add final section
        if current_section['content'].strip():
            section_id = self._generate_section_id(
                current_section['heading'], 
                current_section['content'], 
                url
            )
            
            sections.append(DocumentSection(
                section_id=section_id,
                heading=current_section['heading'],
                content=current_section['content'].strip(),
                content_hash=hashlib.sha256(current_section['content'].encode()).hexdigest(),
                level=current_section['level'],
                path=' > '.join(current_section['path']),
                start_pos=current_section['start_pos'],
                end_pos=pos
            ))
        
        return sections
    
    def _generate_section_id(self, heading: str, content: str, url: str) -> str:
        """Generate a stable ID for a document section"""
        # Use heading and first 100 chars of content for stability
        stable_content = content[:100] if content else ""
        id_source = f"{url}#{heading}#{stable_content}"
        return hashlib.sha256(id_source.encode()).hexdigest()[:16]
    
    def compare_documents(self, old_sections: List[DocumentSection], 
                         new_sections: List[DocumentSection]) -> List[ChunkDiff]:
        """Compare two versions of a document and identify changes"""
        diffs = []
        
        # Create lookup maps
        old_sections_map = {s.section_id: s for s in old_sections}
        new_sections_map = {s.section_id: s for s in new_sections}
        
        old_ids = set(old_sections_map.keys())
        new_ids = set(new_sections_map.keys())
        
        # Find deleted sections
        deleted_ids = old_ids - new_ids
        for section_id in deleted_ids:
            old_section = old_sections_map[section_id]
            diffs.append(ChunkDiff(
                chunk_id=section_id,
                action='deleted',
                old_content=old_section.content,
                section_path=old_section.path
            ))
        
        # Find added sections
        added_ids = new_ids - old_ids
        for section_id in added_ids:
            new_section = new_sections_map[section_id]
            diffs.append(ChunkDiff(
                chunk_id=section_id,
                action='added',
                new_content=new_section.content,
                section_path=new_section.path
            ))
        
        # Find modified or unchanged sections
        common_ids = old_ids & new_ids
        for section_id in common_ids:
            old_section = old_sections_map[section_id]
            new_section = new_sections_map[section_id]
            
            if old_section.content_hash != new_section.content_hash:
                # Calculate similarity score
                similarity = self._calculate_similarity(old_section.content, new_section.content)
                
                diffs.append(ChunkDiff(
                    chunk_id=section_id,
                    action='modified',
                    old_content=old_section.content,
                    new_content=new_section.content,
                    similarity_score=similarity,
                    section_path=new_section.path
                ))
            else:
                diffs.append(ChunkDiff(
                    chunk_id=section_id,
                    action='unchanged',
                    old_content=old_section.content,
                    new_content=new_section.content,
                    similarity_score=1.0,
                    section_path=new_section.path
                ))
        
        return diffs
    
    def _calculate_similarity(self, old_content: str, new_content: str) -> float:
        """Calculate similarity between two text contents"""
        if not old_content and not new_content:
            return 1.0
        if not old_content or not new_content:
            return 0.0
        
        # Use difflib's sequence matcher
        matcher = difflib.SequenceMatcher(None, old_content, new_content)
        return matcher.ratio()
    
    def get_sections_to_reprocess(self, diffs: List[ChunkDiff]) -> List[str]:
        """Get list of section IDs that need reprocessing"""
        sections_to_reprocess = []
        
        for diff in diffs:
            if diff.action in ['added', 'modified', 'deleted']:
                sections_to_reprocess.append(diff.chunk_id)
            elif diff.action == 'unchanged':
                # Skip unchanged sections
                continue
        
        return sections_to_reprocess
    
    def should_reprocess_section(self, diff: ChunkDiff) -> bool:
        """Determine if a section should be reprocessed based on changes"""
        if diff.action in ['added', 'deleted']:
            return True
        
        if diff.action == 'modified':
            # Reprocess if similarity is below threshold
            return diff.similarity_score < self.similarity_threshold
        
        return False
    
    def generate_incremental_chunks(self, url: str, old_content: str, new_content: str) -> Dict[str, Any]:
        """Generate incremental chunks for a document update"""
        try:
            # Extract sections from both versions
            old_sections = self.extract_sections(old_content, url) if old_content else []
            new_sections = self.extract_sections(new_content, url)
            
            # Compare sections
            diffs = self.compare_documents(old_sections, new_sections)
            
            # Get sections that need reprocessing
            sections_to_reprocess = []
            sections_to_delete = []
            
            for diff in diffs:
                if self.should_reprocess_section(diff):
                    if diff.action == 'deleted':
                        sections_to_delete.append(diff.chunk_id)
                    else:
                        sections_to_reprocess.append({
                            'section_id': diff.chunk_id,
                            'content': diff.new_content,
                            'heading': next((s.heading for s in new_sections if s.section_id == diff.chunk_id), ''),
                            'path': diff.section_path,
                            'action': diff.action
                        })
            
            # Calculate statistics
            stats = {
                'total_sections_old': len(old_sections),
                'total_sections_new': len(new_sections),
                'sections_added': len([d for d in diffs if d.action == 'added']),
                'sections_modified': len([d for d in diffs if d.action == 'modified']),
                'sections_deleted': len([d for d in diffs if d.action == 'deleted']),
                'sections_unchanged': len([d for d in diffs if d.action == 'unchanged']),
                'reprocess_percentage': (len(sections_to_reprocess) + len(sections_to_delete)) / max(len(new_sections), 1) * 100
            }
            
            logger.info(f"Incremental analysis for {url}: {stats['reprocess_percentage']:.1f}% needs reprocessing")
            
            return {
                'sections_to_reprocess': sections_to_reprocess,
                'sections_to_delete': sections_to_delete,
                'diffs': diffs,
                'stats': stats,
                'new_sections': new_sections
            }
            
        except Exception as e:
            logger.error(f"Error in incremental chunking for {url}: {str(e)}")
            # Fallback to full reprocessing
            new_sections = self.extract_sections(new_content, url)
            return {
                'sections_to_reprocess': [{
                    'section_id': s.section_id,
                    'content': s.content,
                    'heading': s.heading,
                    'path': s.path,
                    'action': 'added'
                } for s in new_sections],
                'sections_to_delete': [],
                'diffs': [],
                'stats': {'reprocess_percentage': 100.0},
                'new_sections': new_sections
            }

def create_differential_chunker(similarity_threshold: float = 0.8) -> DifferentialChunker:
    """Factory function to create a differential chunker"""
    return DifferentialChunker(similarity_threshold=similarity_threshold)