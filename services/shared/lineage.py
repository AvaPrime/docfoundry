"""Data lineage tracking and analysis utilities."""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from .models import Document, Chunk

logger = logging.getLogger(__name__)

class LineageTracker:
    """Tracks and analyzes data lineage for DocFoundry."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of processing status and lineage."""
        # Basic counts
        total_docs = self.db.query(Document).count()
        processed_docs = self.db.query(Document).filter(Document.last_processed_at.isnot(None)).count()
        total_chunks = self.db.query(Chunk).count()
        
        # Processing version distribution
        version_stats = self.db.query(
            Document.processing_version,
            func.count(Document.id).label('count')
        ).group_by(Document.processing_version).all()
        
        # Embedding model distribution
        embedding_stats = self.db.query(
            Chunk.embedding_model,
            func.count(Chunk.id).label('count')
        ).group_by(Chunk.embedding_model).all()
        
        # Chunker version distribution
        chunker_stats = self.db.query(
            Chunk.chunker_version,
            func.count(Chunk.id).label('count')
        ).group_by(Chunk.chunker_version).all()
        
        # Recent processing activity
        last_24h = datetime.utcnow() - timedelta(hours=24)
        recent_docs = self.db.query(Document).filter(
            Document.last_processed_at >= last_24h
        ).count()
        
        recent_chunks = self.db.query(Chunk).filter(
            Chunk.processing_timestamp >= last_24h
        ).count()
        
        return {
            "overview": {
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "unprocessed_documents": total_docs - processed_docs,
                "total_chunks": total_chunks,
                "processing_coverage": (processed_docs / total_docs * 100) if total_docs > 0 else 0
            },
            "versions": {
                "processing_versions": [{
                    "version": version or "null",
                    "document_count": count
                } for version, count in version_stats],
                "embedding_models": [{
                    "model": model or "null",
                    "chunk_count": count
                } for model, count in embedding_stats],
                "chunker_versions": [{
                    "version": version or "null",
                    "chunk_count": count
                } for version, count in chunker_stats]
            },
            "recent_activity": {
                "documents_processed_24h": recent_docs,
                "chunks_created_24h": recent_chunks
            }
        }
    
    def find_reprocessing_candidates(self, 
                                   target_processing_version: str,
                                   target_embedding_model: str,
                                   target_chunker_version: str,
                                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find documents that need reprocessing due to version mismatches."""
        candidates = []
        
        # Documents with outdated processing version
        outdated_processing = self.db.query(Document).filter(
            or_(
                Document.processing_version != target_processing_version,
                Document.processing_version.is_(None)
            )
        )
        
        if limit:
            outdated_processing = outdated_processing.limit(limit)
        
        for doc in outdated_processing:
            candidates.append({
                "document_id": doc.id,
                "url": doc.url,
                "title": doc.title,
                "reason": "outdated_processing_version",
                "current_version": doc.processing_version,
                "target_version": target_processing_version,
                "last_processed": doc.last_processed_at.isoformat() if doc.last_processed_at else None,
                "chunk_count": len(doc.chunks)
            })
        
        # Documents with chunks using outdated embedding model
        if not limit or len(candidates) < limit:
            remaining_limit = (limit - len(candidates)) if limit else None
            
            docs_with_outdated_embeddings = self.db.query(Document).join(Chunk).filter(
                or_(
                    Chunk.embedding_model != target_embedding_model,
                    Chunk.embedding_model.is_(None)
                )
            ).distinct()
            
            if remaining_limit:
                docs_with_outdated_embeddings = docs_with_outdated_embeddings.limit(remaining_limit)
            
            for doc in docs_with_outdated_embeddings:
                # Skip if already added
                if any(c["document_id"] == doc.id for c in candidates):
                    continue
                
                outdated_chunks = [c for c in doc.chunks if c.embedding_model != target_embedding_model]
                candidates.append({
                    "document_id": doc.id,
                    "url": doc.url,
                    "title": doc.title,
                    "reason": "outdated_embedding_model",
                    "current_model": outdated_chunks[0].embedding_model if outdated_chunks else None,
                    "target_model": target_embedding_model,
                    "last_processed": doc.last_processed_at.isoformat() if doc.last_processed_at else None,
                    "outdated_chunks": len(outdated_chunks),
                    "total_chunks": len(doc.chunks)
                })
        
        # Documents with chunks using outdated chunker version
        if not limit or len(candidates) < limit:
            remaining_limit = (limit - len(candidates)) if limit else None
            
            docs_with_outdated_chunker = self.db.query(Document).join(Chunk).filter(
                or_(
                    Chunk.chunker_version != target_chunker_version,
                    Chunk.chunker_version.is_(None)
                )
            ).distinct()
            
            if remaining_limit:
                docs_with_outdated_chunker = docs_with_outdated_chunker.limit(remaining_limit)
            
            for doc in docs_with_outdated_chunker:
                # Skip if already added
                if any(c["document_id"] == doc.id for c in candidates):
                    continue
                
                outdated_chunks = [c for c in doc.chunks if c.chunker_version != target_chunker_version]
                candidates.append({
                    "document_id": doc.id,
                    "url": doc.url,
                    "title": doc.title,
                    "reason": "outdated_chunker_version",
                    "current_version": outdated_chunks[0].chunker_version if outdated_chunks else None,
                    "target_version": target_chunker_version,
                    "last_processed": doc.last_processed_at.isoformat() if doc.last_processed_at else None,
                    "outdated_chunks": len(outdated_chunks),
                    "total_chunks": len(doc.chunks)
                })
        
        return candidates[:limit] if limit else candidates
    
    def get_reprocessing_stats(self, 
                             target_processing_version: str,
                             target_embedding_model: str,
                             target_chunker_version: str) -> Dict[str, Any]:
        """Get statistics about documents needing reprocessing."""
        # Count documents by reprocessing reason
        outdated_processing = self.db.query(Document).filter(
            or_(
                Document.processing_version != target_processing_version,
                Document.processing_version.is_(None)
            )
        ).count()
        
        docs_with_outdated_embeddings = self.db.query(Document).join(Chunk).filter(
            or_(
                Chunk.embedding_model != target_embedding_model,
                Chunk.embedding_model.is_(None)
            )
        ).distinct().count()
        
        docs_with_outdated_chunker = self.db.query(Document).join(Chunk).filter(
            or_(
                Chunk.chunker_version != target_chunker_version,
                Chunk.chunker_version.is_(None)
            )
        ).distinct().count()
        
        # Count chunks needing reprocessing
        outdated_embedding_chunks = self.db.query(Chunk).filter(
            or_(
                Chunk.embedding_model != target_embedding_model,
                Chunk.embedding_model.is_(None)
            )
        ).count()
        
        outdated_chunker_chunks = self.db.query(Chunk).filter(
            or_(
                Chunk.chunker_version != target_chunker_version,
                Chunk.chunker_version.is_(None)
            )
        ).count()
        
        total_docs = self.db.query(Document).count()
        total_chunks = self.db.query(Chunk).count()
        
        return {
            "targets": {
                "processing_version": target_processing_version,
                "embedding_model": target_embedding_model,
                "chunker_version": target_chunker_version
            },
            "documents": {
                "total": total_docs,
                "outdated_processing": outdated_processing,
                "outdated_embeddings": docs_with_outdated_embeddings,
                "outdated_chunker": docs_with_outdated_chunker,
                "up_to_date": total_docs - max(outdated_processing, docs_with_outdated_embeddings, docs_with_outdated_chunker)
            },
            "chunks": {
                "total": total_chunks,
                "outdated_embeddings": outdated_embedding_chunks,
                "outdated_chunker": outdated_chunker_chunks,
                "up_to_date": total_chunks - max(outdated_embedding_chunks, outdated_chunker_chunks)
            }
        }
    
    def audit_data_consistency(self) -> Dict[str, Any]:
        """Audit data consistency and identify potential issues."""
        issues = []
        
        # Documents without chunks
        docs_without_chunks = self.db.query(Document).outerjoin(Chunk).filter(
            Chunk.id.is_(None)
        ).all()
        
        if docs_without_chunks:
            issues.append({
                "type": "documents_without_chunks",
                "count": len(docs_without_chunks),
                "description": "Documents that have no associated chunks",
                "examples": [{
                    "document_id": doc.id,
                    "url": doc.url,
                    "last_processed": doc.last_processed_at.isoformat() if doc.last_processed_at else None
                } for doc in docs_without_chunks[:5]]
            })
        
        # Chunks without embeddings
        chunks_without_embeddings = self.db.query(Chunk).filter(
            Chunk.embedding.is_(None)
        ).count()
        
        if chunks_without_embeddings > 0:
            issues.append({
                "type": "chunks_without_embeddings",
                "count": chunks_without_embeddings,
                "description": "Chunks that have no embedding vectors"
            })
        
        # Chunks with missing lineage information
        chunks_missing_lineage = self.db.query(Chunk).filter(
            or_(
                Chunk.embedding_model.is_(None),
                Chunk.chunker_version.is_(None),
                Chunk.processing_timestamp.is_(None)
            )
        ).count()
        
        if chunks_missing_lineage > 0:
            issues.append({
                "type": "chunks_missing_lineage",
                "count": chunks_missing_lineage,
                "description": "Chunks with incomplete lineage information"
            })
        
        # Documents with inconsistent processing timestamps
        docs_with_newer_chunks = self.db.query(Document).join(Chunk).filter(
            and_(
                Document.last_processed_at.isnot(None),
                Chunk.processing_timestamp > Document.last_processed_at
            )
        ).distinct().count()
        
        if docs_with_newer_chunks > 0:
            issues.append({
                "type": "inconsistent_timestamps",
                "count": docs_with_newer_chunks,
                "description": "Documents with chunks processed after document timestamp"
            })
        
        # Summary
        total_docs = self.db.query(Document).count()
        total_chunks = self.db.query(Chunk).count()
        
        return {
            "summary": {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "issues_found": len(issues),
                "audit_timestamp": datetime.utcnow().isoformat()
            },
            "issues": issues,
            "health_score": max(0, 100 - (len(issues) * 10))  # Simple scoring
        }
    
    def get_lineage_for_document(self, document_id: int) -> Dict[str, Any]:
        """Get detailed lineage information for a specific document."""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return {"error": "Document not found"}
        
        chunks_info = []
        for chunk in document.chunks:
            chunks_info.append({
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "content_length": len(chunk.content),
                "lineage": chunk.get_lineage_info()
            })
        
        return {
            "document": {
                "id": document.id,
                "url": document.url,
                "title": document.title,
                "content_hash": document.content_hash,
                "last_processed_at": document.last_processed_at.isoformat() if document.last_processed_at else None,
                "processing_version": document.processing_version,
                "etag": document.etag,
                "last_modified": document.last_modified.isoformat() if document.last_modified else None
            },
            "chunks": chunks_info,
            "summary": {
                "total_chunks": len(chunks_info),
                "unique_embedding_models": len(set(c["lineage"]["embedding_model"] for c in chunks_info if c["lineage"]["embedding_model"])),
                "unique_chunker_versions": len(set(c["lineage"]["chunker_version"] for c in chunks_info if c["lineage"]["chunker_version"]))
            }
        }
    
    def cleanup_orphaned_data(self) -> Dict[str, int]:
        """Clean up orphaned data and return cleanup statistics."""
        # Find and delete chunks without documents
        orphaned_chunks = self.db.query(Chunk).outerjoin(Document).filter(
            Document.id.is_(None)
        ).count()
        
        if orphaned_chunks > 0:
            self.db.query(Chunk).outerjoin(Document).filter(
                Document.id.is_(None)
            ).delete(synchronize_session=False)
        
        self.db.commit()
        
        return {
            "orphaned_chunks_deleted": orphaned_chunks
        }