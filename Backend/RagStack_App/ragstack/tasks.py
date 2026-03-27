"""
RAG System Celery Tasks
Place this in: apps/rag/tasks.py
"""

from celery import shared_task
import logging

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_document_task(self, document_id):
    """
    Background task to process uploaded document
    - Extract text
    - Chunk text
    - Generate embeddings
    - Store in database
    """
    try:
        from .document_processor import DocumentProcessor
        from .models import Document
        from django.utils import timezone
        
        logger.info(f"Starting document processing for {document_id}")
        
        # Get document
        document = Document.objects.get(id=document_id)
        document.status = 'processing'
        document.save()
        
        # Process document
        processor = DocumentProcessor()
        processor.process_document(document_id)
        
        # Update status
        document.refresh_from_db()
        if document.status != 'failed':
            document.status = 'completed'
            document.processed_at = timezone.now()
            document.save()
        
        logger.info(f"Document processing completed for {document_id}")
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}", exc_info=True)
        
        try:
            document = Document.objects.get(id=document_id)
            document.status = 'failed'
            document.metadata['error'] = str(e)
            document.save()
        except Exception:
            pass
        
        # Retry task
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))


@shared_task
def cleanup_old_conversations(days=90):
    """
    Cleanup old inactive conversations
    """
    try:
        from datetime import timedelta
        from django.utils import timezone
        from .models import Conversation
        
        cutoff_date = timezone.now() - timedelta(days=days)
        
        old_conversations = Conversation.objects.filter(
            is_active=False,
            last_activity__lt=cutoff_date
        )
        
        count = old_conversations.count()
        old_conversations.delete()
        
        logger.info(f"Cleaned up {count} old conversations")
        
    except Exception as e:
        logger.error(f"Conversation cleanup failed: {str(e)}", exc_info=True)


@shared_task
def generate_daily_metrics():
    """
    Generate daily usage metrics
    """
    try:
        from datetime import date
        from django.db.models import Count, Avg
        from .models import Query, Document, UsageMetrics
        
        today = date.today()
        
        # Calculate metrics
        queries_count = Query.objects.filter(
            created_at__date=today,
            deleted=False
        ).count()
        
        documents_uploaded = Document.objects.filter(
            created_at__date=today,
            deleted=False
        ).count()
        
        chunks_processed = Document.objects.filter(
            created_at__date=today,
            deleted=False
        ).aggregate(total=Count('chunks'))['total'] or 0
        
        avg_response_time = Query.objects.filter(
            created_at__date=today,
            deleted=False
        ).aggregate(avg=Avg('processing_time'))['avg'] or 0.0
        
        avg_confidence = Query.objects.filter(
            created_at__date=today,
            deleted=False
        ).aggregate(avg=Avg('confidence_score'))['avg'] or 0.0
        
        # Save metrics
        UsageMetrics.objects.create(
            date=today,
            queries_count=queries_count,
            documents_uploaded=documents_uploaded,
            chunks_processed=chunks_processed,
            avg_response_time=avg_response_time,
            avg_confidence_score=avg_confidence
        )
        
        logger.info(f"Generated daily metrics for {today}")
        
    except Exception as e:
        logger.error(f"Metrics generation failed: {str(e)}", exc_info=True)


@shared_task
def extract_knowledge_graph(document_id):
    """
    Extract entities and relationships from document
    """
    try:
        from .models import Document, GraphEntity, GraphRelationship
        from .rag_service import extract_entities_and_relationships
        
        logger.info(f"Starting knowledge graph extraction for {document_id}")
        
        document = Document.objects.get(id=document_id)
        
        # Extract entities and relationships
        result = extract_entities_and_relationships(document)
        
        logger.info(
            f"Extracted {result['entities_count']} entities and "
            f"{result['relationships_count']} relationships"
        )
        
        # Update document
        document.entities_count = result['entities_count']
        document.save()
        
    except Exception as e:
        logger.error(f"Knowledge graph extraction failed: {str(e)}", exc_info=True)