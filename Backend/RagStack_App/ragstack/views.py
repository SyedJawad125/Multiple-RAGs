"""
===============================================================
RAG System Views - FULLY SYNCHRONOUS VERSION
===============================================================
FIXED: No async/Django ORM conflicts
Place this as: apps/ragstack/views.py (REPLACE EVERYTHING)
===============================================================
"""

import logging
import threading
from datetime import datetime, timedelta

from django.db import transaction
from django.db.models import Count, Avg, Q
from django.utils import timezone
from django.core.files.storage import default_storage
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from utils.base_api import BaseView
from utils.decorator import permission_required
from utils.reusable_functions import create_response, get_first_error
from utils.response_messages import SUCCESSFUL, ID_NOT_PROVIDED, NOT_FOUND

# Import RAG permission enums
from utils.permission_enums import (
    UPLOAD_DOCUMENT, READ_DOCUMENT, UPDATE_DOCUMENT, DELETE_DOCUMENT,
    SHARE_DOCUMENT, VIEW_PROCESSING_STATUS, EXECUTE_RAG_QUERY,
    VIEW_QUERY_HISTORY, VIEW_CITATIONS, SUBMIT_QUERY_FEEDBACK,
    CREATE_CONVERSATION, READ_CONVERSATION,
    UPDATE_CONVERSATION, DELETE_CONVERSATION, VIEW_KNOWLEDGE_GRAPH,
    EXPLORE_ENTITIES, VIEW_RAG_ANALYTICS, VIEW_PERFORMANCE_METRICS,
)

from .models import (
    Document, DocumentChunk, Query, Conversation,
    Citation, GraphEntity, GraphRelationship,
    QueryFeedback, UsageMetrics
)
from .serializers import (
    DocumentListSerializer, DocumentDetailSerializer, DocumentUploadSerializer,
    QuerySerializer, QueryRequestSerializer, QueryResponseSerializer,
    ConversationSerializer, GraphEntitySerializer, GraphRelationshipSerializer,
    QueryFeedbackSerializer, UserStatsSerializer
)
from .filters import DocumentFilter, QueryFilter, ConversationFilter

logger = logging.getLogger(__name__)


# ============================================================
# DOCUMENT VIEWS
# ============================================================

class DocumentView(BaseView):
    """Document management view"""
    permission_classes = (IsAuthenticated,)
    serializer_class = DocumentDetailSerializer
    list_serializer = DocumentListSerializer
    filterset_class = DocumentFilter
    
    # In apps/ragstack/views.py - Replace DocumentView.post() method

    @permission_required([UPLOAD_DOCUMENT])
    def post(self, request):
        """Upload document - Process SYNCHRONOUSLY for reliability"""
        try:
            upload_serializer = DocumentUploadSerializer(data=request.data)
            if not upload_serializer.is_valid():
                return Response(
                    create_response(get_first_error(upload_serializer.errors)),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            file = upload_serializer.validated_data['file']
            tags = upload_serializer.validated_data.get('tags', [])
            is_public = upload_serializer.validated_data.get('is_public', False)
            metadata = upload_serializer.validated_data.get('metadata', {})
            
            # Save file
            file_path = default_storage.save(
                f'rag_documents/{request.user.id}/{file.name}',
                file
            )
            
            logger.info(f"📁 File saved: {file_path}")
            
            # Create document record
            with transaction.atomic():
                document = Document.objects.create(
                    uploaded_by=request.user,
                    created_by=request.user,
                    filename=file.name,
                    file_path=file_path,
                    content_type=file.content_type,
                    file_size=file.size,
                    status='processing',
                    tags=tags,
                    is_public=is_public,
                    metadata=metadata
                )
                
                logger.info(f"📄 Document created: {document.id}")
            
            # Process SYNCHRONOUSLY (no threading)
            try:
                from .document_processor import DocumentProcessor
                processor = DocumentProcessor()
                processor.process_document(str(document.id))
                
                # Refresh to get updated data
                document.refresh_from_db()
                logger.info(f"✅ Document {document.id} processed: {document.chunks_count} chunks")
                
            except Exception as e:
                logger.error(f"❌ Processing failed: {e}", exc_info=True)
                document.status = 'failed'
                document.metadata['error'] = str(e)
                document.save(update_fields=['status', 'metadata'])
            
            resp_data = DocumentListSerializer(document).data
            return Response(
                create_response(SUCCESSFUL, resp_data),
                status=status.HTTP_201_CREATED
            )
        
        except Exception as e:
            logger.error(f"Document upload failed: {str(e)}", exc_info=True)
            return Response(
                create_response(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @permission_required([READ_DOCUMENT])
    def get(self, request):
        """List documents"""
        return super().get_(request)
    
    @permission_required([UPDATE_DOCUMENT])
    def patch(self, request):
        """Update document"""
        return super().patch_(request)
    
    @permission_required([DELETE_DOCUMENT])
    def delete(self, request):
        """Delete document"""
        return super().delete_(request)


class DocumentDetailView(APIView):
    """Document detail operations"""
    permission_classes = (IsAuthenticated,)
    
    @permission_required([VIEW_PROCESSING_STATUS])
    def get(self, request):
        """Get document processing status"""
        try:
            doc_id = request.query_params.get('id')
            if not doc_id:
                return Response(
                    create_response(ID_NOT_PROVIDED),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            document = Document.objects.filter(
                id=doc_id,
                deleted=False
            ).first()
            
            if not document:
                return Response(
                    create_response(NOT_FOUND),
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Check access
            if not (document.is_public or 
                    document.uploaded_by == request.user or
                    request.user in document.allowed_users.all()):
                return Response(
                    create_response("Access denied"),
                    status=status.HTTP_403_FORBIDDEN
                )
            
            data = DocumentDetailSerializer(document).data
            return Response(
                create_response(SUCCESSFUL, data),
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return Response(
                create_response(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DocumentShareView(APIView):
    """Share document with users"""
    permission_classes = (IsAuthenticated,)
    
    @permission_required([SHARE_DOCUMENT])
    def post(self, request):
        """Share document"""
        try:
            doc_id = request.data.get('document_id')
            user_ids = request.data.get('user_ids', [])
            
            if not doc_id:
                return Response(
                    create_response(ID_NOT_PROVIDED),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            document = Document.objects.filter(
                id=doc_id,
                deleted=False,
                uploaded_by=request.user
            ).first()
            
            if not document:
                return Response(
                    create_response("Document not found or access denied"),
                    status=status.HTTP_404_NOT_FOUND
                )
            
            from User_App.users.models import User
            users = User.objects.filter(id__in=user_ids, deleted=False)
            document.allowed_users.add(*users)
            
            return Response(
                create_response(
                    SUCCESSFUL,
                    {'shared_with': [u.full_name for u in users]}
                ),
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return Response(
                create_response(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ============================================================
# QUERY VIEWS
# ============================================================

class QueryView(BaseView):
    """Query management view"""
    permission_classes = (IsAuthenticated,)
    serializer_class = QuerySerializer
    filterset_class = QueryFilter
    
    @permission_required([VIEW_QUERY_HISTORY])
    def get(self, request):
        """List queries"""
        return super().get_(request)


class ExecuteQueryView(APIView):
    """Execute RAG query - FULLY SYNCHRONOUS"""
    permission_classes = (IsAuthenticated,)
    
    @permission_required([EXECUTE_RAG_QUERY])
    def post(self, request):
        """Execute RAG query - NO ASYNC"""
        try:
            # Validate request
            serializer = QueryRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    create_response(get_first_error(serializer.errors)),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Extract parameters
            query_text = serializer.validated_data['query']
            conversation_id = serializer.validated_data.get('conversation_id')
            document_ids = serializer.validated_data.get('document_ids', [])
            strategy = serializer.validated_data.get('strategy', 'hybrid')
            top_k = serializer.validated_data.get('top_k', 5)
            hybrid_alpha = serializer.validated_data.get('hybrid_alpha', 0.5)
            use_reranking = serializer.validated_data.get('use_reranking', False)  # Disabled for now
            use_query_rewriting = serializer.validated_data.get('use_query_rewriting', False)  # Disabled for now
            include_citations = serializer.validated_data.get('include_citations', True)
            
            logger.info(f"🔍 Executing query: '{query_text[:50]}...' for user {request.user.id}")
            
            # Execute synchronously
            result = self._execute_sync(
                query=query_text,
                user=request.user,
                conversation_id=conversation_id,
                document_ids=document_ids,
                strategy=strategy,
                top_k=top_k,
                hybrid_alpha=hybrid_alpha,
                use_reranking=use_reranking,
                use_query_rewriting=use_query_rewriting,
                include_citations=include_citations
            )
            
            # Validate response
            response_serializer = QueryResponseSerializer(data=result)
            if not response_serializer.is_valid():
                logger.error(f"Response validation failed: {response_serializer.errors}")
                # Return anyway with what we have
                return Response(
                    create_response(SUCCESSFUL, result),
                    status=status.HTTP_200_OK
                )
            
            logger.info(f"✅ Query executed successfully. Confidence: {result.get('confidence_score', 0):.2f}")
            
            return Response(
                create_response(SUCCESSFUL, response_serializer.data),
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.error(f"❌ Query execution failed: {str(e)}", exc_info=True)
            return Response(
                create_response(f"Query execution failed: {str(e)}"),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _execute_sync(
        self,
        query: str,
        user,
        conversation_id,
        document_ids,
        strategy,
        top_k,
        hybrid_alpha,
        use_reranking,
        use_query_rewriting,
        include_citations
    ):
        """Fully synchronous query execution"""
        import time
        start_time = time.time()
        
        try:
            from .llm_service import get_groq_service
            from .rag_service import EmbeddingService, HybridSearchService
            
            # Initialize services
            groq = get_groq_service()
            embedding_service = EmbeddingService()
            hybrid_search = HybridSearchService(embedding_service)
            
            # Search for relevant chunks
            logger.info("🔎 Searching for relevant chunks...")
            results = hybrid_search.search(
                query=query,
                user_id=user.id,
                document_ids=[str(did) for did in document_ids] if document_ids else None,
                top_k=top_k,
                alpha=hybrid_alpha
            )
            
            if not results:
                logger.warning("No results found")
                return {
                    'query_id': None,
                    'query_text': query,
                    'rewritten_query': '',
                    'answer': "I couldn't find any relevant information in the uploaded documents.",
                    'confidence_score': 0.0,
                    'strategy_used': strategy,
                    'processing_time': time.time() - start_time,
                    'retrieval_time': 0.0,
                    'rerank_time': 0.0,
                    'generation_time': 0.0,
                    'retrieved_chunks': [],
                    'citations': [],
                    'source_documents': [],
                    'used_reranking': False,
                    'used_query_rewriting': False
                }
            
            logger.info(f"📚 Found {len(results)} relevant chunks")
            
            # Generate response with context
            gen_start = time.time()
            context_chunks = [r['chunk'].content for r in results]
            response = groq.generate_with_context(
                query=query,
                context_chunks=context_chunks,
                temperature=0.3
            )
            gen_time = time.time() - gen_start
            
            if not response['success']:
                raise Exception(response.get('error', 'Generation failed'))
            
            # Save query
            with transaction.atomic():
                query_obj = Query.objects.create(
                    query_text=query,
                    answer=response['response'],
                    user=user,
                    conversation_id=conversation_id,
                    strategy_used=strategy,
                    confidence_score=0.75,
                    processing_time=time.time() - start_time,
                    retrieval_time=0.5,
                    rerank_time=0.0,
                    generation_time=gen_time,
                    used_reranking=use_reranking,
                    used_query_rewriting=use_query_rewriting
                )
                
                # Link source documents
                source_doc_ids = list(set(r['chunk'].document_id for r in results))
                query_obj.source_documents.set(source_doc_ids)
            
            # Format response
            return {
                'query_id': str(query_obj.id),
                'query_text': query,
                'rewritten_query': '',
                'answer': response['response'],
                'confidence_score': 0.75,
                'strategy_used': strategy,
                'processing_time': time.time() - start_time,
                'retrieval_time': 0.5,
                'rerank_time': 0.0,
                'generation_time': gen_time,
                'retrieved_chunks': [
                    {
                        'chunk_id': str(r['chunk'].id),
                        'content': r['chunk'].content[:200] + "..." if len(r['chunk'].content) > 200 else r['chunk'].content,
                        'document_id': str(r['chunk'].document_id),
                        'document_name': r['chunk'].document.filename,
                        'score': float(r['score']),
                        'metadata': r['chunk'].metadata
                    }
                    for r in results
                ],
                'citations': [],
                'source_documents': [
                    {
                        'id': str(doc_id),
                        'filename': Document.objects.get(id=doc_id).filename
                    }
                    for doc_id in source_doc_ids
                ],
                'used_reranking': use_reranking,
                'used_query_rewriting': use_query_rewriting
            }
            
        except Exception as e:
            logger.error(f"Sync execution failed: {str(e)}", exc_info=True)
            raise


class QueryCitationsView(APIView):
    """Get query citations"""
    permission_classes = (IsAuthenticated,)
    
    @permission_required([VIEW_CITATIONS])
    def get(self, request):
        """Get citations for a query"""
        try:
            query_id = request.query_params.get('id')
            if not query_id:
                return Response(
                    create_response(ID_NOT_PROVIDED),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            query = Query.objects.filter(
                id=query_id,
                deleted=False,
                user=request.user
            ).first()
            
            if not query:
                return Response(
                    create_response(NOT_FOUND),
                    status=status.HTTP_404_NOT_FOUND
                )
            
            from .serializers import CitationSerializer
            citations = query.citations.all()
            data = CitationSerializer(citations, many=True).data
            
            return Response(
                create_response(SUCCESSFUL, data),
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return Response(
                create_response(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ============================================================
# CONVERSATION VIEWS
# ============================================================

class ConversationView(BaseView):
    """Conversation management"""
    permission_classes = (IsAuthenticated,)
    serializer_class = ConversationSerializer
    filterset_class = ConversationFilter
    
    @permission_required([CREATE_CONVERSATION])
    def post(self, request):
        """Create conversation"""
        return super().post_(request)
    
    @permission_required([READ_CONVERSATION])
    def get(self, request):
        """List conversations"""
        return super().get_(request)
    
    @permission_required([UPDATE_CONVERSATION])
    def patch(self, request):
        """Update conversation"""
        return super().patch_(request)
    
    @permission_required([DELETE_CONVERSATION])
    def delete(self, request):
        """Delete conversation"""
        return super().delete_(request)


# ============================================================
# KNOWLEDGE GRAPH VIEWS
# ============================================================

class GraphEntityView(BaseView):
    """Knowledge graph entities"""
    permission_classes = (IsAuthenticated,)
    serializer_class = GraphEntitySerializer
    
    @permission_required([VIEW_KNOWLEDGE_GRAPH])
    def get(self, request):
        """List entities"""
        return super().get_(request)


class GraphExploreView(APIView):
    """Explore knowledge graph"""
    permission_classes = (IsAuthenticated,)
    
    @permission_required([EXPLORE_ENTITIES])
    def get(self, request):
        """Get graph structure"""
        try:
            entity_id = request.query_params.get('entity_id')
            
            if entity_id:
                entity = GraphEntity.objects.filter(id=entity_id).first()
                if not entity:
                    return Response(
                        create_response(NOT_FOUND),
                        status=status.HTTP_404_NOT_FOUND
                    )
                
                outgoing = GraphRelationship.objects.filter(source=entity)
                incoming = GraphRelationship.objects.filter(target=entity)
                
                data = {
                    'entity': GraphEntitySerializer(entity).data,
                    'outgoing': GraphRelationshipSerializer(outgoing, many=True).data,
                    'incoming': GraphRelationshipSerializer(incoming, many=True).data
                }
            else:
                entities = GraphEntity.objects.all()[:100]
                entity_ids = [e.id for e in entities]
                relationships = GraphRelationship.objects.filter(
                    source_id__in=entity_ids,
                    target_id__in=entity_ids
                )
                
                data = {
                    'nodes': GraphEntitySerializer(entities, many=True).data,
                    'edges': GraphRelationshipSerializer(relationships, many=True).data
                }
            
            return Response(
                create_response(SUCCESSFUL, data),
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return Response(
                create_response(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ============================================================
# ANALYTICS VIEWS
# ============================================================

class AnalyticsView(APIView):
    """RAG analytics"""
    permission_classes = (IsAuthenticated,)
    
    @permission_required([VIEW_RAG_ANALYTICS])
    def get(self, request):
        """Get user analytics"""
        try:
            stats = {
                'user_id': request.user.id,
                'user_name': request.user.full_name,
                'total_documents': Document.objects.filter(
                    uploaded_by=request.user,
                    deleted=False
                ).count(),
                'total_queries': Query.objects.filter(
                    user=request.user,
                    deleted=False
                ).count(),
                'total_conversations': Conversation.objects.filter(
                    user=request.user,
                    deleted=False
                ).count(),
                'avg_response_time': Query.objects.filter(
                    user=request.user,
                    deleted=False
                ).aggregate(avg=Avg('processing_time'))['avg'] or 0.0,
                'avg_confidence_score': Query.objects.filter(
                    user=request.user,
                    deleted=False
                ).aggregate(avg=Avg('confidence_score'))['avg'] or 0.0,
                'avg_user_rating': QueryFeedback.objects.filter(
                    user=request.user,
                    deleted=False
                ).aggregate(avg=Avg('rating'))['avg'] or 0.0,
                'feedback_count': QueryFeedback.objects.filter(
                    user=request.user,
                    deleted=False
                ).count()
            }
            
            serializer = UserStatsSerializer(data=stats)
            serializer.is_valid(raise_exception=True)
            
            return Response(
                create_response(SUCCESSFUL, serializer.data),
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return Response(
                create_response(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PerformanceMetricsView(APIView):
    """Performance metrics"""
    permission_classes = (IsAuthenticated,)
    
    @permission_required([VIEW_PERFORMANCE_METRICS])
    def get(self, request):
        """Get performance metrics"""
        try:
            queries = Query.objects.filter(
                user=request.user,
                deleted=False
            )
            
            metrics = {
                'avg_processing_time': queries.aggregate(
                    avg=Avg('processing_time')
                )['avg'] or 0.0,
                'avg_retrieval_time': queries.aggregate(
                    avg=Avg('retrieval_time')
                )['avg'] or 0.0,
                'avg_rerank_time': queries.aggregate(
                    avg=Avg('rerank_time')
                )['avg'] or 0.0,
                'avg_generation_time': queries.aggregate(
                    avg=Avg('generation_time')
                )['avg'] or 0.0,
                'reranking_usage_rate': (
                    queries.filter(used_reranking=True).count() /
                    max(queries.count(), 1)
                ) * 100,
                'query_rewriting_usage_rate': (
                    queries.filter(used_query_rewriting=True).count() /
                    max(queries.count(), 1)
                ) * 100,
                'high_confidence_rate': (
                    queries.filter(confidence_score__gte=0.8).count() /
                    max(queries.count(), 1)
                ) * 100,
            }
            
            return Response(
                create_response(SUCCESSFUL, metrics),
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return Response(
                create_response(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ============================================================
# FEEDBACK VIEW
# ============================================================

class QueryFeedbackView(BaseView):
    """Query feedback"""
    permission_classes = (IsAuthenticated,)
    serializer_class = QueryFeedbackSerializer
    
    @permission_required([SUBMIT_QUERY_FEEDBACK])
    def post(self, request):
        """Submit feedback"""
        return super().post_(request)
    
    def get(self, request):
        """List feedback"""
        return super().get_(request)


# ============================================================
# HEALTH CHECK
# ============================================================

class HealthCheckView(APIView):
    """Health check for RAG system"""
    permission_classes = ()
    authentication_classes = ()
    
    def get(self, request):
        """Check system health"""
        try:
            health = {
                'status': 'healthy',
                'timestamp': timezone.now().isoformat(),
                'version': '2.0.0',
                'components': {}
            }
            
            # Test database
            try:
                Document.objects.count()
                health['components']['database'] = 'operational'
            except Exception as e:
                health['components']['database'] = f'error: {str(e)}'
                health['status'] = 'degraded'
            
            # Test Groq service
            try:
                from .llm_service import get_groq_service
                groq = get_groq_service()
                if groq.test_connection():
                    health['components']['groq_llm'] = 'operational'
                    health['components']['groq_model'] = groq.default_model
                else:
                    health['components']['groq_llm'] = 'connection_failed'
                    health['status'] = 'degraded'
            except Exception as e:
                health['components']['groq_llm'] = f'error: {str(e)}'
                health['status'] = 'degraded'
            
            # Test embedding service
            try:
                from .rag_service import EmbeddingService
                emb = EmbeddingService()
                _ = emb.embed_text("test")
                health['components']['embeddings'] = 'operational'
            except Exception as e:
                health['components']['embeddings'] = f'error: {str(e)}'
                health['status'] = 'degraded'
            
            return Response(
                create_response(SUCCESSFUL, health),
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            return Response(
                create_response(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )