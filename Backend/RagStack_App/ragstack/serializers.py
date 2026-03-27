"""
RAG System Serializers
Place this in: apps/rag/serializers.py
"""

from rest_framework import serializers
from django.db import transaction
from .models import (
    Document, DocumentChunk, Query, Conversation,
    Citation, GraphEntity, GraphRelationship,
    QueryFeedback, UsageMetrics
)
from apps.users.models import User
from utils.reusable_functions import get_first_error


# ============================================================
# DOCUMENT SERIALIZERS
# ============================================================

class DocumentChunkSerializer(serializers.ModelSerializer):
    """Document chunk serializer"""
    class Meta:
        model = DocumentChunk
        fields = [
            'id', 'content', 'chunk_index', 'metadata',
            'word_count', 'created_at'
        ]
        read_only_fields = ['id', 'created_at', 'content_hash']


class DocumentListSerializer(serializers.ModelSerializer):
    """Document list serializer (minimal info)"""
    uploaded_by_name = serializers.CharField(source='uploaded_by.full_name', read_only=True)
    created_by_name = serializers.CharField(source='created_by.full_name', read_only=True)
    
    class Meta:
        model = Document
        fields = [
            'id', 'filename', 'content_type', 'file_size', 'status',
            'chunks_count', 'uploaded_by_name', 'created_by_name',
            'created_at', 'processed_at', 'tags', 'is_public'
        ]
        read_only_fields = ['id', 'created_at', 'processed_at']


class DocumentDetailSerializer(serializers.ModelSerializer):
    """Document detail serializer"""
    uploaded_by = serializers.CharField(source='uploaded_by.full_name', read_only=True)
    created_by = serializers.CharField(source='created_by.full_name', read_only=True)
    updated_by = serializers.CharField(source='updated_by.full_name', read_only=True)
    chunks = DocumentChunkSerializer(many=True, read_only=True)
    
    class Meta:
        model = Document
        fields = '__all__'
        read_only_fields = [
            'id', 'created_at', 'updated_at', 'processed_at',
            'chunks_count', 'entities_count'
        ]


class DocumentUploadSerializer(serializers.Serializer):
    """Serializer for document upload"""
    file = serializers.FileField()
    tags = serializers.ListField(
        child=serializers.CharField(max_length=50),
        required=False,
        default=list
    )
    is_public = serializers.BooleanField(default=False)
    metadata = serializers.JSONField(required=False, default=dict)
    
    def validate_file(self, value):
        """Validate file type and size"""
        allowed_types = [
            'application/pdf',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ]
        
        if value.content_type not in allowed_types:
            raise serializers.ValidationError(
                f"File type {value.content_type} not supported. "
                f"Allowed: PDF, TXT, DOCX, CSV, XLS, XLSX"
            )
        
        # Max file size: 50MB
        if value.size > 50 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 50MB")
        
        return value


# ============================================================
# QUERY & CONVERSATION SERIALIZERS
# ============================================================

class ConversationSerializer(serializers.ModelSerializer):
    """Conversation serializer"""
    user_name = serializers.CharField(source='user.full_name', read_only=True)
    created_by_name = serializers.CharField(source='created_by.full_name', read_only=True)
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'user', 'user_name', 'created_by_name', 'title',
            'started_at', 'last_activity', 'is_active', 'message_count'
        ]
        read_only_fields = ['id', 'started_at', 'last_activity', 'message_count']


class CitationSerializer(serializers.ModelSerializer):
    """Citation serializer"""
    chunk_content = serializers.CharField(source='chunk.content', read_only=True)
    document_name = serializers.CharField(source='chunk.document.filename', read_only=True)
    
    class Meta:
        model = Citation
        fields = [
            'id', 'chunk', 'chunk_content', 'document_name',
            'quote', 'relevance_score', 'position_in_answer', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class QuerySerializer(serializers.ModelSerializer):
    """Query serializer"""
    user_name = serializers.CharField(source='user.full_name', read_only=True)
    created_by_name = serializers.CharField(source='created_by.full_name', read_only=True)
    citations = CitationSerializer(many=True, read_only=True)
    
    class Meta:
        model = Query
        exclude = ('deleted',)
        read_only_fields = [
            'id', 'created_at', 'updated_at', 'processing_time',
            'retrieval_time', 'rerank_time', 'generation_time'
        ]
    
    def to_representation(self, instance):
        data = super().to_representation(instance)
        data['source_document_names'] = list(
            instance.source_documents.values_list('filename', flat=True)
        )
        return data


class QueryRequestSerializer(serializers.Serializer):
    """Serializer for query requests"""
    query = serializers.CharField(min_length=1, max_length=2000)
    conversation_id = serializers.UUIDField(required=False, allow_null=True)
    document_ids = serializers.ListField(
        child=serializers.UUIDField(),
        required=False,
        default=list
    )
    
    # RAG strategy options
    strategy = serializers.ChoiceField(
        choices=['simple', 'hybrid', 'rerank', 'agentic'],
        default='hybrid'
    )
    
    # Search parameters
    top_k = serializers.IntegerField(min_value=1, max_value=20, default=5)
    hybrid_alpha = serializers.FloatField(min_value=0.0, max_value=1.0, default=0.5)
    
    # Advanced features
    use_reranking = serializers.BooleanField(default=True)
    use_query_rewriting = serializers.BooleanField(default=True)
    include_citations = serializers.BooleanField(default=True)
    streaming = serializers.BooleanField(default=False)


class QueryResponseSerializer(serializers.Serializer):
    """Serializer for query responses"""
    query_id = serializers.UUIDField()
    query_text = serializers.CharField()
    rewritten_query = serializers.CharField(required=False, allow_blank=True)
    answer = serializers.CharField()
    
    strategy_used = serializers.CharField()
    confidence_score = serializers.FloatField()
    processing_time = serializers.FloatField()
    
    retrieval_time = serializers.FloatField()
    rerank_time = serializers.FloatField()
    generation_time = serializers.FloatField()
    
    retrieved_chunks = serializers.ListField()
    citations = CitationSerializer(many=True, required=False)
    source_documents = serializers.ListField()
    
    used_reranking = serializers.BooleanField()
    used_query_rewriting = serializers.BooleanField()


# ============================================================
# KNOWLEDGE GRAPH SERIALIZERS
# ============================================================

class GraphEntitySerializer(serializers.ModelSerializer):
    """Knowledge graph entity serializer"""
    source_document_count = serializers.SerializerMethodField()
    
    class Meta:
        model = GraphEntity
        fields = [
            'id', 'name', 'entity_type', 'description', 'properties',
            'source_document_count', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_source_document_count(self, obj):
        return obj.source_documents.count()


class GraphRelationshipSerializer(serializers.ModelSerializer):
    """Knowledge graph relationship serializer"""
    source_name = serializers.CharField(source='source.name', read_only=True)
    target_name = serializers.CharField(source='target.name', read_only=True)
    
    class Meta:
        model = GraphRelationship
        fields = [
            'id', 'source', 'source_name', 'target', 'target_name',
            'relation_type', 'properties', 'weight', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


# ============================================================
# FEEDBACK & ANALYTICS SERIALIZERS
# ============================================================

class QueryFeedbackSerializer(serializers.ModelSerializer):
    """Query feedback serializer"""
    user_name = serializers.CharField(source='user.full_name', read_only=True)
    created_by_name = serializers.CharField(source='created_by.full_name', read_only=True)
    
    class Meta:
        model = QueryFeedback
        exclude = ('deleted',)
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def validate_rating(self, value):
        if value < 1 or value > 5:
            raise serializers.ValidationError("Rating must be between 1 and 5")
        return value


class UsageMetricsSerializer(serializers.ModelSerializer):
    """Usage metrics serializer"""
    class Meta:
        model = UsageMetrics
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


class SystemStatsSerializer(serializers.Serializer):
    """System-wide statistics"""
    total_users = serializers.IntegerField()
    total_documents = serializers.IntegerField()
    total_chunks = serializers.IntegerField()
    total_queries = serializers.IntegerField()
    total_conversations = serializers.IntegerField()
    
    avg_query_time = serializers.FloatField()
    avg_confidence_score = serializers.FloatField()
    
    strategy_distribution = serializers.DictField()
    queries_last_24h = serializers.IntegerField()
    documents_last_24h = serializers.IntegerField()


class UserStatsSerializer(serializers.Serializer):
    """User-specific statistics"""
    user_id = serializers.IntegerField()
    user_name = serializers.CharField()
    
    total_documents = serializers.IntegerField()
    total_queries = serializers.IntegerField()
    total_conversations = serializers.IntegerField()
    
    avg_response_time = serializers.FloatField()
    avg_confidence_score = serializers.FloatField()
    avg_user_rating = serializers.FloatField()
    feedback_count = serializers.IntegerField()