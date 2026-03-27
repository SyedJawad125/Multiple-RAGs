"""
RAG System Models - Integrated with existing User and RBAC system
WITHOUT pgvector dependency - Uses JSONField for embeddings
Place this in: apps/ragstack/models.py
"""

import uuid
from django.db import models
from utils.reusable_classes import TimeStamps, TimeUserStamps
from User_App.users.models import User


# ============================================================
# DOCUMENT MANAGEMENT
# ============================================================

class Document(TimeUserStamps):
    """Document model with user tracking"""
    STATUS_CHOICES = (
        ('uploading', 'Uploading'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='uploaded_documents'
    )
    
    # File information
    filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    content_type = models.CharField(max_length=100)
    file_size = models.BigIntegerField()  # in bytes
    
    # Processing status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploading')
    chunks_count = models.IntegerField(default=0)
    entities_count = models.IntegerField(default=0)
    
    # Timestamps (inherited from TimeUserStamps: created_at, updated_at, created_by, updated_by)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    # Metadata
    metadata = models.JSONField(default=dict, blank=True)
    tags = models.JSONField(default=list, blank=True)  # Store as JSON array
    
    # Access control
    is_public = models.BooleanField(default=False)
    allowed_users = models.ManyToManyField(
        User,
        related_name='accessible_rag_documents',
        blank=True
    )
    
    class Meta:
        db_table = 'ragstack_documents'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.filename}"


class DocumentChunk(TimeStamps):
    """Document chunks with embeddings stored as JSON"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    
    # Chunk content
    content = models.TextField()
    chunk_index = models.IntegerField()
    
    # Vector embeddings stored as JSON array (384 dimensions for all-MiniLM-L6-v2)
    # Example: [0.123, -0.456, 0.789, ...]
    embedding = models.JSONField(default=list, blank=True)
    
    # Metadata
    metadata = models.JSONField(default=dict, blank=True)
    content_hash = models.CharField(max_length=64, db_index=True)
    word_count = models.IntegerField(default=0)
    
    class Meta:
        db_table = 'ragstack_document_chunks'
        ordering = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
            models.Index(fields=['content_hash']),
        ]
    
    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.document.filename}"


# ============================================================
# CONVERSATION & QUERY MANAGEMENT
# ============================================================

class Conversation(TimeUserStamps):
    """Conversation threads with memory"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='rag_conversations'
    )
    
    title = models.CharField(max_length=255, blank=True)
    context = models.JSONField(default=dict, blank=True)
    
    started_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    message_count = models.IntegerField(default=0)
    
    class Meta:
        db_table = 'ragstack_conversations'
        ordering = ['-last_activity']
    
    def __str__(self):
        return f"Conversation {self.title or self.id} - {self.user.full_name}"


class Query(TimeUserStamps):
    """Query history with enhanced tracking"""
    STRATEGY_CHOICES = (
        ('simple', 'Simple RAG'),
        ('hybrid', 'Hybrid Search'),
        ('rerank', 'Re-ranked'),
        ('agentic', 'Agentic RAG'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='queries',
        null=True,
        blank=True
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='rag_queries'
    )
    
    # Query details
    query_text = models.TextField()
    rewritten_query = models.TextField(blank=True)
    answer = models.TextField()
    
    # Strategy and scoring
    strategy_used = models.CharField(max_length=20, choices=STRATEGY_CHOICES)
    confidence_score = models.FloatField(default=0.0)
    
    # Source tracking
    retrieved_chunks = models.JSONField(default=list, blank=True)
    source_documents = models.ManyToManyField(
        Document,
        related_name='queries'
    )
    
    # Performance metrics
    processing_time = models.FloatField()
    retrieval_time = models.FloatField(default=0.0)
    rerank_time = models.FloatField(default=0.0)
    generation_time = models.FloatField(default=0.0)
    
    # Advanced features
    used_reranking = models.BooleanField(default=False)
    used_query_rewriting = models.BooleanField(default=False)
    hybrid_search_alpha = models.FloatField(null=True, blank=True)
    
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'ragstack_queries'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Query: {self.query_text[:50]}..."


class Citation(TimeStamps):
    """Citations linking answers to source chunks"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query = models.ForeignKey(
        Query,
        on_delete=models.CASCADE,
        related_name='citations'
    )
    chunk = models.ForeignKey(
        DocumentChunk,
        on_delete=models.CASCADE,
        related_name='citations'
    )
    
    quote = models.TextField()
    relevance_score = models.FloatField()
    position_in_answer = models.IntegerField()
    
    class Meta:
        db_table = 'ragstack_citations'
        ordering = ['query', 'position_in_answer']
    
    def __str__(self):
        return f"Citation for {self.query_id}"


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

class GraphEntity(TimeStamps):
    """Knowledge graph entities"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    name = models.CharField(max_length=255, db_index=True)
    entity_type = models.CharField(max_length=50, db_index=True)
    description = models.TextField(blank=True)
    
    # Vector embeddings stored as JSON array (384 dimensions)
    embedding = models.JSONField(default=list, blank=True)
    
    properties = models.JSONField(default=dict, blank=True)
    source_documents = models.ManyToManyField(
        Document,
        related_name='entities'
    )
    
    class Meta:
        db_table = 'ragstack_graph_entities'
        indexes = [
            models.Index(fields=['entity_type']),
            models.Index(fields=['name']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.entity_type})"


class GraphRelationship(TimeStamps):
    """Relationships between entities"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    source = models.ForeignKey(
        GraphEntity,
        on_delete=models.CASCADE,
        related_name='outgoing_relations'
    )
    target = models.ForeignKey(
        GraphEntity,
        on_delete=models.CASCADE,
        related_name='incoming_relations'
    )
    
    relation_type = models.CharField(max_length=100, db_index=True)
    properties = models.JSONField(default=dict, blank=True)
    weight = models.FloatField(default=1.0)
    
    class Meta:
        db_table = 'ragstack_graph_relationships'
        unique_together = ['source', 'target', 'relation_type']
    
    def __str__(self):
        return f"{self.source.name} -> {self.relation_type} -> {self.target.name}"


# ============================================================
# ANALYTICS & FEEDBACK
# ============================================================

class QueryFeedback(TimeUserStamps):
    """User feedback on queries"""
    RATING_CHOICES = (
        (1, 'Very Poor'),
        (2, 'Poor'),
        (3, 'Fair'),
        (4, 'Good'),
        (5, 'Excellent'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query = models.OneToOneField(
        Query,
        on_delete=models.CASCADE,
        related_name='feedback'
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='rag_feedback_given'
    )
    
    rating = models.IntegerField(choices=RATING_CHOICES)
    comment = models.TextField(blank=True)
    
    answer_helpful = models.BooleanField(null=True, blank=True)
    sources_relevant = models.BooleanField(null=True, blank=True)
    answer_accurate = models.BooleanField(null=True, blank=True)
    
    class Meta:
        db_table = 'ragstack_query_feedback'
    
    def __str__(self):
        return f"Feedback for query {self.query_id} - Rating: {self.rating}"


class UsageMetrics(TimeStamps):
    """Track daily usage metrics"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    date = models.DateField(auto_now_add=True)
    queries_count = models.IntegerField(default=0)
    documents_uploaded = models.IntegerField(default=0)
    chunks_processed = models.IntegerField(default=0)
    
    avg_response_time = models.FloatField(default=0.0)
    avg_confidence_score = models.FloatField(default=0.0)
    
    class Meta:
        db_table = 'ragstack_usage_metrics'
        ordering = ['-date']
    
    def __str__(self):
        return f"Metrics for {self.date}"
    

# Add this to your models.py file

class RerankCache(models.Model):
    """
    Cache for reranking results to avoid recomputation
    """
    query_hash = models.CharField(max_length=64, db_index=True, unique=True)
    document_hash = models.CharField(max_length=64, db_index=True)
    score = models.FloatField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Rerank Cache"
        verbose_name_plural = "Rerank Caches"
        indexes = [
            models.Index(fields=['query_hash', 'document_hash']),
            models.Index(fields=['created_at']),
        ]
    class Meta:
        db_table = 'ragstack_rerank_cache'
    def __str__(self):
        return f"RerankCache: {self.query_hash[:8]}..."