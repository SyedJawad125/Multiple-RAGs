"""
RAG System Filters
Place this in: apps/rag/filters.py
"""

import django_filters
from django.db.models import Q
from .models import Document, Query, Conversation, GraphEntity


class DocumentFilter(django_filters.FilterSet):
    """Filter for documents"""
    search = django_filters.CharFilter(method='search_filter', label='Search')
    filename = django_filters.CharFilter(lookup_expr='icontains')
    status = django_filters.ChoiceFilter(choices=Document.STATUS_CHOICES)
    content_type = django_filters.CharFilter(lookup_expr='icontains')
    is_public = django_filters.BooleanFilter()
    uploaded_after = django_filters.DateTimeFilter(
        field_name='created_at',
        lookup_expr='gte'
    )
    uploaded_before = django_filters.DateTimeFilter(
        field_name='created_at',
        lookup_expr='lte'
    )
    
    class Meta:
        model = Document
        fields = []
    
    def search_filter(self, queryset, name, value):
        """Search in filename and tags"""
        return queryset.filter(
            Q(filename__icontains=value) |
            Q(tags__icontains=value) |
            Q(metadata__icontains=value)
        )


class QueryFilter(django_filters.FilterSet):
    """Filter for queries"""
    search = django_filters.CharFilter(method='search_filter', label='Search')
    strategy = django_filters.ChoiceFilter(
        field_name='strategy_used',
        choices=Query.STRATEGY_CHOICES
    )
    confidence_min = django_filters.NumberFilter(
        field_name='confidence_score',
        lookup_expr='gte'
    )
    confidence_max = django_filters.NumberFilter(
        field_name='confidence_score',
        lookup_expr='lte'
    )
    used_reranking = django_filters.BooleanFilter()
    used_query_rewriting = django_filters.BooleanFilter()
    created_after = django_filters.DateTimeFilter(
        field_name='created_at',
        lookup_expr='gte'
    )
    created_before = django_filters.DateTimeFilter(
        field_name='created_at',
        lookup_expr='lte'
    )
    
    class Meta:
        model = Query
        fields = []
    
    def search_filter(self, queryset, name, value):
        """Search in query text and answer"""
        return queryset.filter(
            Q(query_text__icontains=value) |
            Q(answer__icontains=value)
        )


class ConversationFilter(django_filters.FilterSet):
    """Filter for conversations"""
    search = django_filters.CharFilter(method='search_filter', label='Search')
    is_active = django_filters.BooleanFilter()
    started_after = django_filters.DateTimeFilter(
        field_name='started_at',
        lookup_expr='gte'
    )
    started_before = django_filters.DateTimeFilter(
        field_name='started_at',
        lookup_expr='lte'
    )
    
    class Meta:
        model = Conversation
        fields = []
    
    def search_filter(self, queryset, name, value):
        """Search in title"""
        return queryset.filter(title__icontains=value)


class GraphEntityFilter(django_filters.FilterSet):
    """Filter for knowledge graph entities"""
    search = django_filters.CharFilter(method='search_filter', label='Search')
    name = django_filters.CharFilter(lookup_expr='icontains')
    entity_type = django_filters.CharFilter()
    
    class Meta:
        model = GraphEntity
        fields = []
    
    def search_filter(self, queryset, name, value):
        """Search in name and description"""
        return queryset.filter(
            Q(name__icontains=value) |
            Q(description__icontains=value)
        )