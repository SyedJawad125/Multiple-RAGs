"""
RAG System URLs
Place this in: apps/rag/urls.py
"""

from django.urls import path
from .views import (
    DocumentView, DocumentDetailView, DocumentShareView,
    QueryView, ExecuteQueryView, QueryCitationsView,
    ConversationView,
    GraphEntityView, GraphExploreView,
    AnalyticsView, PerformanceMetricsView,
    QueryFeedbackView,
    HealthCheckView
)

urlpatterns = [
    # Health Check
    path('v1/health/', HealthCheckView.as_view()),
    
    # Document Management
    path('v1/document/', DocumentView.as_view()),
    path('v1/document/detail/', DocumentDetailView.as_view()),
    path('v1/document/share/', DocumentShareView.as_view()),
    
    # Query Execution
    path('v1/query/', QueryView.as_view()),
    path('v1/query/execute/', ExecuteQueryView.as_view()),
    path('v1/query/citations/', QueryCitationsView.as_view()),
    
    # Conversations
    path('v1/conversation/', ConversationView.as_view()),
    
    # Knowledge Graph
    path('v1/graph/entity/', GraphEntityView.as_view()),
    path('v1/graph/explore/', GraphExploreView.as_view()),
    
    # Analytics
    path('v1/analytics/', AnalyticsView.as_view()),
    path('v1/analytics/performance/', PerformanceMetricsView.as_view()),
    
    # Feedback
    path('v1/feedback/', QueryFeedbackView.as_view()),
]

"""
API Endpoints Summary:

DOCUMENT MANAGEMENT:
- POST   /api/rag/v1/document/           - Upload document (requires: upload_document)
- GET    /api/rag/v1/document/           - List documents (requires: read_document)
- PATCH  /api/rag/v1/document/           - Update document (requires: update_document)
- DELETE /api/rag/v1/document/           - Delete document (requires: delete_document)
- GET    /api/rag/v1/document/detail/    - Get document details (requires: view_processing_status)
- POST   /api/rag/v1/document/share/     - Share document (requires: share_document)

QUERY EXECUTION:
- GET    /api/rag/v1/query/              - List queries (requires: view_query_history)
- POST   /api/rag/v1/query/execute/      - Execute RAG query (requires: execute_rag_query)
- GET    /api/rag/v1/query/citations/    - Get citations (requires: view_citations)

CONVERSATIONS:
- POST   /api/rag/v1/conversation/       - Create conversation (requires: create_conversation)
- GET    /api/rag/v1/conversation/       - List conversations (requires: read_conversation)
- PATCH  /api/rag/v1/conversation/       - Update conversation (requires: update_conversation)
- DELETE /api/rag/v1/conversation/       - Delete conversation (requires: delete_conversation)

KNOWLEDGE GRAPH:
- GET    /api/rag/v1/graph/entity/       - List entities (requires: view_knowledge_graph)
- GET    /api/rag/v1/graph/explore/      - Explore graph (requires: explore_entities)

ANALYTICS:
- GET    /api/rag/v1/analytics/          - Get user analytics (requires: view_rag_analytics)
- GET    /api/rag/v1/analytics/performance/ - Get performance metrics (requires: view_performance_metrics)

FEEDBACK:
- POST   /api/rag/v1/feedback/           - Submit feedback (requires: submit_query_feedback)
- GET    /api/rag/v1/feedback/           - List feedback

HEALTH:
- GET    /api/rag/v1/health/             - Health check (public)
"""