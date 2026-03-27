"""
===============================================================
Django RAG Dependencies - Service Initialization with Groq
===============================================================
Singleton pattern for service instances
Place this in: apps/ragstack/rag_dependencies.py
===============================================================
"""

import logging
from typing import Optional, Dict
from django.conf import settings

from .rag_service import (
    EmbeddingService, HybridSearchService, RerankingService,
    QueryRewritingService, CitationService, ConversationMemoryService,
    AdvancedRAGService
)
from .llm_service import GroqLLMService, get_groq_service

logger = logging.getLogger(__name__)

# Singleton instances
_embedding_service: Optional[EmbeddingService] = None
_hybrid_search_service: Optional[HybridSearchService] = None
_reranking_service: Optional[RerankingService] = None
_query_rewriting_service: Optional[QueryRewritingService] = None
_citation_service: Optional[CitationService] = None
_conversation_memory_service: Optional[ConversationMemoryService] = None
_rag_service: Optional[AdvancedRAGService] = None


# ============================================================
# SERVICE GETTERS
# ============================================================

def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance"""
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        logger.info("✅ Initialized embedding service")
    
    return _embedding_service


def get_hybrid_search_service() -> HybridSearchService:
    """Get or create hybrid search service instance"""
    global _hybrid_search_service
    
    if _hybrid_search_service is None:
        embedding_service = get_embedding_service()
        _hybrid_search_service = HybridSearchService(embedding_service)
        logger.info("✅ Initialized hybrid search service")
    
    return _hybrid_search_service


def get_reranking_service() -> RerankingService:
    """Get or create re-ranking service instance"""
    global _reranking_service
    
    if _reranking_service is None:
        _reranking_service = RerankingService()
        logger.info("✅ Initialized re-ranking service")
    
    return _reranking_service


def get_query_rewriting_service() -> QueryRewritingService:
    """Get or create query rewriting service instance"""
    global _query_rewriting_service
    
    if _query_rewriting_service is None:
        llm_service = get_groq_service()
        _query_rewriting_service = QueryRewritingService(llm_service)
        logger.info("✅ Initialized query rewriting service with Groq")
    
    return _query_rewriting_service


def get_citation_service() -> CitationService:
    """Get or create citation service instance"""
    global _citation_service
    
    if _citation_service is None:
        _citation_service = CitationService()
        logger.info("✅ Initialized citation service")
    
    return _citation_service


def get_conversation_memory_service() -> ConversationMemoryService:
    """Get or create conversation memory service instance"""
    global _conversation_memory_service
    
    if _conversation_memory_service is None:
        _conversation_memory_service = ConversationMemoryService()
        logger.info("✅ Initialized conversation memory service")
    
    return _conversation_memory_service


def get_rag_service() -> AdvancedRAGService:
    """Get or create advanced RAG service instance with Groq"""
    global _rag_service
    
    if _rag_service is None:
        llm_service = get_groq_service()
        embedding_service = get_embedding_service()
        hybrid_search = get_hybrid_search_service()
        reranking_service = get_reranking_service()
        query_rewriting = get_query_rewriting_service()
        citation_service = get_citation_service()
        conversation_memory = get_conversation_memory_service()
        
        _rag_service = AdvancedRAGService(
            llm_service=llm_service,
            embedding_service=embedding_service,
            hybrid_search=hybrid_search,
            reranking_service=reranking_service,
            query_rewriting=query_rewriting,
            citation_service=citation_service,
            conversation_memory=conversation_memory
        )
        logger.info("✅ Initialized advanced RAG service with Groq")
    
    return _rag_service


# ============================================================
# RESET SERVICES (for testing)
# ============================================================

def reset_services():
    """Reset all service instances (useful for testing)"""
    global _embedding_service, _hybrid_search_service, _reranking_service
    global _query_rewriting_service, _citation_service, _conversation_memory_service
    global _rag_service
    
    _embedding_service = None
    _hybrid_search_service = None
    _reranking_service = None
    _query_rewriting_service = None
    _citation_service = None
    _conversation_memory_service = None
    _rag_service = None
    
    # Also reset Groq service
    from .llm_service import reset_groq_service
    reset_groq_service()
    
    logger.info("All services reset")


# ============================================================
# HEALTH CHECK
# ============================================================

def check_services_health() -> Dict:
    """
    Check health of all services
    
    Returns:
        Dict with status of each service
    """
    health = {}
    
    # Check embedding service
    try:
        emb_service = get_embedding_service()
        _ = emb_service.embed_text("test")
        health['embedding'] = 'operational'
    except Exception as e:
        health['embedding'] = f'error: {str(e)}'
    
    # Check Groq LLM service
    try:
        groq = get_groq_service()
        if groq.test_connection():
            health['llm'] = 'operational'
        else:
            health['llm'] = 'connection_failed'
    except Exception as e:
        health['llm'] = f'error: {str(e)}'
    
    # Check re-ranking service
    try:
        rerank = get_reranking_service()
        health['reranking'] = 'operational'
    except Exception as e:
        health['reranking'] = f'error: {str(e)}'
    
    # Check hybrid search
    try:
        search = get_hybrid_search_service()
        health['hybrid_search'] = 'operational'
    except Exception as e:
        health['hybrid_search'] = f'error: {str(e)}'
    
    # Overall status
    if all(v == 'operational' for v in health.values()):
        health['overall_status'] = 'healthy'
    elif any('error' in str(v) for v in health.values()):
        health['overall_status'] = 'degraded'
    else:
        health['overall_status'] = 'operational'
    
    return health


def initialize_all_services():
    """
    Initialize all services on startup
    Useful for Django app ready() hook
    """
    try:
        logger.info("Initializing RAG services...")
        
        # Initialize in dependency order
        get_groq_service()
        get_embedding_service()
        get_hybrid_search_service()
        get_reranking_service()
        get_query_rewriting_service()
        get_citation_service()
        get_conversation_memory_service()
        get_rag_service()
        
        logger.info("✅ All RAG services initialized successfully")
        
        # Run health check
        health = check_services_health()
        logger.info(f"Service health: {health}")
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise