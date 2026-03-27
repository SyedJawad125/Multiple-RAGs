# apps/ragstack/rag_service.py - FIXED VERSION
"""
===============================================================
Advanced RAG Service - FIXED: No organization_id dependency
===============================================================
"""

import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from django.db.models import Q, F
from django.core.cache import cache
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import tiktoken

from .models import (
    Document, DocumentChunk, Query, Conversation,
    Citation, RerankCache
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings"""
    
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        self.dimension = 384
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class HybridSearchService:
    """Hybrid search combining BM25 (keyword) and vector search"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    def search(
        self,
        query: str,
        organization_id: str = None,  # Now optional
        user_id: int = None,  # Added user_id for filtering
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with alpha blending
        FIXED: Works without organization_id field
        """
        start_time = time.time()
        
        # Build base queryset - FIXED: Use user instead of organization
        queryset = DocumentChunk.objects.select_related('document').filter(
            document__deleted=False
        )
        
        # Filter by user if provided (documents uploaded by user or public)
        if user_id:
            queryset = queryset.filter(
                Q(document__uploaded_by_id=user_id) | 
                Q(document__is_public=True)
            )
        
        if document_ids:
            queryset = queryset.filter(document_id__in=document_ids)
        
        chunks = list(queryset)
        
        if not chunks:
            logger.warning("No chunks found for search")
            return []
        
        logger.info(f"Searching through {len(chunks)} chunks")
        
        # 1. BM25 Search (keyword-based)
        bm25_scores = self._bm25_search(query, chunks) if alpha < 1.0 else [0] * len(chunks)
        
        # 2. Vector Search (semantic)
        vector_scores = self._vector_search(query, chunks) if alpha > 0.0 else [0] * len(chunks)
        
        # 3. Hybrid scoring
        hybrid_scores = []
        for i in range(len(chunks)):
            score = (1 - alpha) * bm25_scores[i] + alpha * vector_scores[i]
            hybrid_scores.append({
                'chunk': chunks[i],
                'score': score,
                'bm25_score': bm25_scores[i],
                'vector_score': vector_scores[i]
            })
        
        # Sort by hybrid score
        hybrid_scores.sort(key=lambda x: x['score'], reverse=True)
        
        results = hybrid_scores[:top_k]
        
        search_time = time.time() - start_time
        logger.info(f"✅ Hybrid search completed in {search_time:.2f}s, found {len(results)} results")
        
        return results
    
    def _bm25_search(self, query: str, chunks: List[DocumentChunk]) -> List[float]:
        """BM25 keyword search"""
        try:
            # Tokenize documents
            tokenized_corpus = [chunk.content.lower().split() for chunk in chunks]
            
            # Build BM25 index
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Search
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            
            # Normalize scores to 0-1
            if max(scores) > 0:
                scores = scores / max(scores)
            
            return scores.tolist()
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return [0.0] * len(chunks)
    
    def _vector_search(self, query: str, chunks: List[DocumentChunk]) -> List[float]:
        """Vector similarity search"""
        try:
            query_embedding = np.array(self.embedding_service.embed_text(query))
            
            scores = []
            for chunk in chunks:
                if not chunk.embedding:
                    scores.append(0.0)
                    continue
                    
                chunk_embedding = np.array(chunk.embedding)
                
                # Cosine similarity
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding) + 1e-8
                )
                
                # Convert to 0-1 range
                score = (similarity + 1) / 2
                scores.append(score)
            
            return scores
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return [0.0] * len(chunks)


class RerankingService:
    """Re-ranking service using cross-encoder"""
    
    def __init__(self):
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info("✅ Reranking service initialized")
        except Exception as e:
            logger.warning(f"⚠️ Reranking service init failed: {e}")
            self.model = None
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Re-rank search results using cross-encoder"""
        if not self.model or not results:
            return results[:top_k]
            
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(query, [r['chunk'].id for r in results])
            cached = cache.get(cache_key)
            
            if cached:
                logger.info("Using cached re-ranking results")
                return cached
            
            # Prepare pairs for cross-encoder
            pairs = [[query, result['chunk'].content] for result in results]
            
            # Get cross-encoder scores
            rerank_scores = self.model.predict(pairs)
            
            # Update scores
            for i, result in enumerate(results):
                result['rerank_score'] = float(rerank_scores[i])
                result['original_score'] = result['score']
                result['score'] = result['rerank_score']
            
            # Sort by new scores
            results.sort(key=lambda x: x['score'], reverse=True)
            reranked = results[:top_k]
            
            # Cache results
            cache.set(cache_key, reranked, timeout=3600)  # 1 hour
            
            rerank_time = time.time() - start_time
            logger.info(f"✅ Re-ranking completed in {rerank_time:.2f}s")
            
            return reranked
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return results[:top_k]
    
    def _get_cache_key(self, query: str, chunk_ids: List[str]) -> str:
        """Generate cache key for re-ranking"""
        content = f"{query}:{''.join(str(cid) for cid in sorted(chunk_ids))}"
        return f"rerank:{hashlib.md5(content.encode()).hexdigest()}"


class QueryRewritingService:
    """Query rewriting for better retrieval"""
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
    
    async def rewrite_query(
        self,
        query: str,
        conversation_context: Optional[str] = None
    ) -> str:
        """Rewrite query for better retrieval"""
        try:
            # Use the llm_service's rewrite_query method if available
            if hasattr(self.llm_service, 'rewrite_query'):
                return self.llm_service.rewrite_query(query, conversation_context)
            
            # Otherwise use generate method
            system_prompt = """You are a query optimization expert. 
Your task is to rewrite user queries to improve information retrieval.

Guidelines:
1. Expand abbreviations and acronyms
2. Add relevant context from conversation
3. Make implicit intent explicit
4. Keep technical terms intact
5. Return ONLY the rewritten query, no explanation"""
            
            user_prompt = f"Original query: {query}"
            
            if conversation_context:
                user_prompt += f"\n\nConversation context:\n{conversation_context}"
            
            user_prompt += "\n\nRewritten query:"
            
            rewritten = await self.llm_service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=100
            )
            
            return rewritten.strip()
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query  # Return original on error


class CitationService:
    """Service for generating citations from retrieved chunks"""
    
    def generate_citations(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        query_id: str
    ) -> List[Citation]:
        """Generate citations linking answer to source chunks"""
        citations = []
        
        try:
            # Simple approach: find text overlap between answer and chunks
            answer_lower = answer.lower()
            
            for position, result in enumerate(retrieved_chunks):
                chunk = result['chunk']
                chunk_content = chunk.content.lower()
                
                # Find common n-grams (sentences/phrases)
                common_phrases = self._find_common_phrases(answer_lower, chunk_content)
                
                if common_phrases:
                    # Create citation for the best matching phrase
                    best_phrase = max(common_phrases, key=len)
                    
                    citation = Citation(
                        query_id=query_id,
                        chunk=chunk,
                        quote=best_phrase,
                        relevance_score=result['score'],
                        position_in_answer=position
                    )
                    citations.append(citation)
        except Exception as e:
            logger.error(f"Citation generation failed: {e}")
        
        return citations
    
    def _find_common_phrases(
        self,
        text1: str,
        text2: str,
        min_length: int = 20
    ) -> List[str]:
        """Find common phrases between two texts"""
        common_phrases = []
        
        try:
            # Split into sentences
            sentences1 = text1.split('.')
            sentences2 = text2.split('.')
            
            for sent1 in sentences1:
                sent1 = sent1.strip()
                if len(sent1) < min_length:
                    continue
                
                for sent2 in sentences2:
                    sent2 = sent2.strip()
                    if len(sent2) < min_length:
                        continue
                    
                    # Check for substring match
                    if sent1 in sent2 or sent2 in sent1:
                        common_phrases.append(sent1 if len(sent1) < len(sent2) else sent2)
        except Exception as e:
            logger.error(f"Phrase finding failed: {e}")
        
        return common_phrases


class ConversationMemoryService:
    """Service for managing conversation memory"""
    
    def __init__(self):
        self.max_context_length = 4000  # tokens
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Tokenizer init failed: {e}")
            self.tokenizer = None
    
    def get_context(self, conversation_id: str) -> str:
        """Get conversation context formatted for LLM"""
        try:
            conversation = Conversation.objects.get(id=conversation_id)
            
            # Get recent queries
            recent_queries = Query.objects.filter(
                conversation=conversation,
                deleted=False
            ).order_by('-created_at')[:5]
            
            context_parts = []
            total_tokens = 0
            
            for query in reversed(list(recent_queries)):
                query_text = f"User: {query.query_text}\nAssistant: {query.answer}"
                
                if self.tokenizer:
                    tokens = len(self.tokenizer.encode(query_text))
                else:
                    tokens = len(query_text.split())  # Rough estimate
                
                if total_tokens + tokens > self.max_context_length:
                    break
                
                context_parts.append(query_text)
                total_tokens += tokens
            
            return "\n\n".join(context_parts)
        
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return ""
    
    def update_context(
        self,
        conversation_id: str,
        query: str,
        answer: str
    ):
        """Update conversation context"""
        try:
            conversation = Conversation.objects.get(id=conversation_id)
            conversation.message_count = F('message_count') + 1
            conversation.save(update_fields=['message_count', 'last_activity'])
        except Exception as e:
            logger.error(f"Context update failed: {e}")


class AdvancedRAGService:
    """
    Advanced RAG service combining all components - FIXED VERSION
    """
    
    def __init__(
        self,
        llm_service,
        embedding_service: EmbeddingService,
        hybrid_search: HybridSearchService,
        reranking_service: RerankingService,
        query_rewriting: QueryRewritingService,
        citation_service: CitationService,
        conversation_memory: ConversationMemoryService
    ):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.hybrid_search = hybrid_search
        self.reranking_service = reranking_service
        self.query_rewriting = query_rewriting
        self.citation_service = citation_service
        self.conversation_memory = conversation_memory
    
    async def execute_query(
        self,
        query: str,
        organization_id: str = None,  # Now optional/ignored
        user_id: int = None,  # Use user_id instead
        conversation_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        strategy: str = 'hybrid',
        top_k: int = 5,
        hybrid_alpha: float = 0.5,
        use_reranking: bool = True,
        use_query_rewriting: bool = True,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Execute RAG query with all advanced features - FIXED VERSION
        """
        start_time = time.time()
        metrics = {
            'retrieval_time': 0,
            'rerank_time': 0,
            'generation_time': 0
        }
        
        logger.info(f"🔍 Executing RAG query: '{query[:50]}...'")
        
        # Step 1: Query rewriting (optional)
        original_query = query
        if use_query_rewriting and conversation_id:
            try:
                context = self.conversation_memory.get_context(conversation_id)
                query = await self.query_rewriting.rewrite_query(query, context)
                logger.info(f"✏️  Query rewritten: '{original_query}' -> '{query}'")
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}")
                query = original_query
        
        # Step 2: Retrieval (hybrid search) - FIXED: Use user_id
        retrieval_start = time.time()
        results = self.hybrid_search.search(
            query=query,
            user_id=user_id,  # FIXED: Use user_id instead of organization_id
            document_ids=document_ids,
            top_k=top_k * 2 if use_reranking else top_k,
            alpha=hybrid_alpha
        )
        metrics['retrieval_time'] = time.time() - retrieval_start
        
        if not results:
            logger.warning("No results found")
            return self._generate_fallback_response(
                query=original_query,
                metrics=metrics,
                start_time=start_time
            )
        
        logger.info(f"📚 Retrieved {len(results)} chunks")
        
        # Step 3: Re-ranking (optional)
        if use_reranking:
            rerank_start = time.time()
            try:
                results = self.reranking_service.rerank(query, results, top_k)
                metrics['rerank_time'] = time.time() - rerank_start
                logger.info(f"📊 Re-ranked to top {len(results)} results")
            except Exception as e:
                logger.warning(f"Re-ranking failed: {e}")
                results = results[:top_k]
        
        # Step 4: Generate answer
        generation_start = time.time()
        answer, confidence = await self._generate_answer(
            query=query,
            results=results,
            conversation_id=conversation_id
        )
        metrics['generation_time'] = time.time() - generation_start
        
        logger.info(f"💬 Answer generated (confidence: {confidence:.2f})")
        
        # Step 5: Save query and generate citations
        query_id = None
        citations = []
        
        if include_citations:
            try:
                from django.db import transaction
                with transaction.atomic():
                    query_obj = Query.objects.create(
                        query_text=original_query,
                        rewritten_query=query if use_query_rewriting else "",
                        answer=answer,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        strategy_used=strategy,
                        confidence_score=confidence,
                        processing_time=time.time() - start_time,
                        retrieval_time=metrics['retrieval_time'],
                        rerank_time=metrics['rerank_time'],
                        generation_time=metrics['generation_time'],
                        used_reranking=use_reranking,
                        used_query_rewriting=use_query_rewriting,
                        hybrid_search_alpha=hybrid_alpha if strategy == 'hybrid' else None
                    )
                    
                    query_id = str(query_obj.id)
                    
                    # Generate and save citations
                    citations = self.citation_service.generate_citations(
                        answer=answer,
                        retrieved_chunks=results,
                        query_id=query_id
                    )
                    if citations:
                        Citation.objects.bulk_create(citations)
                    
                    # Link source documents
                    source_doc_ids = list(set(r['chunk'].document_id for r in results))
                    query_obj.source_documents.set(source_doc_ids)
            except Exception as e:
                logger.error(f"Failed to save query/citations: {e}")
        
        # Step 6: Update conversation memory
        if conversation_id:
            try:
                self.conversation_memory.update_context(
                    conversation_id=conversation_id,
                    query=original_query,
                    answer=answer
                )
            except Exception as e:
                logger.error(f"Failed to update conversation: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"✅ Query completed in {total_time:.2f}s")
        
        return {
            'query_id': query_id,
            'query_text': original_query,
            'rewritten_query': query if use_query_rewriting else "",
            'answer': answer,
            'confidence_score': confidence,
            'strategy_used': strategy,
            'processing_time': total_time,
            'retrieval_time': metrics['retrieval_time'],
            'rerank_time': metrics['rerank_time'],
            'generation_time': metrics['generation_time'],
            'retrieved_chunks': self._format_chunks(results),
            'citations': [self._format_citation(c) for c in citations],
            'used_reranking': use_reranking,
            'used_query_rewriting': use_query_rewriting
        }
    
    async def _generate_answer(
        self,
        query: str,
        results: List[Dict[str, Any]],
        conversation_id: Optional[str] = None
    ) -> Tuple[str, float]:
        """Generate answer from retrieved chunks"""
        try:
            # Build context from chunks
            context_parts = []
            for i, result in enumerate(results, 1):
                chunk = result['chunk']
                context_parts.append(
                    f"[Source {i}] {chunk.document.filename}\n{chunk.content}\n"
                )
            
            context = "\n".join(context_parts)
            
            # Get conversation context
            conversation_context = ""
            if conversation_id:
                conversation_context = self.conversation_memory.get_context(conversation_id)
            
            # Build prompt
            system_prompt = """You are an AI assistant that answers questions based on provided context.

Guidelines:
1. Answer ONLY using information from the provided sources
2. Cite sources using [Source N] format
3. If information is not in sources, say so
4. Be concise but complete
5. Maintain conversation context when relevant"""
            
            user_prompt = f"""Context from documents:
{context}

"""
            
            if conversation_context:
                user_prompt += f"""Previous conversation:
{conversation_context}

"""
            
            user_prompt += f"""Question: {query}

Answer:"""
            
            # Try async generation first
            try:
                answer = await self.llm_service.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=500
                )
            except (TypeError, AttributeError):
                # Fallback to sync if async not supported
                if hasattr(self.llm_service, 'generate_with_context'):
                    result = self.llm_service.generate_with_context(
                        query=query,
                        context_chunks=[context],
                        system_prompt=system_prompt,
                        temperature=0.3
                    )
                    answer = result['response']
                else:
                    raise
            
            # Calculate confidence
            avg_score = np.mean([r['score'] for r in results])
            confidence = float(min(avg_score * 1.2, 1.0))
            
            return answer, confidence
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I encountered an error generating the answer.", 0.0
    
    def _generate_fallback_response(
        self,
        query: str,
        metrics: Dict[str, float],
        start_time: float
    ) -> Dict[str, Any]:
        """Generate fallback response when no chunks found"""
        return {
            'query_id': None,
            'query_text': query,
            'rewritten_query': "",
            'answer': "I couldn't find relevant information in the uploaded documents to answer your question.",
            'confidence_score': 0.0,
            'strategy_used': 'fallback',
            'processing_time': time.time() - start_time,
            'retrieval_time': metrics['retrieval_time'],
            'rerank_time': 0.0,
            'generation_time': 0.0,
            'retrieved_chunks': [],
            'citations': [],
            'used_reranking': False,
            'used_query_rewriting': False
        }
    
    def _format_chunks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format chunks for response"""
        return [
            {
                'chunk_id': str(result['chunk'].id),
                'content': result['chunk'].content[:200] + "..." if len(result['chunk'].content) > 200 else result['chunk'].content,
                'document_id': str(result['chunk'].document_id),
                'document_name': result['chunk'].document.filename,
                'score': result['score'],
                'metadata': result['chunk'].metadata
            }
            for result in results
        ]
    
    def _format_citation(self, citation: Citation) -> Dict[str, Any]:
        """Format citation for response"""
        return {
            'citation_id': str(citation.id),
            'chunk_id': str(citation.chunk_id),
            'document_name': citation.chunk.document.filename,
            'quote': citation.quote,
            'relevance_score': citation.relevance_score,
            'position': citation.position_in_answer
        }