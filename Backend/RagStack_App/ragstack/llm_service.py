"""
===============================================================
LLM Service - Groq API Integration
===============================================================
Handles all LLM operations using Groq API
Place this in: apps/ragstack/llm_service.py
===============================================================
"""

import logging
import os
from typing import Dict, List, Optional
from groq import Groq
from decouple import config

logger = logging.getLogger(__name__)


class GroqLLMService:
    """
    Production-ready Groq LLM Service for RAG system
    """
    
    def __init__(self):
        """Initialize Groq client"""
        try:
            # Try to get API key from multiple sources
            api_key = config('GROQ_API_KEY', default=None)
            if not api_key:
                api_key = os.getenv('GROQ_API_KEY')
            
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not found. Please set it in .env file or environment variables."
                )
            
            self.client = Groq(api_key=api_key)
            
            # Model configuration - using best available models
            self.models = {
                'fast': 'llama-3.1-8b-instant',      # Fast responses
                'balanced': 'llama-3.3-70b-versatile', # Balanced speed/quality
                'quality': 'llama-3.3-70b-versatile'   # Best quality
            }
            
            self.default_model = self.models['balanced']
            
            logger.info(f"✅ Groq LLM Service initialized with model: {self.default_model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq service: {e}")
            raise
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model_type: str = 'balanced'
    ) -> str:
        """
        Generate response using Groq API
        
        Args:
            system_prompt: System instructions for the model
            user_prompt: User query/prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            model_type: 'fast', 'balanced', or 'quality'
        
        Returns:
            Generated text response
        """
        try:
            model = self.models.get(model_type, self.default_model)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            logger.info(f"Generating with model: {model}")
            
            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract response
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            logger.info(f"✅ Generated response ({tokens_used} tokens)")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Groq generation failed: {str(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def generate_with_context(
        self,
        query: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Dict:
        """
        Generate response with RAG context (synchronous version)
        
        Args:
            query: User question
            context_chunks: Retrieved document chunks
            system_prompt: Optional custom system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            Dict with response, tokens_used, and model info
        """
        try:
            if not system_prompt:
                system_prompt = """You are an intelligent assistant that answers questions based on provided context.

Guidelines:
1. Answer ONLY using information from the provided sources
2. Cite sources using [Source N] format when referencing specific information
3. If the answer is not in the sources, clearly state "I don't have this information in the provided documents"
4. Be concise but complete
5. Maintain accuracy and avoid making assumptions"""
            
            # Build context string
            context_str = "\n\n".join([
                f"[Source {i+1}]:\n{chunk}" 
                for i, chunk in enumerate(context_chunks) if chunk
            ])
            
            user_prompt = f"""Context from documents:
{context_str}

Question: {query}

Answer:"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return {
                "success": True,
                "response": answer,
                "tokens_used": tokens_used,
                "model": self.default_model
            }
            
        except Exception as e:
            logger.error(f"❌ Context generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "Error processing your request. Please try again.",
                "tokens_used": 0
            }
    
    def rewrite_query(
        self,
        query: str,
        conversation_context: Optional[str] = None
    ) -> str:
        """
        Rewrite query for better retrieval (synchronous)
        
        Args:
            query: Original user query
            conversation_context: Optional conversation history
        
        Returns:
            Rewritten query
        """
        try:
            system_prompt = """You are a query optimization expert. 
Your task is to rewrite user queries to improve information retrieval.

Guidelines:
1. Expand abbreviations and acronyms
2. Add relevant context from conversation history
3. Make implicit intent explicit
4. Keep technical terms intact
5. Make queries more specific and searchable
6. Return ONLY the rewritten query, no explanation or additional text"""
            
            user_prompt = f"Original query: {query}"
            
            if conversation_context:
                user_prompt += f"\n\nConversation context:\n{conversation_context}"
            
            user_prompt += "\n\nRewritten query:"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.models['fast'],  # Use fast model for query rewriting
                messages=messages,
                temperature=0.3,
                max_tokens=100,
            )
            
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            
            return rewritten
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {str(e)}")
            # Return original query if rewriting fails
            return query
    
    def test_connection(self) -> bool:
        """
        Test Groq API connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.client.chat.completions.create(
                model=self.models['fast'],
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10,
            )
            logger.info("✅ Groq API connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Groq API connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about available models"""
        return {
            'available_models': self.models,
            'default_model': self.default_model,
            'status': 'operational' if self.test_connection() else 'error'
        }


# Singleton instance
_groq_service: Optional[GroqLLMService] = None


def get_groq_service() -> GroqLLMService:
    """
    Get or create singleton Groq service instance
    
    Returns:
        GroqLLMService instance
    """
    global _groq_service
    
    if _groq_service is None:
        _groq_service = GroqLLMService()
        logger.info("Initialized Groq LLM service")
    
    return _groq_service


def reset_groq_service():
    """Reset Groq service (useful for testing)"""
    global _groq_service
    _groq_service = None
    logger.info("Groq service reset")


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test the Groq service"""
    print("="*60)
    print("Testing Groq LLM Service")
    print("="*60)
    
    try:
        # Initialize service
        groq = GroqLLMService()
        
        # Test 1: Connection test
        print("\n1. Testing connection...")
        if groq.test_connection():
            print("   ✅ Connection successful")
        else:
            print("   ❌ Connection failed")
        
        # Test 2: Model info
        print("\n2. Model information:")
        info = groq.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test 3: Simple generation
        print("\n3. Testing simple generation...")
        result = groq.generate_with_context(
            query="What is the capital of France?",
            context_chunks=["France is a country in Europe. Paris is its capital city."],
            temperature=0.3
        )
        
        if result['success']:
            print(f"   ✅ Response: {result['response']}")
            print(f"   Tokens used: {result['tokens_used']}")
        else:
            print(f"   ❌ Error: {result['error']}")
        
        # Test 4: Query rewriting
        print("\n4. Testing query rewriting...")
        original = "What's the cap?"
        rewritten = groq.rewrite_query(
            original,
            conversation_context="We're discussing France and its cities."
        )
        print(f"   Original: {original}")
        print(f"   Rewritten: {rewritten}")
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")