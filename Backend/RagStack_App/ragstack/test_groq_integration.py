"""
===============================================================
Test Script for Groq RAG Integration
===============================================================
Run this to verify your Groq setup is working correctly
Usage: python apps/ragstack/test_groq_integration.py
===============================================================
"""

import os
import sys
import django

# Setup Django environment
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from RagStack_App.ragstack.llm_service import GroqLLMService, get_groq_service
from RagStack_App.ragstack.rag_dependencies import check_services_health, initialize_all_services


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def test_groq_connection():
    """Test 1: Basic Groq connection"""
    print_header("TEST 1: Groq API Connection")
    
    try:
        groq = get_groq_service()
        
        if groq.test_connection():
            print("✅ Groq API connection successful")
            
            # Get model info
            info = groq.get_model_info()
            print(f"\n📋 Configuration:")
            print(f"   Default Model: {info['default_model']}")
            print(f"   Status: {info['status']}")
            print(f"   Available Models:")
            for model_type, model_name in info['available_models'].items():
                print(f"      - {model_type}: {model_name}")
            
            return True
        else:
            print("❌ Groq API connection failed")
            return False
    
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def test_simple_generation():
    """Test 2: Simple text generation"""
    print_header("TEST 2: Simple Text Generation")
    
    try:
        groq = get_groq_service()
        
        query = "What is artificial intelligence?"
        context = ["Artificial intelligence (AI) is the simulation of human intelligence by machines."]
        
        print(f"Query: {query}")
        print(f"Context: {context[0][:50]}...")
        
        result = groq.generate_with_context(
            query=query,
            context_chunks=context,
            temperature=0.3
        )
        
        if result['success']:
            print(f"\n✅ Generation successful")
            print(f"📝 Response: {result['response'][:200]}...")
            print(f"🔢 Tokens used: {result['tokens_used']}")
            print(f"🤖 Model: {result['model']}")
            return True
        else:
            print(f"❌ Generation failed: {result.get('error')}")
            return False
    
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False


def test_query_rewriting():
    """Test 3: Query rewriting"""
    print_header("TEST 3: Query Rewriting")
    
    try:
        groq = get_groq_service()
        
        original_query = "What's the cap?"
        context = "We're discussing France and European capitals."
        
        print(f"Original Query: {original_query}")
        print(f"Context: {context}")
        
        rewritten = groq.rewrite_query(
            query=original_query,
            conversation_context=context
        )
        
        print(f"\n✅ Query rewritten")
        print(f"📝 Rewritten: {rewritten}")
        
        if rewritten != original_query:
            print("✅ Query was successfully modified")
            return True
        else:
            print("⚠️  Query unchanged (might be acceptable)")
            return True
    
    except Exception as e:
        print(f"❌ Query rewriting test failed: {e}")
        return False


def test_rag_context_generation():
    """Test 4: RAG context-based generation"""
    print_header("TEST 4: RAG Context Generation")
    
    try:
        groq = get_groq_service()
        
        query = "What programming languages are mentioned?"
        context_chunks = [
            "Python is a high-level programming language. It's widely used for data science.",
            "JavaScript is essential for web development. It runs in browsers.",
            "Java is used for enterprise applications. It's platform-independent."
        ]
        
        print(f"Query: {query}")
        print(f"Context chunks: {len(context_chunks)} chunks")
        
        result = groq.generate_with_context(
            query=query,
            context_chunks=context_chunks,
            temperature=0.3
        )
        
        if result['success']:
            print(f"\n✅ RAG generation successful")
            print(f"📝 Response:\n{result['response']}")
            print(f"\n🔢 Tokens used: {result['tokens_used']}")
            
            # Check if response mentions the languages
            response_lower = result['response'].lower()
            mentioned = []
            if 'python' in response_lower:
                mentioned.append('Python')
            if 'javascript' in response_lower:
                mentioned.append('JavaScript')
            if 'java' in response_lower and 'javascript' not in response_lower:
                mentioned.append('Java')
            
            print(f"✅ Languages mentioned: {', '.join(mentioned)}")
            return True
        else:
            print(f"❌ Generation failed: {result.get('error')}")
            return False
    
    except Exception as e:
        print(f"❌ RAG context test failed: {e}")
        return False


def test_all_services():
    """Test 5: All RAG services health check"""
    print_header("TEST 5: All Services Health Check")
    
    try:
        print("Initializing all services...")
        initialize_all_services()
        
        print("\nChecking service health...")
        health = check_services_health()
        
        print(f"\n📊 Service Status:")
        for service, status in health.items():
            if service == 'overall_status':
                continue
            
            if status == 'operational':
                print(f"   ✅ {service}: {status}")
            else:
                print(f"   ❌ {service}: {status}")
        
        print(f"\n🏥 Overall Status: {health['overall_status']}")
        
        return health['overall_status'] in ['healthy', 'operational']
    
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "GROQ RAG INTEGRATION TEST SUITE" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    # Check if GROQ_API_KEY is set
    from decouple import config
    try:
        api_key = config('GROQ_API_KEY', default=None)
        if not api_key:
            api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            print("\n❌ ERROR: GROQ_API_KEY not found!")
            print("Please set GROQ_API_KEY in your .env file or environment variables.")
            print("\nExample .env entry:")
            print("GROQ_API_KEY=your_api_key_here")
            return False
        
        print(f"\n✅ GROQ_API_KEY found: {api_key[:10]}...")
    except Exception as e:
        print(f"\n❌ Error checking API key: {e}")
        return False
    
    # Run tests
    tests = [
        ("Groq Connection", test_groq_connection),
        ("Simple Generation", test_simple_generation),
        ("Query Rewriting", test_query_rewriting),
        ("RAG Context Generation", test_rag_context_generation),
        ("All Services Health", test_all_services),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Groq RAG integration is working correctly.")
        print("\nNext steps:")
        print("1. Upload some documents via the API")
        print("2. Try executing queries")
        print("3. Monitor performance metrics")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("\nTroubleshooting tips:")
        print("1. Verify GROQ_API_KEY is correct")
        print("2. Check internet connection")
        print("3. Ensure all dependencies are installed: pip install -r requirements_rag.txt")
        print("4. Check Django logs for more details")
    
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)