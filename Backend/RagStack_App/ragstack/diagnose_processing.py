# apps/ragstack/diagnose_processing.py
"""
Diagnose document processing issues
"""
import os
import sys
import django

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from RagStack_App.ragstack.models import Document, DocumentChunk
from django.core.files.storage import default_storage

def diagnose():
    print("="*70)
    print("  DOCUMENT PROCESSING DIAGNOSIS")
    print("="*70)
    
    # Get document with 0 chunks
    doc = Document.objects.filter(chunks_count=0).first()
    
    if not doc:
        print("✅ No documents with 0 chunks found!")
        return
    
    print(f"\n📄 Analyzing: {doc.filename}")
    print(f"   ID: {doc.id}")
    print(f"   Status: {doc.status}")
    print(f"   File path: {doc.file_path}")
    print(f"   File size: {doc.file_size} bytes")
    print(f"   Content type: {doc.content_type}")
    
    # Check if file exists
    if default_storage.exists(doc.file_path):
        print(f"   ✅ File exists on disk")
        
        # Try to read it
        try:
            with default_storage.open(doc.file_path, 'rb') as f:
                content = f.read()
                print(f"   ✅ File readable: {len(content)} bytes")
        except Exception as e:
            print(f"   ❌ Cannot read file: {e}")
            return
    else:
        print(f"   ❌ File NOT found on disk")
        return
    
    # Try to process it NOW (synchronously)
    print(f"\n🔄 Attempting to process document...")
    
    try:
        from RagStack_App.ragstack.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        
        # Process synchronously
        processor.process_document(str(doc.id))
        
        # Check results
        doc.refresh_from_db()
        print(f"\n📊 Results:")
        print(f"   Status: {doc.status}")
        print(f"   Chunks created: {doc.chunks_count}")
        
        if doc.chunks_count > 0:
            print(f"   ✅ SUCCESS! Document processed correctly")
            
            # Show sample chunks
            chunks = DocumentChunk.objects.filter(document=doc)[:3]
            print(f"\n   Sample chunks:")
            for chunk in chunks:
                print(f"   - Chunk {chunk.chunk_index}: {chunk.content[:100]}...")
        else:
            print(f"   ❌ FAILED: No chunks created")
            print(f"   Metadata: {doc.metadata}")
    
    except Exception as e:
        print(f"   ❌ Processing error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose()