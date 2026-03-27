"""
===============================================================
Reprocess Existing Documents Script
===============================================================
Run this to process documents that have chunks_count=0
Usage: python apps/ragstack/reprocess_documents.py
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

from RagStack_App.ragstack.models import Document, DocumentChunk
from RagStack_App.ragstack.document_processor import DocumentProcessor


def reprocess_documents():
    """Reprocess documents with 0 chunks"""
    
    print("="*70)
    print("  REPROCESSING DOCUMENTS WITH ZERO CHUNKS")
    print("="*70)
    
    # Find documents with 0 chunks
    documents = Document.objects.filter(
        chunks_count=0,
        deleted=False,
        status='completed'  # Documents that thought they completed but have no chunks
    ).order_by('-created_at')
    
    if not documents.exists():
        print("\n✅ No documents need reprocessing!")
        return
    
    print(f"\n📋 Found {documents.count()} documents to reprocess:")
    for doc in documents:
        print(f"   - {doc.filename} (ID: {doc.id})")
    
    print("\n🚀 Starting reprocessing...\n")
    
    processor = DocumentProcessor()
    success_count = 0
    fail_count = 0
    
    for doc in documents:
        print(f"{'='*70}")
        print(f"Processing: {doc.filename}")
        print(f"Document ID: {doc.id}")
        print(f"File size: {doc.file_size} bytes")
        print(f"Content type: {doc.content_type}")
        print(f"{'='*70}")
        
        try:
            # Delete any existing chunks (just in case)
            existing_chunks = DocumentChunk.objects.filter(document=doc).count()
            if existing_chunks > 0:
                DocumentChunk.objects.filter(document=doc).delete()
                print(f"🗑️  Deleted {existing_chunks} existing chunks")
            
            # Process the document
            processor.process_document(str(doc.id))
            
            # Check if chunks were created
            doc.refresh_from_db()
            if doc.chunks_count > 0:
                print(f"✅ SUCCESS: Created {doc.chunks_count} chunks")
                success_count += 1
            else:
                print(f"⚠️  WARNING: Processing completed but no chunks created")
                fail_count += 1
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            fail_count += 1
        
        print()
    
    # Summary
    print("="*70)
    print("  REPROCESSING COMPLETE")
    print("="*70)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📊 Total: {success_count + fail_count}")
    print("="*70)
    
    # Show updated stats
    print("\n📈 Updated Statistics:")
    total_docs = Document.objects.filter(deleted=False).count()
    docs_with_chunks = Document.objects.filter(deleted=False, chunks_count__gt=0).count()
    total_chunks = DocumentChunk.objects.count()
    
    print(f"   Total Documents: {total_docs}")
    print(f"   Documents with Chunks: {docs_with_chunks}")
    print(f"   Total Chunks: {total_chunks}")
    print()


if __name__ == "__main__":
    reprocess_documents()