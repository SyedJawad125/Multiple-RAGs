# apps/ragstack/cleanup_failed_docs.py
"""
Clean up documents with missing files
"""
import os
import sys
import django

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.ragstack.models import Document
from django.core.files.storage import default_storage

def cleanup():
    print("="*70)
    print("  CLEANING UP DOCUMENTS WITH MISSING FILES")
    print("="*70)
    
    documents = Document.objects.filter(deleted=False)
    deleted_count = 0
    
    for doc in documents:
        if not default_storage.exists(doc.file_path):
            print(f"❌ Missing file: {doc.filename} (ID: {doc.id})")
            print(f"   Deleting from database...")
            doc.delete()
            deleted_count += 1
    
    if deleted_count == 0:
        print("\n✅ No cleanup needed - all files exist!")
    else:
        print(f"\n✅ Cleaned up {deleted_count} document(s)")
    
    # Show stats
    remaining = Document.objects.filter(deleted=False).count()
    print(f"\n📊 Remaining documents: {remaining}")

if __name__ == "__main__":
    cleanup()