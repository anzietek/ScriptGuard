import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from scriptguard.rag.qdrant_store import QdrantStore

store = QdrantStore(host='localhost', port=6333, collection_name='malware_knowledge')
info = store.get_collection_info()
print(f"malware_knowledge: {info.get('points_count', 0)} points")
