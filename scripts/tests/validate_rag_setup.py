"""
Quick validation of RAG enhancements setup.
"""

import sys
import traceback

def test_imports():
    """Test if all modules import correctly."""
    print("Testing imports...")
    try:
        from scriptguard.rag import EmbeddingService, ChunkingService, ResultAggregator, CodeSimilarityStore
        print("✓ All RAG modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_embedding_service():
    """Test embedding service basic functionality."""
    print("\nTesting EmbeddingService...")
    try:
        from scriptguard.rag import EmbeddingService
        import numpy as np

        # Test with a simple model path that should work
        service = EmbeddingService(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Using sentence-transformer directly
            pooling_strategy="sentence_transformer",
            normalize=True,
            max_length=128
        )

        # Test encoding
        test_code = "import os\nos.system('ls')"
        embedding = service.encode_single(test_code)

        # Verify normalization
        norm = np.linalg.norm(embedding)
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  L2 norm: {norm:.6f}")

        if abs(norm - 1.0) < 0.01:
            print("✓ EmbeddingService works correctly (normalized)")
            return True
        else:
            print(f"✗ Normalization failed: expected norm ≈ 1.0, got {norm}")
            return False

    except Exception as e:
        print(f"✗ EmbeddingService test failed: {e}")
        traceback.print_exc()
        return False

def test_chunking_service():
    """Test chunking service."""
    print("\nTesting ChunkingService...")
    try:
        from scriptguard.rag import ChunkingService

        chunker = ChunkingService(
            tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=128,
            overlap=16
        )

        # Test short code (no chunking)
        short_code = "def hello():\n    print('Hello')"
        chunks = chunker.chunk_code(short_code, db_id=1, label="benign")

        if len(chunks) == 1:
            print(f"  Short code: {len(chunks)} chunk (correct)")
        else:
            print(f"  Short code: {len(chunks)} chunks (unexpected)")
            return False

        # Test long code (should chunk)
        long_code = "\n".join([f"def func_{i}(): return {i}" for i in range(50)])
        chunks = chunker.chunk_code(long_code, db_id=2, label="benign")

        if len(chunks) > 1:
            print(f"  Long code: {len(chunks)} chunks (correct)")
            print("✓ ChunkingService works correctly")
            return True
        else:
            print(f"  Long code: {len(chunks)} chunk (should be multiple)")
            return False

    except Exception as e:
        print(f"✗ ChunkingService test failed: {e}")
        traceback.print_exc()
        return False

def test_result_aggregator():
    """Test result aggregator."""
    print("\nTesting ResultAggregator...")
    try:
        from scriptguard.rag import ResultAggregator

        # Mock results
        results = [
            {"db_id": 1, "score": 0.9, "chunk_index": 0, "content": "chunk 0", "label": "malicious"},
            {"db_id": 1, "score": 0.7, "chunk_index": 1, "content": "chunk 1", "label": "malicious"},
            {"db_id": 2, "score": 0.8, "chunk_index": 0, "content": "chunk 0", "label": "benign"},
        ]

        # Test max_score aggregation
        aggregated = ResultAggregator.aggregate_results(results, strategy="max_score")

        if len(aggregated) == 2:
            print(f"  Aggregated {len(results)} chunks to {len(aggregated)} documents")
            print(f"  Doc 1 score: {aggregated[0]['score']:.3f}")
            print(f"  Doc 2 score: {aggregated[1]['score']:.3f}")
            print("✓ ResultAggregator works correctly")
            return True
        else:
            print(f"✗ Expected 2 documents, got {len(aggregated)}")
            return False

    except Exception as e:
        print(f"✗ ResultAggregator test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("=" * 70)
    print("RAG ENHANCEMENTS - VALIDATION SCRIPT")
    print("=" * 70)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("EmbeddingService", test_embedding_service()))
    results.append(("ChunkingService", test_chunking_service()))
    results.append(("ResultAggregator", test_result_aggregator()))

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(r[1] for r in results)

    print("=" * 70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("\nNext steps:")
        print("1. Run: python test_rag_enhancements.py (full unit tests)")
        print("2. Run: python test_rag_offline_eval.py (offline evaluation)")
        print("3. Re-vectorize data: python -m scriptguard.steps.vectorize_samples")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
