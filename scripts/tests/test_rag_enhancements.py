"""
Unit Tests for RAG Enhancements
Tests for embedding service, chunking, and normalization.
"""

import pytest
import numpy as np
from scriptguard.rag.embedding_service import EmbeddingService
from scriptguard.rag.chunking_service import ChunkingService, ResultAggregator


class TestEmbeddingService:
    """Test embedding service with different pooling strategies."""

    def test_normalization_single_embedding(self):
        """Test L2 normalization on single embedding."""
        service = EmbeddingService(
            model_name="microsoft/unixcoder-base",
            pooling_strategy="mean_pooling",
            normalize=True,
            max_length=128
        )

        code = "import os\nos.system('ls')"
        embedding = service.encode_single(code)

        # Verify L2 norm is approximately 1
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"Expected L2 norm ≈ 1.0, got {norm}"
        print(f"✓ Single embedding L2 norm: {norm:.6f}")

    def test_normalization_batch_embeddings(self):
        """Test L2 normalization on batch of embeddings."""
        service = EmbeddingService(
            model_name="microsoft/unixcoder-base",
            pooling_strategy="mean_pooling",
            normalize=True,
            max_length=128
        )

        codes = [
            "import os\nos.system('ls')",
            "import pandas as pd\ndf = pd.read_csv('data.csv')",
            "def hello():\n    print('Hello, world!')",
            "for i in range(10):\n    print(i)"
        ]

        embeddings = service.encode(codes)

        # Verify all embeddings have L2 norm ≈ 1
        stats = service.verify_normalization(embeddings)

        assert abs(stats["mean_norm"] - 1.0) < 0.01, f"Mean norm should be ≈ 1.0, got {stats['mean_norm']}"
        assert stats["min_norm"] > 0.95, f"Min norm should be > 0.95, got {stats['min_norm']}"
        assert stats["max_norm"] < 1.05, f"Max norm should be < 1.05, got {stats['max_norm']}"

        print(f"✓ Batch embeddings normalization stats:")
        print(f"  Mean: {stats['mean_norm']:.6f}")
        print(f"  Min:  {stats['min_norm']:.6f}")
        print(f"  Max:  {stats['max_norm']:.6f}")
        print(f"  Std:  {stats['std_norm']:.6f}")

    def test_pooling_strategies(self):
        """Test different pooling strategies produce valid embeddings."""
        strategies = ["cls", "mean_pooling"]
        code = "import os\nos.system('ls')"

        for strategy in strategies:
            service = EmbeddingService(
                model_name="microsoft/unixcoder-base",
                pooling_strategy=strategy,
                normalize=True,
                max_length=128
            )

            embedding = service.encode_single(code)
            norm = np.linalg.norm(embedding)

            assert embedding.shape[0] > 0, f"Strategy {strategy}: embedding dimension should be > 0"
            assert abs(norm - 1.0) < 0.01, f"Strategy {strategy}: L2 norm should be ≈ 1.0, got {norm}"

            print(f"✓ Strategy '{strategy}': dim={embedding.shape[0]}, norm={norm:.6f}")

    def test_without_normalization(self):
        """Test embeddings without normalization have varying norms."""
        service = EmbeddingService(
            model_name="microsoft/unixcoder-base",
            pooling_strategy="mean_pooling",
            normalize=False,
            max_length=128
        )

        codes = [
            "import os",
            "import os\nimport sys\nimport json\nimport pandas as pd"
        ]

        embeddings = service.encode(codes)
        stats = service.verify_normalization(embeddings)

        # Without normalization, norms should vary significantly
        assert stats["max_norm"] - stats["min_norm"] > 0.1, "Norms should vary without normalization"

        print(f"✓ Without normalization - norm range: {stats['min_norm']:.3f} to {stats['max_norm']:.3f}")


class TestChunkingService:
    """Test chunking service for long code files."""

    def test_no_chunking_for_short_code(self):
        """Test that short code doesn't get chunked."""
        chunker = ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            chunk_size=512,
            overlap=64
        )

        short_code = "def hello():\n    print('Hello, world!')"
        chunks = chunker.chunk_code(
            code=short_code,
            db_id=1,
            label="benign",
            source="test"
        )

        assert len(chunks) == 1, "Short code should not be chunked"
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["total_chunks"] == 1

        print(f"✓ Short code: 1 chunk (no splitting)")

    def test_chunking_for_long_code(self):
        """Test that long code gets chunked with overlap."""
        chunker = ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            chunk_size=128,
            overlap=16
        )

        # Generate long code
        long_code = "\n".join([f"def function_{i}():\n    return {i}" for i in range(100)])

        chunks = chunker.chunk_code(
            code=long_code,
            db_id=1,
            label="benign",
            source="test"
        )

        assert len(chunks) > 1, "Long code should be chunked"

        # Verify chunk metadata
        for i, chunk in enumerate(chunks):
            assert chunk["db_id"] == 1
            assert chunk["chunk_index"] == i
            assert chunk["total_chunks"] == len(chunks)
            assert chunk["label"] == "benign"
            assert len(chunk["content"]) > 0

        print(f"✓ Long code: {len(chunks)} chunks")
        print(f"  First chunk length: {len(chunks[0]['content'])} chars")
        print(f"  Last chunk length: {len(chunks[-1]['content'])} chars")

    def test_chunk_samples_batch(self):
        """Test chunking multiple samples."""
        chunker = ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            chunk_size=128,
            overlap=16
        )

        samples = [
            {"id": 1, "content": "def short():\n    pass", "label": "benign", "source": "test"},
            {"id": 2, "content": "\n".join([f"line {i}" for i in range(200)]), "label": "malicious", "source": "test"}
        ]

        chunks = chunker.chunk_samples(samples)

        # Should have more chunks than samples due to long code
        assert len(chunks) > len(samples), "Should have more chunks than samples"

        # Verify db_id mapping
        db_ids = set(c["db_id"] for c in chunks)
        assert db_ids == {1, 2}, "Should have chunks from both samples"

        print(f"✓ Batch chunking: {len(samples)} samples → {len(chunks)} chunks")


class TestResultAggregator:
    """Test chunk aggregation strategies."""

    def test_max_score_aggregation(self):
        """Test max_score aggregation strategy."""
        results = [
            {"db_id": 1, "score": 0.9, "chunk_index": 0, "content": "chunk 0", "label": "malicious"},
            {"db_id": 1, "score": 0.7, "chunk_index": 1, "content": "chunk 1", "label": "malicious"},
            {"db_id": 2, "score": 0.8, "chunk_index": 0, "content": "chunk 0", "label": "benign"},
            {"db_id": 2, "score": 0.6, "chunk_index": 1, "content": "chunk 1", "label": "benign"},
        ]

        aggregated = ResultAggregator.aggregate_results(results, strategy="max_score")

        assert len(aggregated) == 2, "Should aggregate to 2 documents"

        # Verify scores are the max from each document
        doc1 = next(d for d in aggregated if d["db_id"] == 1)
        doc2 = next(d for d in aggregated if d["db_id"] == 2)

        assert doc1["score"] == 0.9, "Doc 1 should have max score 0.9"
        assert doc2["score"] == 0.8, "Doc 2 should have max score 0.8"
        assert doc1["num_chunks"] == 2
        assert doc2["num_chunks"] == 2

        # Verify ordering (highest score first)
        assert aggregated[0]["db_id"] == 1, "Highest scoring doc should be first"

        print(f"✓ Max score aggregation: {len(results)} chunks → {len(aggregated)} docs")

    def test_average_top_n_aggregation(self):
        """Test average_top_n aggregation strategy."""
        results = [
            {"db_id": 1, "score": 0.9, "chunk_index": 0, "content": "chunk 0", "label": "malicious"},
            {"db_id": 1, "score": 0.8, "chunk_index": 1, "content": "chunk 1", "label": "malicious"},
            {"db_id": 1, "score": 0.5, "chunk_index": 2, "content": "chunk 2", "label": "malicious"},
        ]

        aggregated = ResultAggregator.aggregate_results(results, strategy="average_top_n", top_n=2)

        assert len(aggregated) == 1

        # Average of top 2: (0.9 + 0.8) / 2 = 0.85
        expected_score = (0.9 + 0.8) / 2
        assert abs(aggregated[0]["score"] - expected_score) < 0.01, f"Expected score {expected_score}, got {aggregated[0]['score']}"

        print(f"✓ Average top-N aggregation: top-2 average = {aggregated[0]['score']:.3f}")

    def test_weighted_avg_aggregation(self):
        """Test weighted average aggregation strategy."""
        results = [
            {"db_id": 1, "score": 0.9, "chunk_index": 0, "content": "chunk 0", "label": "malicious"},
            {"db_id": 1, "score": 0.6, "chunk_index": 1, "content": "chunk 1", "label": "malicious"},
        ]

        aggregated = ResultAggregator.aggregate_results(results, strategy="weighted_avg")

        assert len(aggregated) == 1

        # Weighted avg: (0.9^2 + 0.6^2) / (0.9 + 0.6)
        total_weight = 0.9 + 0.6
        weighted_score = (0.9**2 + 0.6**2) / total_weight

        assert abs(aggregated[0]["score"] - weighted_score) < 0.01, f"Expected {weighted_score}, got {aggregated[0]['score']}"

        print(f"✓ Weighted average aggregation: score = {aggregated[0]['score']:.3f}")


def test_end_to_end_long_file():
    """
    End-to-end test: Long file with critical logic at the end.
    Verifies that chunking + aggregation can find relevant content anywhere in file.
    """
    # Create a long file with critical malicious code at the end
    benign_code = "\n".join([
        "import pandas as pd",
        "import numpy as np",
        "",
        "def process_data(df):",
        "    df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()",
        "    return df",
        "",
        "def analyze(data):",
        "    results = []",
        "    for row in data:",
        "        results.append(row * 2)",
        "    return results",
        ""
    ] * 20)  # Repeat to make it long

    malicious_code = """
# Critical malicious payload hidden at end
import os
import base64
payload = base64.b64decode('cm0gLXJmIC8=')  # 'rm -rf /'
os.system(payload.decode())
"""

    full_code = benign_code + malicious_code

    # Chunk the code
    chunker = ChunkingService(
        tokenizer_name="microsoft/unixcoder-base",
        chunk_size=256,
        overlap=32
    )

    chunks = chunker.chunk_code(
        code=full_code,
        db_id=1,
        label="malicious",
        source="test"
    )

    print(f"\n✓ End-to-end test: File chunked into {len(chunks)} chunks")

    # Verify malicious code is in at least one chunk
    malicious_found = any("base64.b64decode" in chunk["content"] for chunk in chunks)
    assert malicious_found, "Malicious code should be captured in at least one chunk"

    print(f"✓ Malicious payload found in chunks")

    # Simulate search results (last chunk has highest score due to malicious content)
    mock_results = []
    for i, chunk in enumerate(chunks):
        if "base64.b64decode" in chunk["content"]:
            score = 0.95  # High score for malicious chunk
        else:
            score = 0.3 + (i / len(chunks)) * 0.2  # Lower scores for benign chunks

        mock_results.append({
            "db_id": 1,
            "score": score,
            "chunk_index": i,
            "content": chunk["content"][:100],
            "label": "malicious"
        })

    # Aggregate using max_score
    aggregated = ResultAggregator.aggregate_results(mock_results, strategy="max_score")

    assert len(aggregated) == 1
    assert aggregated[0]["score"] == 0.95, "Should pick the highest scoring chunk"

    print(f"✓ Aggregation correctly identifies document with score: {aggregated[0]['score']:.3f}")
    print(f"✓ System can find critical logic at end of long file")


if __name__ == "__main__":
    print("Running RAG Enhancement Tests...\n")
    print("=" * 70)

    # Test Embedding Service
    print("\n[1] Testing Embedding Service")
    print("-" * 70)
    test_embed = TestEmbeddingService()
    test_embed.test_normalization_single_embedding()
    test_embed.test_normalization_batch_embeddings()
    test_embed.test_pooling_strategies()
    test_embed.test_without_normalization()

    # Test Chunking Service
    print("\n[2] Testing Chunking Service")
    print("-" * 70)
    test_chunk = TestChunkingService()
    test_chunk.test_no_chunking_for_short_code()
    test_chunk.test_chunking_for_long_code()
    test_chunk.test_chunk_samples_batch()

    # Test Result Aggregator
    print("\n[3] Testing Result Aggregator")
    print("-" * 70)
    test_agg = TestResultAggregator()
    test_agg.test_max_score_aggregation()
    test_agg.test_average_top_n_aggregation()
    test_agg.test_weighted_avg_aggregation()

    # End-to-end test
    print("\n[4] End-to-End Test: Long File with Critical Logic at End")
    print("-" * 70)
    test_end_to_end_long_file()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
