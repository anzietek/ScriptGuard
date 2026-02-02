"""
Code Similarity Store for Few-Shot RAG
Stores vectorized code samples from PostgreSQL for retrieval during inference.
Enhanced with unified embedding strategies, L2 normalization, and chunking support.
"""

import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import hashlib

from scriptguard.utils.logger import logger
from .embedding_service import EmbeddingService
from .chunking_service import ChunkingService, ResultAggregator


class CodeSimilarityStore:
    """
    Manages code sample embeddings in Qdrant for Few-Shot RAG.
    Uses dedicated code embedding models for better similarity search.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = "code_samples",
        embedding_model: str = "microsoft/unixcoder-base",
        pooling_strategy: str = "mean_pooling",
        normalize: bool = True,
        max_length: int = 512,
        enable_chunking: bool = True,
        chunk_overlap: int = 64,
        api_key: Optional[str] = None,
        use_https: bool = False
    ):
        """
        Initialize Code Similarity Store with enhanced embedding and chunking.

        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection (default: "code_samples")
            embedding_model: Code embedding model
                Options:
                - "microsoft/unixcoder-base" (recommended for code)
                - "Salesforce/codet5p-110m-embedding" (alternative)
            pooling_strategy: Pooling strategy ("cls", "mean_pooling", "pooler_output", "sentence_transformer")
            normalize: Apply L2 normalization to embeddings
            max_length: Maximum sequence length in tokens
            enable_chunking: Enable sliding window chunking for long code
            chunk_overlap: Overlap between chunks in tokens
            api_key: Optional API key for Qdrant Cloud
            use_https: Use HTTPS connection
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name
        self.enable_chunking = enable_chunking

        logger.info(f"Initializing Code Similarity Store: {self.host}:{self.port}")
        logger.info(f"  Chunking: {'enabled' if enable_chunking else 'disabled'}")

        # Initialize Qdrant client
        if api_key:
            self.client = QdrantClient(
                url=f"{'https' if use_https else 'http'}://{self.host}:{self.port}",
                api_key=api_key
            )
        else:
            self.client = QdrantClient(host=self.host, port=self.port)

        # Initialize embedding service
        self.embedding_service = EmbeddingService(
            model_name=embedding_model,
            pooling_strategy=pooling_strategy,
            normalize=normalize,
            max_length=max_length
        )
        self.embedding_dim = self.embedding_service.get_embedding_dim()

        # Initialize chunking service if enabled
        if enable_chunking:
            self.chunking_service = ChunkingService(
                tokenizer_name=embedding_model,
                chunk_size=max_length,
                overlap=chunk_overlap
            )
        else:
            self.chunking_service = None

        logger.info(f"✓ Code Similarity Store ready (dim={self.embedding_dim})")

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure collection exists with proper configuration."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=5000
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=100
                    )
                )
                logger.info(f"✓ Collection '{self.collection_name}' created")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")

            # Create payload indexes for faster filtering
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="label",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="language",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info("✓ Payload indexes created")
            except UnexpectedResponse:
                # Indexes might already exist
                pass

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def _generate_id(self, content: str) -> str:
        """Generate deterministic ID from content."""
        return hashlib.md5(content.encode()).hexdigest()

    def _encode_code(self, code: str) -> List[float]:
        """
        Generate embedding for code snippet using EmbeddingService.

        Args:
            code: Source code string

        Returns:
            Embedding vector as list of floats (L2 normalized if configured)
        """
        try:
            return self.embedding_service.encode_single(code).tolist()
        except Exception as e:
            logger.error(f"Failed to encode code: {e}")
            raise

    def upsert_code_samples(self, samples: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upsert code samples to Qdrant with optional chunking.

        Args:
            samples: List of sample dictionaries with:
                - id: int (database ID)
                - content: str (code content) - REQUIRED
                - label: str ("malicious" or "benign") - REQUIRED
                - source: str (data source)
                - language: str (programming language, default: "python")
                - metadata: dict (additional metadata)
            batch_size: Number of samples to process in each batch
        """
        if not samples:
            logger.warning("No samples to upsert")
            return

        logger.info(f"Upserting {len(samples)} code samples to Qdrant...")

        # Apply chunking if enabled
        if self.enable_chunking and self.chunking_service:
            logger.info("Applying sliding window chunking...")
            chunks = self.chunking_service.chunk_samples(samples)
            logger.info(f"Created {len(chunks)} chunks from {len(samples)} samples")
        else:
            # No chunking - process samples as-is
            chunks = []
            for sample in samples:
                chunks.append({
                    "content": sample.get("content", ""),
                    "db_id": sample.get("id"),
                    "chunk_index": 0,
                    "chunk_id": self._generate_id(sample.get("content", "")),
                    "total_chunks": 1,
                    "label": sample.get("label"),
                    "source": sample.get("source"),
                    "metadata": sample.get("metadata", {})
                })

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            points = []

            for chunk in batch:
                content = chunk.get("content", "")
                label = chunk.get("label", "")

                if not content or not label:
                    logger.warning(f"Skipping chunk - missing content or label")
                    continue

                # Convert label to binary
                label_binary = 1 if label.lower() == "malicious" else 0

                # Generate embedding
                try:
                    vector = self._encode_code(content)
                except Exception as e:
                    logger.warning(f"Failed to encode chunk: {e}")
                    continue

                # Use chunk_id as point ID
                point_id = chunk.get("chunk_id")

                # Prepare payload with chunk metadata
                payload = {
                    "db_id": chunk.get("db_id"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 1),
                    "code_content": content[:1000],  # Store truncated content
                    "label": label.lower(),
                    "label_binary": label_binary,
                    "source": chunk.get("source", "unknown"),
                    "language": "python",
                    "metadata": chunk.get("metadata", {})
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )

            # Batch upsert
            if points:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    logger.info(f"✓ Upserted batch {i // batch_size + 1}: {len(points)} chunks")
                except Exception as e:
                    logger.error(f"Failed to upsert batch {i // batch_size + 1}: {e}")

        logger.info(f"✓ Code sample synchronization complete")

    def search_similar_code(
        self,
        query_code: str,
        k: int = 3,
        filter_label: Optional[str] = None,
        balance_labels: bool = True,
        score_threshold: float = 0.3,
        aggregate_chunks: bool = True,
        aggregation_strategy: str = "max_score"
    ) -> List[Dict[str, Any]]:
        """
        Search for similar code samples using vector similarity with chunk aggregation.

        Args:
            query_code: Code to find similar samples for
            k: Number of results to return (at document level if aggregating)
            filter_label: Optional filter by label ("malicious" or "benign")
            balance_labels: If True, ensure mixed results (min 1 malicious, 1 benign)
            score_threshold: Minimum similarity score
            aggregate_chunks: Aggregate chunk results to document level
            aggregation_strategy: Strategy for aggregation ("max_score", "average_top_n", "weighted_avg")

        Returns:
            List of similar code samples with scores
        """
        # Generate query embedding (normalized if configured)
        try:
            query_vector = self._encode_code(query_code)
        except Exception as e:
            logger.error(f"Failed to encode query code: {e}")
            return []

        # Increase search limit if chunking is enabled to get more candidates
        search_limit = k * 5 if self.enable_chunking and aggregate_chunks else k

        # Build filter
        search_filter = None
        if filter_label:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="label",
                        match=models.MatchValue(value=filter_label.lower())
                    )
                ]
            )

        # Search
        try:
            if balance_labels and not filter_label:
                # Get separate results for each label
                malicious_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=max(search_limit // 2, 1),
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="label",
                                match=models.MatchValue(value="malicious")
                            )
                        ]
                    ),
                    score_threshold=score_threshold
                )

                benign_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=max(search_limit // 2, 1),
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="label",
                                match=models.MatchValue(value="benign")
                            )
                        ]
                    ),
                    score_threshold=score_threshold
                )

                # Combine and sort by score
                combined = list(malicious_results) + list(benign_results)
                combined.sort(key=lambda x: x.score, reverse=True)
                search_result = combined

            else:
                # Regular search
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=search_limit,
                    query_filter=search_filter,
                    score_threshold=score_threshold
                )

            # Format results
            results = []
            for hit in search_result:
                results.append({
                    "score": float(hit.score),
                    "code": hit.payload.get("code_content", ""),
                    "label": hit.payload.get("label", ""),
                    "label_binary": hit.payload.get("label_binary", 0),
                    "source": hit.payload.get("source", ""),
                    "language": hit.payload.get("language", "python"),
                    "db_id": hit.payload.get("db_id"),
                    "chunk_index": hit.payload.get("chunk_index", 0),
                    "total_chunks": hit.payload.get("total_chunks", 1)
                })

            # Aggregate chunks to document level if enabled
            if aggregate_chunks and self.enable_chunking:
                logger.debug(f"Aggregating {len(results)} chunk results to document level...")
                results = ResultAggregator.aggregate_results(
                    results,
                    strategy=aggregation_strategy,
                    top_n=3
                )
                # Limit to k documents
                results = results[:k]

            logger.debug(f"Found {len(results)} similar code samples")
            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_full_context(
        self,
        db_manager,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich results with full code context from database.

        Args:
            db_manager: DatasetManager instance
            results: Search results

        Returns:
            Results with full_content field added
        """
        return ResultAggregator.reconstruct_full_context(db_manager, results)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)

            # Count by label
            malicious_count = self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="label",
                            match=models.MatchValue(value="malicious")
                        )
                    ]
                )
            )

            benign_count = self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="label",
                            match=models.MatchValue(value="benign")
                        )
                    ]
                )
            )

            return {
                "name": self.collection_name,
                "total_samples": info.points_count,
                "malicious_samples": malicious_count.count,
                "benign_samples": benign_count.count,
                "embedding_dim": self.embedding_dim,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def clear_collection(self):
        """Clear all data from collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info("Collection cleared and recreated")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")


if __name__ == "__main__":
    # Example usage
    store = CodeSimilarityStore()

    # Test samples
    test_samples = [
        {
            "id": 1,
            "content": "import os\nos.system('rm -rf /')",
            "label": "malicious",
            "source": "test"
        },
        {
            "id": 2,
            "content": "import pandas as pd\ndf = pd.read_csv('data.csv')",
            "label": "benign",
            "source": "test"
        }
    ]

    # Upsert
    store.upsert_code_samples(test_samples)

    # Search
    results = store.search_similar_code("import os\nos.system('ls')", k=2)
    print(f"\nSearch results: {len(results)}")
    for r in results:
        print(f"  {r['label']} (score: {r['score']:.3f}): {r['code'][:50]}...")

    # Info
    info = store.get_collection_info()
    print(f"\nCollection info: {info}")
