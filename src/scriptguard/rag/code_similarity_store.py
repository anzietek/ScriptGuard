"""
Code Similarity Store for Few-Shot RAG
Stores vectorized code samples from PostgreSQL for retrieval during inference.
"""

import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from transformers import AutoTokenizer, AutoModel
import torch
import hashlib

from scriptguard.utils.logger import logger


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
        api_key: Optional[str] = None,
        use_https: bool = False
    ):
        """
        Initialize Code Similarity Store.

        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection (default: "code_samples")
            embedding_model: Code embedding model
                Options:
                - "microsoft/unixcoder-base" (recommended for code)
                - "Salesforce/codet5p-110m-embedding" (alternative)
            api_key: Optional API key for Qdrant Cloud
            use_https: Use HTTPS connection
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        logger.info(f"Initializing Code Similarity Store: {self.host}:{self.port}")

        # Initialize Qdrant client
        if api_key:
            self.client = QdrantClient(
                url=f"{'https' if use_https else 'http'}://{self.host}:{self.port}",
                api_key=api_key
            )
        else:
            self.client = QdrantClient(host=self.host, port=self.port)

        # Initialize code embedding model
        logger.info(f"Loading code embedding model: {self.embedding_model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.model = AutoModel.from_pretrained(
                self.embedding_model_name,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()

            # Get embedding dimension
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt", truncation=True, max_length=512)
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
                test_output = self.model(**test_input)
                self.embedding_dim = test_output.last_hidden_state[:, 0, :].shape[-1]

            logger.info(f"✓ Code embedding model loaded (dim={self.embedding_dim}, device={self.device})")

        except Exception as e:
            logger.error(f"Failed to load code embedding model: {e}")
            raise

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
        Generate embedding for code snippet.

        Args:
            code: Source code string

        Returns:
            Embedding vector as list of floats
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to encode code: {e}")
            raise

    def upsert_code_samples(self, samples: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upsert code samples to Qdrant.

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

        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            points = []

            for sample in batch:
                content = sample.get("content", "")
                label = sample.get("label", "")

                if not content or not label:
                    logger.warning(f"Skipping sample {sample.get('id')} - missing content or label")
                    continue

                # Convert label to binary
                label_binary = 1 if label.lower() == "malicious" else 0

                # Generate embedding
                try:
                    vector = self._encode_code(content)
                except Exception as e:
                    logger.warning(f"Failed to encode sample {sample.get('id')}: {e}")
                    continue

                # Generate point ID
                point_id = self._generate_id(content)

                # Prepare payload
                payload = {
                    "db_id": sample.get("id"),
                    "code_content": content[:1000],  # Store truncated content
                    "label": label.lower(),
                    "label_binary": label_binary,
                    "source": sample.get("source", "unknown"),
                    "language": sample.get("language", "python"),
                    "metadata": sample.get("metadata", {})
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
                    logger.info(f"✓ Upserted batch {i // batch_size + 1}: {len(points)} samples")
                except Exception as e:
                    logger.error(f"Failed to upsert batch {i // batch_size + 1}: {e}")

        logger.info(f"✓ Code sample synchronization complete")

    def search_similar_code(
        self,
        query_code: str,
        k: int = 3,
        filter_label: Optional[str] = None,
        balance_labels: bool = True,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar code samples using vector similarity.

        Args:
            query_code: Code to find similar samples for
            k: Number of results to return
            filter_label: Optional filter by label ("malicious" or "benign")
            balance_labels: If True, ensure mixed results (min 1 malicious, 1 benign)
            score_threshold: Minimum similarity score

        Returns:
            List of similar code samples with scores
        """
        # Generate query embedding
        try:
            query_vector = self._encode_code(query_code)
        except Exception as e:
            logger.error(f"Failed to encode query code: {e}")
            return []

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
                    limit=max(k // 2, 1),
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
                    limit=max(k // 2, 1),
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
                search_result = combined[:k]

            else:
                # Regular search
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=k,
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
                    "db_id": hit.payload.get("db_id")
                })

            logger.debug(f"Found {len(results)} similar code samples")
            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

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
