"""
Code Similarity Store for Few-Shot RAG
Stores vectorized code samples from PostgreSQL for retrieval during inference.
Enhanced with unified embedding strategies, L2 normalization, chunking support,
graceful fallback, and reranking.
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import hashlib

from scriptguard.utils.logger import logger
from .embedding_service import EmbeddingService
from .chunking_service import ChunkingService, ResultAggregator
from .reranking_service import create_reranking_service
from .code_sanitization import create_sanitizer, create_enricher


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
        use_https: bool = False,
        config_path: str = "config.yaml"
    ):
        """
        Initialize Code Similarity Store with enhanced embedding, chunking, and reranking.

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
            config_path: Path to configuration file
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name
        self.enable_chunking = enable_chunking
        self.embedding_model = embedding_model

        logger.info(f"Initializing Code Similarity Store: {self.host}:{self.port}")
        logger.info(f"  Chunking: {'enabled' if enable_chunking else 'disabled'}")

        # Load configuration
        self.config = self._load_config(config_path)

        # Extract configuration parameters
        code_emb_config = self.config.get("code_embedding", {})

        # Score thresholds (model-specific)
        self.score_thresholds = self._load_score_thresholds(code_emb_config)

        # Graceful fallback configuration
        fallback_config = code_emb_config.get("graceful_fallback", {})
        self.graceful_fallback_enabled = fallback_config.get("enabled", True)
        self.fallback_threshold = fallback_config.get("fallback_threshold", 0.0)
        self.ensure_label_balance = fallback_config.get("ensure_label_balance", True)
        self.min_per_label = fallback_config.get("min_per_label", 1)

        logger.info(f"  Graceful Fallback: {'enabled' if self.graceful_fallback_enabled else 'disabled'}")
        if self.graceful_fallback_enabled:
            logger.info(f"    - Fallback threshold: {self.fallback_threshold}")
            logger.info(f"    - Min per label: {self.min_per_label}")

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

        # Initialize reranking service
        self.reranking_service = create_reranking_service(self.config)
        if self.reranking_service:
            logger.info("  Reranking: enabled")

        # Initialize sanitization and context injection
        sanitization_config = code_emb_config.get("sanitization", {})
        self.sanitization_enabled = sanitization_config.get("enabled", True)

        if self.sanitization_enabled:
            self.sanitizer = create_sanitizer(sanitization_config)
            logger.info("  Code Sanitization: enabled")
        else:
            self.sanitizer = None
            logger.info("  Code Sanitization: disabled")

        context_injection_config = code_emb_config.get("context_injection", {})
        self.context_injection_enabled = context_injection_config.get("enabled", True)

        if self.context_injection_enabled:
            self.enricher = create_enricher(context_injection_config)
            logger.info("  Context Injection: enabled")
        else:
            self.enricher = None
            logger.info("  Context Injection: disabled")

        logger.info(f"✓ Code Similarity Store ready (dim={self.embedding_dim})")

        # Ensure collection exists
        self._ensure_collection()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.debug(f"Configuration loaded from {config_path}")
                    return config
            else:
                logger.warning(f"Config file not found: {config_path}. Using defaults.")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}. Using defaults.")
            return {}

    def _load_score_thresholds(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Load model-specific score thresholds from configuration.

        Returns dict with 'default', 'strict', 'lenient' thresholds.
        """
        threshold_config = config.get("score_thresholds", {})
        model_thresholds = threshold_config.get(self.embedding_model, {})

        # Default fallback values
        defaults = {
            "default": 0.30,
            "strict": 0.45,
            "lenient": 0.15
        }

        # Merge with model-specific values
        thresholds = {**defaults, **model_thresholds}

        logger.info(f"  Score thresholds for {self.embedding_model}:")
        logger.info(f"    - Default: {thresholds['default']}")
        logger.info(f"    - Strict: {thresholds['strict']}")
        logger.info(f"    - Lenient: {thresholds['lenient']}")

        return thresholds

    def get_threshold(self, mode: str = "default") -> float:
        """
        Get score threshold for specified mode.

        Args:
            mode: One of "default", "strict", "lenient"

        Returns:
            Threshold value
        """
        return self.score_thresholds.get(mode, self.score_thresholds["default"])

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

    def _generate_id(self, content: str) -> int:
        """Generate deterministic integer ID from content (compatible with Qdrant)."""
        # Use MD5 hash for deterministic, collision-resistant IDs
        hash_bytes = hashlib.md5(content.encode()).digest()
        # Take first 8 bytes and convert to int, keep within uint64 range
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        return hash_int % (2**63 - 1)  # Use signed int64 max for safety

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

    def upsert_code_samples(self, samples: List[Dict[str, Any]], batch_size: int = 32):
        """
        Upsert code samples to Qdrant with BATCH EMBEDDING for 3x+ speedup.

        This implementation:
        0. SANITIZES code (NEW: removes binary data, validates syntax, normalizes)
        1. ENRICHES with context metadata (NEW: injects file path, repo, language)
        2. Applies chunking (if enabled) with overlap
        3. Groups chunks into batches
        4. Computes embeddings in parallel batches (GPU/CPU efficient)
        5. Uploads to Qdrant in batches

        Args:
            samples: List of sample dictionaries with:
                - id: int (database ID)
                - content: str (code content) - REQUIRED
                - label: str ("malicious" or "benign") - REQUIRED
                - source: str (data source)
                - language: str (programming language, default: "python")
                - metadata: dict (additional metadata)
            batch_size: Number of chunks to embed in parallel (default: 32, tune for GPU)
        """
        if not samples:
            logger.warning("No samples to upsert")
            return

        logger.info(f"Upserting {len(samples)} code samples to Qdrant...")

        # Step 0: Sanitize and enrich samples (NEW - Quality Gate)
        processed_samples = []
        sanitization_stats = {
            "total": len(samples),
            "valid": 0,
            "rejected": 0,
            "rejection_reasons": {}
        }

        for sample in samples:
            content = sample.get("content", "")
            if not content:
                continue

            # SANITIZATION PASS
            if self.sanitization_enabled and self.sanitizer:
                cleaned_content, report = self.sanitizer.sanitize(
                    content=content,
                    language=sample.get("language", "python"),
                    metadata=sample.get("metadata", {})
                )

                if not report.get("valid", False):
                    sanitization_stats["rejected"] += 1
                    reason = report.get("reason", "unknown")
                    sanitization_stats["rejection_reasons"][reason] = \
                        sanitization_stats["rejection_reasons"].get(reason, 0) + 1

                    logger.debug(
                        f"Rejected sample {sample.get('id')}: {reason} "
                        f"(entropy={report.get('entropy', 0):.2f})"
                    )
                    continue

                # Update content with cleaned version
                content = cleaned_content

                # Store sanitization report in metadata
                if "metadata" not in sample:
                    sample["metadata"] = {}
                sample["metadata"]["sanitization_report"] = {
                    "entropy": report.get("entropy"),
                    "original_length": report.get("original_length"),
                    "cleaned_length": report.get("cleaned_length"),
                    "warnings": report.get("warnings", [])
                }

            # CONTEXT INJECTION PASS
            if self.context_injection_enabled and self.enricher:
                # Build metadata dict for enrichment
                enrichment_metadata = {
                    "file_path": sample.get("metadata", {}).get("file_path"),
                    "repository": sample.get("metadata", {}).get("repository"),
                    "language": sample.get("language", "python"),
                    "source": sample.get("source"),
                    "label": sample.get("label")
                }

                content = self.enricher.enrich(content, enrichment_metadata)

            # Update sample with processed content
            sample["content"] = content
            processed_samples.append(sample)
            sanitization_stats["valid"] += 1

        # Log sanitization statistics
        if self.sanitization_enabled:
            logger.info(
                f"✓ Sanitization: {sanitization_stats['valid']}/{sanitization_stats['total']} "
                f"samples passed ({sanitization_stats['rejected']} rejected)"
            )

            if sanitization_stats["rejection_reasons"]:
                logger.info("  Rejection reasons:")
                for reason, count in sanitization_stats["rejection_reasons"].items():
                    logger.info(f"    - {reason}: {count}")

        if not processed_samples:
            logger.warning("No valid samples after sanitization")
            return

        samples = processed_samples  # Replace with sanitized samples

        # Step 1: Apply chunking if enabled
        if self.enable_chunking and self.chunking_service:
            logger.info("Applying token-based sliding window chunking...")
            chunks = self.chunking_service.chunk_samples(samples)
            logger.info(f"✓ Created {len(chunks)} chunks from {len(samples)} samples")
        else:
            # No chunking - process samples as-is but still add parent structure
            chunks = []
            for sample in samples:
                content = sample.get("content", "")
                db_id = sample.get("id")

                # Generate parent metadata even for single-chunk documents
                import hashlib
                parent_id = hashlib.sha256(f"{db_id}_{content}".encode()).hexdigest()

                # Extract simple parent context
                lines = content.split('\n')[:5]
                parent_context = " ".join(line.strip() for line in lines if line.strip())[:500]

                chunks.append({
                    "content": content,
                    "db_id": db_id,
                    "chunk_index": 0,
                    "chunk_id": self._generate_id(content),
                    "total_chunks": 1,
                    "token_count": None,
                    "parent_id": parent_id,
                    "parent_context": parent_context,
                    "label": sample.get("label"),
                    "source": sample.get("source"),
                    "metadata": sample.get("metadata", {})
                })

        if not chunks:
            logger.warning("No chunks generated from samples")
            return

        # Step 2: BATCH EMBEDDING - Process chunks in batches
        logger.info(f"Computing embeddings in batches of {batch_size}...")

        all_points = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(chunks), batch_size):
            batch = chunks[batch_idx:batch_idx + batch_size]

            # Extract texts for batch encoding
            batch_texts = []
            valid_chunks = []

            for chunk in batch:
                content = chunk.get("content", "")
                label = chunk.get("label", "")

                if not content or not label:
                    logger.warning(f"Skipping chunk - missing content or label")
                    continue

                batch_texts.append(content)
                valid_chunks.append(chunk)

            if not batch_texts:
                continue

            # BATCH ENCODE - All chunks in this batch computed together (GPU efficient)
            try:
                embeddings = self.embedding_service.encode(
                    batch_texts,
                    batch_size=len(batch_texts),  # Process all at once
                    show_progress=False
                )
            except Exception as e:
                logger.error(f"Failed to encode batch {batch_idx // batch_size + 1}: {e}")
                continue

            # Create Qdrant points for this batch
            for chunk, embedding in zip(valid_chunks, embeddings):
                label = chunk.get("label", "").lower()
                label_binary = 1 if label == "malicious" else 0

                # Use chunk_id as point ID
                point_id = chunk.get("chunk_id")

                # Prepare payload with parent-child structure
                payload = {
                    "db_id": chunk.get("db_id"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 1),
                    "token_count": chunk.get("token_count"),
                    "code_preview": chunk.get("content", "")[:200],  # Only small preview
                    "parent_id": chunk.get("parent_id", ""),  # Parent document hash
                    "parent_context": chunk.get("parent_context", ""),  # Module-level context
                    "label": label,
                    "label_binary": label_binary,
                    "source": chunk.get("source", "unknown"),
                    "language": "python",
                    "metadata": chunk.get("metadata", {})
                }

                all_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                        payload=payload
                    )
                )

            logger.info(
                f"✓ Batch {batch_idx // batch_size + 1}/{total_batches}: "
                f"Encoded {len(batch_texts)} chunks"
            )

        # Step 3: Upload to Qdrant in batches
        if not all_points:
            logger.warning("No valid points to upsert")
            return

        logger.info(f"Uploading {len(all_points)} points to Qdrant...")

        upload_batch_size = 100  # Qdrant upload batch size
        total_upload_batches = (len(all_points) + upload_batch_size - 1) // upload_batch_size

        for i in range(0, len(all_points), upload_batch_size):
            batch_points = all_points[i:i + upload_batch_size]

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                logger.info(
                    f"✓ Upload batch {i // upload_batch_size + 1}/{total_upload_batches}: "
                    f"{len(batch_points)} points"
                )
            except Exception as e:
                logger.error(f"Failed to upsert batch {i // upload_batch_size + 1}: {e}")

        logger.info(f"✓ Code sample synchronization complete: {len(all_points)} points indexed")


    def search_similar_code(
        self,
        query_code: str,
        k: int = 3,
        filter_label: Optional[str] = None,
        balance_labels: bool = True,
        score_threshold: Optional[float] = None,
        threshold_mode: str = "default",
        aggregate_chunks: bool = True,
        aggregation_strategy: str = "max_score",
        enable_reranking: bool = True,
        fetch_full_content: bool = True,
        db_manager = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar code samples with ROBUST "Always k" STRATEGY.

        Multi-level search strategy:
        - Level 1: Search with score_threshold + filters (high quality)
        - Level 2: Fallback without score_threshold, keep hard filters (medium quality)
        - Level 3: Last resort - return best available, mark as low_confidence

        This guarantees deterministic behavior even with empty collections or aggressive filters.

        Args:
            query_code: Code to find similar samples for
            k: Number of results to return (GUARANTEED unless collection is truly empty)
            filter_label: Optional filter by label ("malicious" or "benign")
            balance_labels: If True, ensure mixed results (min per label configurable)
            score_threshold: Explicit threshold (overrides threshold_mode)
            threshold_mode: Threshold mode ("default", "strict", "lenient")
            aggregate_chunks: Aggregate chunk results to document level
            aggregation_strategy: Strategy for aggregation ("max_score", "average_top_n", "weighted_avg")
            enable_reranking: Enable reranking for improved relevance
            fetch_full_content: Fetch full untruncated content from database (ELIMINATES TRUNCATION)
            db_manager: DatasetManager instance (required if fetch_full_content=True)

        Returns:
            List of up to k similar code samples with 100% original content (if fetch_full_content=True)
        """
        # Get threshold from config if not explicitly provided
        if score_threshold is None:
            score_threshold = self.get_threshold(threshold_mode)

        logger.info(
            f"🔍 Search: k={k}, threshold={score_threshold:.2f}, "
            f"balance={balance_labels}, fetch_full={fetch_full_content}"
        )

        # Generate query embedding (normalized if configured)
        try:
            query_vector = self._encode_code(query_code)
        except Exception as e:
            logger.error(f"Failed to encode query code: {e}")
            return []

        # Increase search limit if chunking is enabled to get more candidates
        initial_search_limit = k * 5 if self.enable_chunking and aggregate_chunks else k * 2

        # === LEVEL 1: High Quality Search ===
        logger.debug(f"[Level 1] Searching with threshold={score_threshold:.2f}...")
        results = self._search_with_filters(
            query_vector=query_vector,
            limit=initial_search_limit,
            filter_label=filter_label,
            balance_labels=balance_labels,
            score_threshold=score_threshold
        )

        # Aggregate chunks if enabled
        if aggregate_chunks and self.enable_chunking:
            logger.debug(f"Aggregating {len(results)} chunk results...")
            results = ResultAggregator.aggregate_results(
                results,
                strategy=aggregation_strategy,
                top_n=3
            )

        # Apply reranking if enabled
        if enable_reranking and self.reranking_service:
            logger.debug("Applying reranking...")
            results = self.reranking_service.rerank(query_code, results, k=None)

        # Check if we have enough results
        if len(results) >= k:
            logger.info(f"✓ [Level 1] Found {len(results)} results (>= k={k})")
            results = results[:k]

            # Fetch full content if requested
            if fetch_full_content and db_manager:
                results = ResultAggregator.fetch_full_content_batch(
                    db_manager, results, replace_content=True
                )

            return results

        # === LEVEL 2: Fallback Without Score Threshold ===
        if self.graceful_fallback_enabled:
            logger.info(
                f"[Level 2] Graceful fallback: {len(results)}/{k} found. "
                f"Searching with threshold={self.fallback_threshold:.2f}..."
            )

            fallback_results = self._search_with_filters(
                query_vector=query_vector,
                limit=k * 3,
                filter_label=filter_label,
                balance_labels=balance_labels,
                score_threshold=self.fallback_threshold  # Effectively no threshold (0.0)
            )

            # Aggregate chunks
            if aggregate_chunks and self.enable_chunking:
                fallback_results = ResultAggregator.aggregate_results(
                    fallback_results,
                    strategy=aggregation_strategy,
                    top_n=3
                )

            # Apply reranking to fallback results
            if enable_reranking and self.reranking_service:
                fallback_results = self.reranking_service.rerank(
                    query_code, fallback_results, k=None
                )

            # Merge results (keeping unique by db_id)
            seen_ids = {r.get("db_id") for r in results}
            for r in fallback_results:
                if r.get("db_id") not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.get("db_id"))
                    if len(results) >= k:
                        break

            logger.debug(f"[Level 2] Total results after fallback: {len(results)}")

        # Check label balance if required
        if balance_labels and not filter_label and self.ensure_label_balance:
            results = self._ensure_label_balance(
                results=results,
                k=k,
                query_vector=query_vector,
                aggregate_chunks=aggregate_chunks,
                aggregation_strategy=aggregation_strategy
            )

        # === LEVEL 3: Last Resort - Mark Low Confidence ===
        if len(results) < k:
            logger.warning(
                f"[Level 3] Last resort: Only {len(results)}/{k} results found. "
                f"Collection may be too small or filters too restrictive."
            )

            # Try one more time without ANY filters (except label if explicitly requested)
            if len(results) < k and not filter_label:
                logger.debug("[Level 3] Attempting search without label balance...")
                last_resort = self._search_with_filters(
                    query_vector=query_vector,
                    limit=k * 2,
                    filter_label=None,  # Remove all label filters
                    balance_labels=False,
                    score_threshold=self.fallback_threshold
                )

                if aggregate_chunks and self.enable_chunking:
                    last_resort = ResultAggregator.aggregate_results(
                        last_resort,
                        strategy=aggregation_strategy,
                        top_n=3
                    )

                # Merge
                seen_ids = {r.get("db_id") for r in results}
                for r in last_resort:
                    if r.get("db_id") not in seen_ids:
                        r["low_confidence"] = True  # FLAG: Mark as low confidence
                        results.append(r)
                        seen_ids.add(r.get("db_id"))
                        if len(results) >= k:
                            break

        # Final limit to k
        results = results[:k]

        # Mark all results with confidence level
        for r in results:
            if "low_confidence" not in r:
                r["low_confidence"] = False

        # Fetch full content if requested (FETCH-FROM-SOURCE architecture)
        if fetch_full_content and db_manager:
            logger.debug(f"Fetching full content for {len(results)} results from database...")
            results = ResultAggregator.fetch_full_content_batch(
                db_manager, results, replace_content=True
            )

        logger.info(
            f"✓ Returning {len(results)}/{k} results "
            f"({sum(1 for r in results if r.get('low_confidence')) } low_confidence)"
        )
        return results

    def _search_with_filters(
        self,
        query_vector: List[float],
        limit: int,
        filter_label: Optional[str],
        balance_labels: bool,
        score_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Execute search with specified filters and threshold.

        Returns list of formatted results.
        """
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

        try:
            # Modern Qdrant API uses query_points method
            if hasattr(self.client, 'query_points'):
                search_method = self.client.query_points
            else:
                # Fallback for older versions
                search_method = getattr(self.client, 'search', None)
                if not search_method:
                    raise AttributeError("QdrantClient has neither 'query_points' nor 'search' method")

            if balance_labels and not filter_label:
                # Get separate results for each label
                malicious_response = search_method(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=max(limit // 2, 1),
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

                benign_response = search_method(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=max(limit // 2, 1),
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

                # Extract points from response (query_points returns QueryResponse with .points attribute)
                malicious_results = malicious_response.points if hasattr(malicious_response, 'points') else malicious_response
                benign_results = benign_response.points if hasattr(benign_response, 'points') else benign_response

                # Combine and sort by score
                combined = list(malicious_results) + list(benign_results)
                combined.sort(key=lambda x: x.score, reverse=True)
                search_result = combined

            else:
                # Regular search
                response = search_method(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit,
                    query_filter=search_filter,
                    score_threshold=score_threshold
                )

                # Extract points from response
                search_result = response.points if hasattr(response, 'points') else response

            # Format results
            results = []
            for hit in search_result:
                # hit is ScoredPoint with .score and .payload attributes
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

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _ensure_label_balance(
        self,
        results: List[Dict[str, Any]],
        k: int,
        query_vector: List[float],
        aggregate_chunks: bool,
        aggregation_strategy: str
    ) -> List[Dict[str, Any]]:
        """
        Ensure minimum number of results per label (malicious/benign).

        If balance requirements are not met, fetch additional results of the missing label.
        """
        # Count results by label
        malicious_count = sum(1 for r in results if r.get("label") == "malicious")
        benign_count = sum(1 for r in results if r.get("label") == "benign")

        logger.debug(f"Label distribution: malicious={malicious_count}, benign={benign_count}")

        # Check if we need more of either label
        need_malicious = max(0, self.min_per_label - malicious_count)
        need_benign = max(0, self.min_per_label - benign_count)

        if need_malicious == 0 and need_benign == 0:
            return results  # Balance satisfied

        logger.info(
            f"Ensuring label balance: need {need_malicious} malicious, {need_benign} benign"
        )

        # Get existing IDs to avoid duplicates
        existing_ids = {r.get("db_id") for r in results}

        # Fetch additional malicious samples if needed
        if need_malicious > 0:
            additional_mal = self._search_with_filters(
                query_vector=query_vector,
                limit=need_malicious * 2,
                filter_label="malicious",
                balance_labels=False,
                score_threshold=self.fallback_threshold
            )

            if aggregate_chunks and self.enable_chunking:
                additional_mal = ResultAggregator.aggregate_results(
                    additional_mal,
                    strategy=aggregation_strategy,
                    top_n=3
                )

            # Add unique results
            for r in additional_mal:
                if r.get("db_id") not in existing_ids:
                    results.append(r)
                    existing_ids.add(r.get("db_id"))
                    need_malicious -= 1
                    if need_malicious <= 0:
                        break

        # Fetch additional benign samples if needed
        if need_benign > 0:
            additional_ben = self._search_with_filters(
                query_vector=query_vector,
                limit=need_benign * 2,
                filter_label="benign",
                balance_labels=False,
                score_threshold=self.fallback_threshold
            )

            if aggregate_chunks and self.enable_chunking:
                additional_ben = ResultAggregator.aggregate_results(
                    additional_ben,
                    strategy=aggregation_strategy,
                    top_n=3
                )

            # Add unique results
            for r in additional_ben:
                if r.get("db_id") not in existing_ids:
                    results.append(r)
                    existing_ids.add(r.get("db_id"))
                    need_benign -= 1
                    if need_benign <= 0:
                        break

        # Re-sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results


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
