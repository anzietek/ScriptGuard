"""
Code Similarity Store for Few-Shot RAG
Stores vectorized code samples from PostgreSQL for retrieval during inference.
Enhanced with unified embedding strategies, L2 normalization, chunking support,
graceful fallback, and reranking.
"""

import os
import yaml
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

# Load environment variables - find_dotenv() searches parent dirs automatically
load_dotenv(find_dotenv(usecwd=True))

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
        config_path: Optional[str] = None,
        timeout: int = 60,
        upsert_timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        ensure_label_balance: Optional[bool] = None,
        min_per_label: Optional[int] = None
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
            timeout: Timeout for Qdrant operations (seconds)
            upsert_timeout: Timeout for upsert operations (seconds)
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff factor for retries
            ensure_label_balance: Override config for label balancing (None = use config default)
            min_per_label: Override config for min samples per label (None = use config default)
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name
        self.enable_chunking = enable_chunking
        self.embedding_model = embedding_model

        # Store retry configuration
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.upsert_timeout = upsert_timeout

        # Get API key - prioritize parameter, fallback to env var, normalize empty strings to None
        self._api_key = api_key if api_key else os.getenv("QDRANT_API_KEY")
        if not self._api_key:  # Handles None, empty string, whitespace
            self._api_key = None

        logger.info(f"Initializing Code Similarity Store:")
        logger.info(f"  Host: {self.host}")
        logger.info(f"  Port: {self.port}")
        logger.info(f"  Collection: {self.collection_name}")
        logger.info(f"  Chunking: {'enabled' if enable_chunking else 'disabled'}")
        logger.info(f"  Timeout: {timeout}s (Upsert: {upsert_timeout}s)")
        logger.info(f"  Max Retries: {max_retries}")
        if self._api_key:
            logger.info(f"  API Key: ***{self._api_key[-8:]}")
            logger.info("  Auth: Enabled")
        else:
            logger.warning("  Auth: Disabled (no API key)")
            logger.warning("  Set QDRANT_API_KEY in .env if authentication is required")

        # Load configuration from environment variable or default
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "config.yaml")
        self.config = self._load_config(config_path)

        # Extract configuration parameters
        code_emb_config = self.config.get("code_embedding", {})

        # Score thresholds (model-specific)
        self.score_thresholds = self._load_score_thresholds(code_emb_config)

        # Graceful fallback configuration
        fallback_config = code_emb_config.get("graceful_fallback", {})
        self.graceful_fallback_enabled = fallback_config.get("enabled", True)
        self.fallback_threshold = fallback_config.get("fallback_threshold", 0.0)

        # Allow constructor parameters to override config for label balancing
        # This enables API to force disable label balancing for inference
        if ensure_label_balance is not None:
            self.ensure_label_balance = ensure_label_balance
        else:
            self.ensure_label_balance = fallback_config.get("ensure_label_balance", True)

        if min_per_label is not None:
            self.min_per_label = min_per_label
        else:
            self.min_per_label = fallback_config.get("min_per_label", 1)

        logger.info(f"  Graceful Fallback: {'enabled' if self.graceful_fallback_enabled else 'disabled'}")
        if self.graceful_fallback_enabled:
            logger.info(f"    - Fallback threshold: {self.fallback_threshold}")
            logger.info(f"    - Min per label: {self.min_per_label}")

        # Initialize Qdrant client with timeout
        if self._api_key:
            self.client = QdrantClient(
                url=f"{'https' if use_https else 'http'}://{self.host}:{self.port}",
                api_key=self._api_key,
                timeout=timeout
            )
        else:
            self.client = QdrantClient(host=self.host, port=self.port, timeout=timeout)

        # Initialize retry statistics tracking
        from scriptguard.utils.retry_utils import RetryStats
        self.retry_stats = RetryStats()

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
            # Read chunking strategy from config
            chunking_strategy = code_emb_config.get("chunking_strategy", "sliding_window")

            # Read hierarchical config
            hierarchical_config = code_emb_config.get("hierarchical", {})
            max_function_tokens = hierarchical_config.get("max_function_tokens", 1024)

            # Read sliding window config (with backward compatibility)
            sliding_config = code_emb_config.get("sliding_window", {})
            sliding_chunk_size = sliding_config.get("max_code_length", max_length)
            sliding_overlap = sliding_config.get("chunk_overlap", chunk_overlap)

            logger.info(f"  Chunking Strategy: {chunking_strategy}")
            if chunking_strategy == "hierarchical":
                logger.info(f"    - Max function tokens: {max_function_tokens}")
                logger.info(f"    - Fallback (sliding window): {sliding_chunk_size} tokens, {sliding_overlap} overlap")

            self.chunking_service = ChunkingService(
                tokenizer_name=embedding_model,
                chunk_size=sliding_chunk_size,
                overlap=sliding_overlap,
                max_function_tokens=max_function_tokens,
                strategy=chunking_strategy
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

    @property
    def api_key(self) -> Optional[str]:
        """Get API key (for backward compatibility)."""
        return self._api_key

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for diagnostics."""
        connection_type = "cloud" if self._api_key else "local"
        connection_url = (
            f"{'https' if self._api_key else 'http'}://{self.host}:{self.port}"
            if self._api_key else f"http://localhost:{self.port}"
        )

        return {
            "connection_type": connection_type,
            "connection_url": connection_url,
            "host": self.host,
            "port": self.port,
            "has_api_key": bool(self._api_key),
            "collection_name": self.collection_name
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file and substitute environment variables.

        Supports syntax: ${ENV_VAR:-default_value} or ${ENV_VAR}
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                def substitute_env_vars(obj):
                    """Recursively substitute environment variables in config."""
                    if isinstance(obj, dict):
                        return {k: substitute_env_vars(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [substitute_env_vars(item) for item in obj]
                    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                        env_expr = obj[2:-1]
                        if ":-" in env_expr:
                            env_var, default = env_expr.split(":-", 1)
                            return os.getenv(env_var, default)
                        else:
                            env_var = env_expr
                            return os.getenv(env_var, "")
                    else:
                        return obj

                config = substitute_env_vars(config)
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

            # COMPONENT 2 - STAGE 2C: Create feature indexes for hybrid search
            self._create_feature_indexes()

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def _create_feature_indexes(self):
        """
        Create payload indexes for static features (Component 2 - Stage 2C).

        Enables efficient filtering by:
        - Complexity metrics (entropy, complexity_score)
        - API usage flags (has_network_api, has_file_api, etc.)
        - Dangerous patterns (dangerous_api_calls, suspicious_combinations)

        Indexes are idempotent - safe to call multiple times.
        """
        from qdrant_client.http.exceptions import UnexpectedResponse

        logger.info("Creating feature payload indexes for hybrid search...")

        try:
            # Scalar indexes for range queries
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="features.entropy",
                    field_schema=models.PayloadSchemaType.FLOAT
                )
                logger.info("  ✓ Index created: features.entropy (FLOAT)")
            except UnexpectedResponse:
                logger.debug("  - Index features.entropy already exists")

            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="features.complexity_score",
                    field_schema=models.PayloadSchemaType.INTEGER
                )
                logger.info("  ✓ Index created: features.complexity_score (INTEGER)")
            except UnexpectedResponse:
                logger.debug("  - Index features.complexity_score already exists")

            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="features.code_length",
                    field_schema=models.PayloadSchemaType.INTEGER
                )
                logger.info("  ✓ Index created: features.code_length (INTEGER)")
            except UnexpectedResponse:
                logger.debug("  - Index features.code_length already exists")

            # Boolean indexes for exact match filtering
            for flag in ["has_network_api", "has_file_api", "has_process_api", "has_crypto_api",
                         "has_urls", "has_ips", "has_base64", "has_hex"]:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=f"features.{flag}",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    logger.info(f"  ✓ Index created: features.{flag} (KEYWORD)")
                except UnexpectedResponse:
                    logger.debug(f"  - Index features.{flag} already exists")

            # Array indexes for contains queries
            for field in ["dangerous_api_calls", "suspicious_combinations"]:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=f"features.{field}",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    logger.info(f"  ✓ Index created: features.{field} (KEYWORD array)")
                except UnexpectedResponse:
                    logger.debug(f"  - Index features.{field} already exists")

            logger.info("✓ Feature indexes created successfully")

        except Exception as e:
            logger.warning(f"Failed to create some feature indexes: {e}")
            logger.warning("Hybrid search may be slower without indexes, but will still work")

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

    def _encode_batch(self, batch_texts: List[str], batch_idx: int):
        """
        Encode batch with retry logic (handles transient GPU OOM).

        Args:
            batch_texts: List of text strings to encode
            batch_idx: Batch index for logging

        Returns:
            Embeddings array

        Raises:
            Exception: If encoding fails after all retries
        """
        from scriptguard.utils.retry_utils import retry_with_backoff

        @retry_with_backoff(
            max_retries=2,  # Fewer retries for embedding (GPU may be OOM)
            backoff_factor=1.5,
            exceptions=(Exception,)
        )
        def _do_encode():
            return self.embedding_service.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress=False
            )

        try:
            return _do_encode()
        except Exception as e:
            logger.error(f"Failed to encode batch {batch_idx} after retries: {e}")
            raise

    def _upsert_batch_with_retry(
        self,
        batch_points: List,
        batch_num: int,
        total_batches: int
    ) -> bool:
        """
        Upsert a single batch with retry logic.

        Args:
            batch_points: List of PointStruct objects to upsert
            batch_num: Current batch number (1-indexed)
            total_batches: Total number of batches

        Returns:
            True if successful, False if failed after all retries
        """
        from scriptguard.utils.retry_utils import retry_with_backoff

        @retry_with_backoff(
            max_retries=self.max_retries,
            backoff_factor=self.retry_backoff,
            exceptions=(Exception,),
            on_retry=lambda e, attempt: logger.warning(
                f"Batch {batch_num}/{total_batches} retry {attempt}/{self.max_retries}: {e}"
            )
        )
        def _do_upsert():
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_points,
                wait=True  # Wait for completion
            )
            return True

        try:
            _do_upsert()
            logger.info(f"✓ Batch {batch_num}/{total_batches}: {len(batch_points)} points")
            self.retry_stats.record_attempt("upsert_batch", True, 0)
            return True

        except Exception as e:
            logger.error(
                f"❌ Batch {batch_num}/{total_batches} FAILED after {self.max_retries} retries: {e}"
            )
            self.retry_stats.record_attempt("upsert_batch", False, self.max_retries)
            return False

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
                # Get both point_id and db_id
                point_id = sample.get("id")  # Qdrant point ID (always present)
                db_id = sample.get("db_id")  # Database ID (None for synthetic samples)

                # Generate parent metadata even for single-chunk documents
                import hashlib
                parent_id = hashlib.sha256(f"{point_id}_{content}".encode()).hexdigest()

                # Extract simple parent context
                lines = content.split('\n')[:5]
                parent_context = " ".join(line.strip() for line in lines if line.strip())[:500]

                chunks.append({
                    "content": content,
                    "db_id": db_id,  # Real database ID (can be None)
                    "chunk_index": 0,
                    "chunk_id": point_id,  # Use point_id as chunk_id
                    "total_chunks": 1,
                    "token_count": None,
                    "parent_id": parent_id,
                    "parent_context": parent_context,
                    "label": sample.get("label"),
                    "source": sample.get("source"),
                    "metadata": sample.get("metadata", {}),
                    "features": sample.get("features", {})  # CRITICAL FIX: Copy features from sample
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

            # BATCH ENCODE with retry - All chunks in this batch computed together (GPU efficient)
            try:
                embeddings = self._encode_batch(batch_texts, batch_idx // batch_size + 1)
            except Exception as e:
                logger.error(f"Failed to encode batch {batch_idx // batch_size + 1} after retries: {e}")
                continue  # Skip batch but log it

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
                    "metadata": chunk.get("metadata", {}),
                    # COMPONENT 2 - STAGE 2B: Static features for hybrid search
                    "features": chunk.get("features", {})  # NEW: Feature payload from vectorize_samples.py
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
        failed_batches = []

        logger.info(f"Uploading {len(all_points)} points in {total_upload_batches} batches with retry support")

        for i in range(0, len(all_points), upload_batch_size):
            batch_points = all_points[i:i + upload_batch_size]
            batch_num = i // upload_batch_size + 1

            success = self._upsert_batch_with_retry(batch_points, batch_num, total_upload_batches)

            if not success:
                failed_batches.append((batch_num, batch_points))

        # Report results
        if failed_batches:
            logger.error(
                f"⚠️ {len(failed_batches)}/{total_upload_batches} batches failed permanently. "
                f"Lost {sum(len(b[1]) for b in failed_batches)} points."
            )
            logger.error(f"Failed batch numbers: {[b[0] for b in failed_batches]}")
        else:
            logger.info(f"✓ All {total_upload_batches} batches uploaded successfully")

        # Print retry statistics
        stats = self.retry_stats.get_summary()
        logger.info("=" * 60)
        logger.info("VECTORIZATION RETRY STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total Attempts: {stats['total_attempts']}")
        logger.info(f"Total Retries: {stats['total_retries']}")
        logger.info(f"Total Failures: {stats['total_failures']}")
        logger.info(f"Success Rate: {stats['success_rate']}")
        logger.info("=" * 60)

        # WARNING: Alert if >5% failure rate
        if stats['total_failures'] > stats['total_attempts'] * 0.05:
            logger.warning(
                f"⚠️ HIGH FAILURE RATE: {stats['total_failures']} / "
                f"{stats['total_attempts']} batches failed permanently."
            )

        logger.info(f"✓ Code sample synchronization complete: {len(all_points)} points indexed")


    def _extract_query_features(self, query_code: str) -> Dict[str, Any]:
        """
        Extract static features from query code (Component 2 - Stage 2D).

        Args:
            query_code: Code to extract features from

        Returns:
            Feature dictionary matching schema in docs/FEATURE_SCHEMA.md
        """
        try:
            from scriptguard.steps.feature_extraction import (
                extract_ast_features,
                calculate_entropy,
                extract_api_patterns,
                extract_string_features
            )

            ast_features = extract_ast_features(query_code)
            entropy = calculate_entropy(query_code)
            api_patterns = extract_api_patterns(query_code)
            string_features = extract_string_features(query_code)

            return {
                "complexity_score": ast_features.get("complexity_score", 0),
                "entropy": entropy,
                "code_length": len(query_code),
                "code_lines": query_code.count("\n") + 1,
                "dangerous_api_calls": ast_features.get("dangerous_patterns", []),
                "suspicious_combinations": api_patterns.get("suspicious_combinations", []),
                "has_network_api": len(api_patterns.get("network_apis", [])) > 0,
                "has_file_api": len(api_patterns.get("file_apis", [])) > 0,
                "has_process_api": len(api_patterns.get("process_apis", [])) > 0,
                "has_crypto_api": len(api_patterns.get("crypto_apis", [])) > 0,
                "has_urls": string_features.get("has_urls", False),
                "has_ips": string_features.get("has_ips", False),
                "has_base64": string_features.get("has_base64", False),
                "has_hex": string_features.get("has_hex", False),
                "network_apis": api_patterns.get("network_apis", []),
                "file_apis": api_patterns.get("file_apis", []),
                "process_apis": api_patterns.get("process_apis", []),
                "crypto_apis": api_patterns.get("crypto_apis", [])
            }
        except Exception as e:
            logger.warning(f"Failed to extract query features: {e}")
            return {}

    def _build_hybrid_filter(
        self,
        filter_label: Optional[str] = None,
        feature_filters: Optional[Dict[str, Any]] = None,
        query_features: Optional[Dict[str, Any]] = None
    ) -> Optional[models.Filter]:
        """
        Build hybrid filter combining label and feature constraints (Component 2 - Stage 2D).

        Args:
            filter_label: Label filter ("malicious" or "benign")
            feature_filters: Manual feature constraints, e.g.:
                {
                    "min_entropy": 6.0,
                    "max_entropy": 8.0,
                    "min_complexity": 40,
                    "required_apis": ["has_network_api", "has_process_api"],
                    "min_dangerous_patterns": 2
                }
            query_features: Auto-extracted query features for smart boosting

        Returns:
            Qdrant Filter object or None if no filters specified
        """
        conditions = []

        # Label filter
        if filter_label:
            conditions.append(
                models.FieldCondition(
                    key="label",
                    match=models.MatchValue(value=filter_label.lower())
                )
            )

        # Manual feature filters
        if feature_filters:
            # Entropy range
            if "min_entropy" in feature_filters:
                conditions.append(
                    models.FieldCondition(
                        key="features.entropy",
                        range=models.Range(
                            gte=feature_filters["min_entropy"],
                            lte=feature_filters.get("max_entropy")
                        )
                    )
                )

            # Complexity range
            if "min_complexity" in feature_filters:
                conditions.append(
                    models.FieldCondition(
                        key="features.complexity_score",
                        range=models.Range(
                            gte=feature_filters["min_complexity"],
                            lte=feature_filters.get("max_complexity")
                        )
                    )
                )

            # Code length range
            if "min_code_length" in feature_filters:
                conditions.append(
                    models.FieldCondition(
                        key="features.code_length",
                        range=models.Range(
                            gte=feature_filters["min_code_length"],
                            lte=feature_filters.get("max_code_length")
                        )
                    )
                )

            # Required API flags (all must be true)
            if "required_apis" in feature_filters:
                for api_flag in feature_filters["required_apis"]:
                    conditions.append(
                        models.FieldCondition(
                            key=f"features.{api_flag}",
                            match=models.MatchValue(value=True)
                        )
                    )

            # Dangerous patterns count (must have at least N)
            if "min_dangerous_patterns" in feature_filters:
                # Note: Qdrant doesn't support array length queries directly
                # This is a workaround using match any (presence check)
                # Better approach: Pre-compute dangerous_pattern_count field
                logger.debug(f"min_dangerous_patterns filter requires custom reranking")

        # DISABLED: Auto-filtering is too aggressive - causes 0% recall
        # Use weighted BOOSTING instead (soft reranking, not hard filtering)
        if query_features and False:  # DISABLED
            # CRITICAL: Query has code execution patterns → filter for dangerous samples only
            query_dangerous = query_features.get("dangerous_api_calls", [])
            CRITICAL_PATTERNS = {'eval', 'exec', 'compile', '__import__'}
            has_critical = any(p in CRITICAL_PATTERNS for p in query_dangerous)

            if has_critical:
                # STRONG signal - limit search to samples with ANY dangerous API
                logger.info(f"Query has CRITICAL patterns {[p for p in query_dangerous if p in CRITICAL_PATTERNS]}, filtering for dangerous samples")
                # Use match any on dangerous_api_calls array
                conditions.append(
                    models.FieldCondition(
                        key="features.dangerous_api_calls",
                        match=models.MatchAny(any=list(CRITICAL_PATTERNS))  # Must have at least one critical pattern
                    )
                )

            # HIGH ENTROPY + dangerous APIs → likely obfuscated malware
            elif query_features.get("entropy", 0) > 6.5 and len(query_dangerous) > 0:
                logger.info(f"Query has high entropy ({query_features['entropy']:.2f}) + dangerous APIs, filtering for obfuscated samples")
                conditions.append(
                    models.FieldCondition(
                        key="features.entropy",
                        range=models.Range(gte=6.0)  # Very high entropy only
                    )
                )

            # DISABLED: Don't filter on common APIs (network, file) - creates cross-label bias
            # if query_features.get("has_network_api", False):
            #     logger.debug("Query uses network API, filtering for network samples")
            #     conditions.append(
            #         models.FieldCondition(
            #             key="features.has_network_api",
            #             match=models.MatchValue(value=True)
            #         )
            #     )

            # Query uses dangerous APIs → prefer samples with dangerous patterns
            if len(query_features.get("dangerous_api_calls", [])) > 0:
                logger.debug(f"Query has {len(query_features['dangerous_api_calls'])} dangerous patterns")
                # This is a soft preference, handled in reranking instead

        # Return filter only if we have conditions
        if conditions:
            return models.Filter(must=conditions)
        return None

    def _rerank_by_features(
        self,
        results: List[Dict[str, Any]],
        query_features: Dict[str, Any],
        boost_factor: float = 1.05
    ) -> List[Dict[str, Any]]:
        """
        Rerank results based on feature similarity (Component 2 - Stage 2D).

        Boosts scores for samples with similar feature characteristics:
        - Similar entropy levels
        - Matching API usage patterns
        - Similar complexity

        Args:
            results: Search results
            query_features: Query feature dict
            boost_factor: Multiplier for matching features (e.g., 1.05 = 5% boost)

        Returns:
            Reranked results (sorted by boosted score)
        """
        if not query_features or not results:
            return results

        logger.info(f"📊 Reranking {len(results)} results by feature similarity (boost_factor={boost_factor})...")

        boosted_count = 0
        total_boost_applied = 0.0

        for result in results:
            result_features = result.get("features", {})
            if not result_features:
                continue

            boost = 1.0

            # CONSERVATIVE BOOSTING: Only boost on rare/dangerous features
            # Common features (network, file APIs) appear in both benign and malicious code
            # Boosting on them creates label bias (benign Flask -> malicious C2)

            # DISABLED: Entropy matching (too broad - most code is 4-6 bits)
            # query_entropy = query_features.get("entropy", 0)
            # result_entropy = result_features.get("entropy", 0)
            # if abs(query_entropy - result_entropy) < 1.0:
            #     boost *= boost_factor

            # DISABLED: General API pattern matching (creates cross-label boosting)
            # benign network code (Flask) matches malicious network code (C2)
            # query_apis = {...}
            # result_apis = {...}
            # matching_apis = set(query_apis.keys()) & set(result_apis.keys())
            # if matching_apis:
            #     boost *= boost_factor ** len(matching_apis)

            # WEIGHTED BOOSTING: Different weights for different danger levels
            query_dangerous = set(query_features.get("dangerous_api_calls", []))
            result_dangerous = set(result_features.get("dangerous_api_calls", []))
            matching_dangerous = query_dangerous & result_dangerous

            if matching_dangerous:
                # Categorize patterns by danger level
                CRITICAL_PATTERNS = {'eval', 'exec', 'compile', '__import__'}  # Code execution - 3x boost
                HIGH_RISK_PATTERNS = {'system', 'popen', 'spawn'}  # Command execution - 2x boost
                MEDIUM_RISK_PATTERNS = {'decode', 'loads', 'load'}  # Deserialization - 1.2x boost (common in benign too)

                for pattern in matching_dangerous:
                    if pattern in CRITICAL_PATTERNS:
                        boost *= (boost_factor * 3.0)  # 1.15 * 3 = 3.45x boost
                        logger.debug(f"  CRITICAL pattern match: {pattern}")
                    elif pattern in HIGH_RISK_PATTERNS:
                        boost *= (boost_factor * 2.0)  # 1.15 * 2 = 2.3x boost
                        logger.debug(f"  HIGH RISK pattern match: {pattern}")
                    elif pattern in MEDIUM_RISK_PATTERNS:
                        boost *= (boost_factor * 0.5)  # 1.15 * 0.5 = 1.075x boost (minimal)
                        logger.debug(f"  MEDIUM RISK pattern match: {pattern}")
                    else:
                        # Unknown pattern - conservative boost
                        boost *= boost_factor
                        logger.debug(f"  Unknown dangerous pattern match: {pattern}")

            # Apply boost to score
            original_score = result.get("score", 0.0)
            result["score"] = original_score * boost
            if boost > 1.0:
                boosted_count += 1
                total_boost_applied += (boost - 1.0)
                logger.debug(f"  Boosted score: {original_score:.4f} → {result['score']:.4f} (×{boost:.2f})")

        # Re-sort by boosted score
        results = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)

        # Log summary
        if boosted_count > 0:
            avg_boost = (total_boost_applied / boosted_count) + 1.0
            logger.info(f"✓ Reranking complete: {boosted_count}/{len(results)} results boosted (avg boost: ×{avg_boost:.2f})")
        else:
            logger.warning(f"⚠️  No results were boosted (no feature matches found)")

        return results

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
        db_manager = None,
        # COMPONENT 2 - STAGE 2D: Hybrid search parameters
        feature_filters: Optional[Dict[str, Any]] = None,
        enable_feature_boosting: bool = False  # DISABLED: Hurts performance (70% -> 60% accuracy)
    ) -> List[Dict[str, Any]]:
        """
        Search for similar code samples with ROBUST "Always k" STRATEGY + HYBRID SEARCH.

        Multi-level search strategy:
        - Level 1: Search with score_threshold + filters (high quality)
        - Level 2: Fallback without score_threshold, keep hard filters (medium quality)
        - Level 3: Last resort - return best available, mark as low_confidence

        HYBRID SEARCH (Component 2 - Stage 2D):
        - Combines vector similarity with static feature filtering
        - Supports manual feature constraints (min_entropy, required_apis, etc.)
        - Auto feature boosting based on query characteristics
        - Feature-based reranking for improved relevance

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
            feature_filters: Manual feature constraints (NEW), e.g.:
                {
                    "min_entropy": 6.0,
                    "required_apis": ["has_network_api"],
                    "min_complexity": 40
                }
            enable_feature_boosting: Auto-boost results matching query features (NEW)

        Returns:
            List of up to k similar code samples with 100% original content (if fetch_full_content=True)

        Examples:
            # Find obfuscated malware
            results = store.search_similar_code(
                query_code="import socket; ...",
                k=5,
                feature_filters={"min_entropy": 6.0, "required_apis": ["has_network_api"]}
            )

            # Auto feature boosting
            results = store.search_similar_code(
                query_code="eval(input())",
                k=3,
                enable_feature_boosting=True  # Automatically prefer samples with dangerous APIs
            )
        """
        # Get threshold from config if not explicitly provided
        if score_threshold is None:
            score_threshold = self.get_threshold(threshold_mode)

        # COMPONENT 2 - STAGE 2D: Extract query features for hybrid search
        query_features = None
        if enable_feature_boosting or feature_filters:
            logger.info("🔬 Extracting query features for hybrid search...")
            query_features = self._extract_query_features(query_code)
            if query_features:
                logger.info(
                    f"✓ Query features: entropy={query_features.get('entropy', 0):.2f}, "
                    f"complexity={query_features.get('complexity_score', 0)}, "
                    f"dangerous_apis={query_features.get('dangerous_api_calls', [])}"
                )
            else:
                logger.warning("⚠️  Failed to extract query features")

        # Build hybrid filter (label + features)
        hybrid_filter = self._build_hybrid_filter(
            filter_label=filter_label,
            feature_filters=feature_filters,
            query_features=query_features if enable_feature_boosting else None
        )

        logger.info(
            f"🔍 Search: k={k}, threshold={score_threshold:.2f}, "
            f"balance={balance_labels}, fetch_full={fetch_full_content}, "
            f"feature_filters={'enabled' if feature_filters else 'none'}, "
            f"feature_boosting={'enabled' if enable_feature_boosting else 'disabled'}"
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
            filter_label=filter_label if not hybrid_filter else None,  # Use hybrid_filter if available
            balance_labels=balance_labels,
            score_threshold=score_threshold,
            custom_filter=hybrid_filter  # COMPONENT 2 - STAGE 2D: Pass hybrid filter
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

        # COMPONENT 2 - STAGE 2D: Feature-based reranking
        if enable_feature_boosting and query_features:
            logger.info(f"🚀 Applying feature-based reranking (boost_factor=1.15) to {len(results)} results...")
            results = self._rerank_by_features(results, query_features, boost_factor=1.15)  # Increased from 1.05 to 1.15
        elif enable_feature_boosting and not query_features:
            logger.warning("⚠️  Feature boosting enabled but no query features extracted!")

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
                filter_label=filter_label if not hybrid_filter else None,
                balance_labels=balance_labels,
                score_threshold=self.fallback_threshold,  # Effectively no threshold (0.0)
                custom_filter=hybrid_filter  # COMPONENT 2 - STAGE 2D: Use hybrid filter in fallback too
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

            # COMPONENT 2 - STAGE 2D: Feature reranking for fallback results too
            if enable_feature_boosting and query_features:
                fallback_results = self._rerank_by_features(fallback_results, query_features, boost_factor=1.15)  # Increased from 1.05 to 1.15

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

        # Log retrieval metrics (P1.2 fix)
        self._log_retrieval_metrics(results, query_metadata={
            "k": k,
            "balance_labels": balance_labels,
            "score_threshold": score_threshold,
            "threshold_mode": threshold_mode,
            "aggregate_chunks": aggregate_chunks,
            "enable_reranking": enable_reranking
        })

        logger.info(
            f"✓ Returning {len(results)}/{k} results "
            f"({sum(1 for r in results if r.get('low_confidence'))} low_confidence)"
        )
        return results

    def _log_retrieval_metrics(self, results: List[Dict[str, Any]], query_metadata: dict):
        """
        Log retrieval quality metrics for monitoring (P1.2 fix).

        Tracks:
        - Result quality (scores, confidence)
        - Label distribution (balance)
        - Fallback usage (level tracking)

        Args:
            results: List of retrieval results
            query_metadata: Query parameters for context
        """
        if not results:
            logger.warning("[Retrieval Metrics] Empty results returned")
            return

        # Extract metrics
        scores = [r.get('score', 0.0) for r in results]
        labels = [r.get('label', 'unknown') for r in results]
        low_confidence_count = sum(1 for r in results if r.get('low_confidence', False))

        # Calculate statistics
        metrics = {
            "num_results": len(results),
            "requested_k": query_metadata.get("k", "unknown"),
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
            "median_score": float(np.median(scores)) if scores else 0.0,
            "low_confidence_count": low_confidence_count,
            "low_confidence_rate": low_confidence_count / len(results) if results else 0.0,
            "label_distribution": {
                "malicious": labels.count("malicious"),
                "benign": labels.count("benign"),
                "unknown": labels.count("unknown")
            },
            "label_balance_achieved": (
                labels.count("malicious") > 0 and labels.count("benign") > 0
            ) if query_metadata.get("balance_labels") else None,
            "query_params": {
                "score_threshold": query_metadata.get("score_threshold"),
                "threshold_mode": query_metadata.get("threshold_mode"),
                "balance_labels": query_metadata.get("balance_labels"),
                "aggregate_chunks": query_metadata.get("aggregate_chunks"),
                "enable_reranking": query_metadata.get("enable_reranking")
            }
        }

        # Log metrics
        logger.info(
            f"[Retrieval Metrics] "
            f"Results: {metrics['num_results']}/{metrics['requested_k']}, "
            f"Avg score: {metrics['avg_score']:.3f}, "
            f"Min: {metrics['min_score']:.3f}, "
            f"Max: {metrics['max_score']:.3f}, "
            f"Low confidence: {metrics['low_confidence_count']}"
        )

        logger.info(
            f"[Retrieval Metrics] "
            f"Labels: malicious={metrics['label_distribution']['malicious']}, "
            f"benign={metrics['label_distribution']['benign']}, "
            f"unknown={metrics['label_distribution']['unknown']}, "
            f"Balanced: {metrics['label_balance_achieved']}"
        )

        # Alert if quality is degraded
        if metrics['avg_score'] < 0.2:
            logger.warning(
                f"[Retrieval Metrics] LOW QUALITY ALERT: Average score {metrics['avg_score']:.3f} "
                f"is below 0.2. Consider adjusting thresholds or model."
            )

        if metrics['low_confidence_rate'] > 0.5:
            logger.warning(
                f"[Retrieval Metrics] HIGH FALLBACK RATE: {metrics['low_confidence_rate']:.1%} "
                f"of results required Level 3 fallback. Collection may be too small."
            )

    def _search_with_filters(
        self,
        query_vector: List[float],
        limit: int,
        filter_label: Optional[str],
        balance_labels: bool,
        score_threshold: float,
        custom_filter: Optional[models.Filter] = None  # COMPONENT 2 - STAGE 2D: Support hybrid filter
    ) -> List[Dict[str, Any]]:
        """
        Execute search with specified filters and threshold.

        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            filter_label: Label filter (ignored if custom_filter provided)
            balance_labels: Balance label distribution
            score_threshold: Minimum score threshold
            custom_filter: Custom Qdrant filter (overrides filter_label)

        Returns:
            List of formatted results.
        """
        # Use custom filter if provided (hybrid search), otherwise build from filter_label
        if custom_filter:
            search_filter = custom_filter
        elif filter_label:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="label",
                        match=models.MatchValue(value=filter_label.lower())
                    )
                ]
            )
        else:
            search_filter = None

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
                    "code": hit.payload.get("code_preview", "") or hit.payload.get("content", ""),  # FIXED: code_preview (not code_content)
                    "code_preview": hit.payload.get("code_preview", ""),  # Keep separate for compatibility
                    "label": hit.payload.get("label", ""),
                    "label_binary": hit.payload.get("label_binary", 0),
                    "source": hit.payload.get("source", ""),
                    "language": hit.payload.get("language", "python"),
                    "db_id": hit.payload.get("db_id"),
                    "chunk_index": hit.payload.get("chunk_index", 0),
                    "total_chunks": hit.payload.get("total_chunks", 1),
                    "features": hit.payload.get("features", {}),  # CRITICAL FIX: Extract features for reranking
                    "payload": hit.payload  # Add full payload for API compatibility
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
