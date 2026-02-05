"""
Chunking Service - Sliding Window Chunking for Long Code Files
Handles tokenization-aware chunking with overlap for better context preservation.
"""

import hashlib
import warnings
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
from scriptguard.utils.logger import logger


class ChunkingService:
    """
    Token-aware chunking service for code files.

    Uses sliding window approach to handle files exceeding max token length.
    Preserves metadata and enables document-level aggregation.
    """

    def __init__(
        self,
        tokenizer_name: str = "microsoft/unixcoder-base",
        chunk_size: int = 512,
        overlap: int = 64
    ):
        """
        Initialize chunking service.

        Args:
            tokenizer_name: Tokenizer to use for token counting
            chunk_size: Maximum tokens per chunk
            overlap: Overlap tokens between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        logger.info(f"Initializing ChunkingService:")
        logger.info(f"  Tokenizer: {tokenizer_name}")
        logger.info(f"  Chunk size: {chunk_size} tokens")
        logger.info(f"  Overlap: {overlap} tokens")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info("✓ Chunking service ready")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _generate_chunk_id(self, content: str, chunk_index: int) -> int:
        """Generate unique integer ID for chunk (Qdrant-compatible)."""
        # Use MD5 hash for deterministic, collision-resistant IDs
        # Convert to int but keep within uint64 range (2^64 - 1)
        hash_bytes = hashlib.md5(f"{content}_{chunk_index}".encode()).digest()
        # Take first 8 bytes and convert to int
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        # Ensure it's positive and within uint64 range
        return hash_int % (2**63 - 1)  # Use signed int64 max for safety

    def _generate_parent_id(self, content: str, db_id: Optional[int] = None) -> str:
        """Generate parent document ID (hash of full content)."""
        # Use SHA256 for parent ID to avoid collisions
        identifier = f"{db_id}_{content}" if db_id else content
        return hashlib.sha256(identifier.encode()).hexdigest()

    def _extract_parent_context(self, code: str, max_length: int = 500) -> str:
        """
        Extract parent context from code (module-level info).

        Returns:
            String containing module docstring, imports, and class/function signatures
        """
        import ast

        try:
            # Suppress SyntaxWarning for invalid escape sequences in analyzed code
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=SyntaxWarning)
                tree = ast.parse(code)
            context_parts = []

            # Extract module docstring
            if ast.get_docstring(tree):
                docstring = ast.get_docstring(tree)[:200]  # First 200 chars
                context_parts.append(f"# Module: {docstring}")

            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            if imports:
                context_parts.append(f"# Imports: {', '.join(imports[:10])}")

            # Extract top-level function/class signatures
            signatures = []
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    args_list = [arg.arg for arg in node.args.args[:3]]  # First 3 args
                    sig = f"def {node.name}({', '.join(args_list)}...)"
                    signatures.append(sig)
                elif isinstance(node, ast.ClassDef):
                    signatures.append(f"class {node.name}")

            if signatures:
                context_parts.append(f"# Definitions: {'; '.join(signatures[:5])}")

            parent_context = " | ".join(context_parts)
            return parent_context[:max_length]

        except Exception as e:
            # If AST parsing fails, use first few lines as context
            lines = code.split('\n')[:5]
            return " ".join(line.strip() for line in lines if line.strip())[:max_length]

    def chunk_code(
        self,
        code: str,
        db_id: Optional[int] = None,
        label: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk code into overlapping segments using TOKEN-BASED sliding window.

        This implementation uses precise token-based boundaries instead of line-based
        approximations. This ensures consistent chunk sizes regardless of code formatting
        (minified, long lines, etc.).

        Args:
            code: Source code to chunk
            db_id: Database ID of the original sample
            label: Label (malicious/benign)
            source: Data source
            metadata: Additional metadata

        Returns:
            List of chunk dictionaries with:
                - content: Chunk text (decoded from tokens)
                - db_id: Original document ID
                - chunk_index: Chunk position
                - chunk_id: Unique chunk identifier
                - total_chunks: Total number of chunks for this document
                - token_count: Actual token count of this chunk
                - parent_id: Hash of parent document
                - parent_context: Module-level context (imports, signatures)
                - label: Label
                - source: Source
                - metadata: Metadata
        """
        if not code or not code.strip():
            return []

        # Generate parent metadata ONCE for all chunks
        parent_id = self._generate_parent_id(code, db_id)
        parent_context = self._extract_parent_context(code)

        # Tokenize the entire code WITHOUT truncation
        try:
            tokens = self.tokenizer.encode(
                code,
                add_special_tokens=False,
                truncation=False
            )
        except Exception as e:
            logger.error(f"Failed to tokenize code: {e}")
            return []

        # If code fits in one chunk, return as-is
        if len(tokens) <= self.chunk_size:
            return [{
                "content": code,
                "db_id": db_id,
                "chunk_index": 0,
                "chunk_id": self._generate_chunk_id(code, 0),
                "total_chunks": 1,
                "token_count": len(tokens),
                "parent_id": parent_id,
                "parent_context": parent_context,
                "label": label,
                "source": source,
                "metadata": metadata or {}
            }]

        # Calculate stride (step size between chunks)
        # Default: chunk_size - overlap (e.g., 512 - 64 = 448 tokens per step)
        stride = self.chunk_size - self.overlap

        if stride <= 0:
            logger.warning(f"Invalid stride={stride} (overlap >= chunk_size). Using stride=chunk_size/2")
            stride = max(1, self.chunk_size // 2)

        # Generate chunks using sliding window
        chunks = []
        chunk_index = 0
        start_idx = 0

        while start_idx < len(tokens):
            # Extract chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode tokens back to text
            try:
                chunk_text = self.tokenizer.decode(
                    chunk_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            except Exception as e:
                logger.warning(f"Failed to decode chunk {chunk_index}: {e}")
                # Move to next chunk
                start_idx += stride
                continue

            # Create chunk metadata with parent-child structure
            chunks.append({
                "content": chunk_text,
                "db_id": db_id,
                "chunk_index": chunk_index,
                "chunk_id": self._generate_chunk_id(chunk_text, chunk_index),
                "total_chunks": -1,  # Will be updated later
                "token_count": len(chunk_tokens),
                "start_token": start_idx,
                "end_token": end_idx,
                "parent_id": parent_id,  # Parent document hash
                "parent_context": parent_context,  # Module-level context
                "label": label,
                "source": source,
                "metadata": metadata or {}
            })

            chunk_index += 1

            # Move to next chunk position
            start_idx += stride

            # Stop if we've reached the end
            if end_idx >= len(tokens):
                break

        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

        logger.debug(
            f"Chunked document (db_id={db_id}, parent_id={parent_id[:8]}...): {len(tokens)} tokens → "
            f"{len(chunks)} chunks (size={self.chunk_size}, overlap={self.overlap}, stride={stride})"
        )
        return chunks

    def chunk_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple code samples.

        Args:
            samples: List of sample dicts with:
                - id: Database ID
                - content: Code content
                - label: Label
                - source: Source
                - metadata: Metadata (optional)

        Returns:
            List of all chunks from all samples
        """
        all_chunks = []

        for sample in samples:
            chunks = self.chunk_code(
                code=sample.get("content", ""),
                db_id=sample.get("id"),
                label=sample.get("label"),
                source=sample.get("source"),
                metadata=sample.get("metadata")
            )
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(samples)} samples into {len(all_chunks)} chunks")
        return all_chunks

    def get_stats(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get chunking statistics for a list of samples.

        Args:
            samples: List of sample dicts

        Returns:
            Statistics dictionary
        """
        chunks = self.chunk_samples(samples)

        # Count documents that needed chunking
        docs_with_multiple_chunks = len(set(
            c["db_id"] for c in chunks if c["total_chunks"] > 1
        ))

        # Average chunks per document
        from collections import Counter
        chunks_per_doc = Counter(c["db_id"] for c in chunks)
        avg_chunks = sum(chunks_per_doc.values()) / len(chunks_per_doc) if chunks_per_doc else 0

        return {
            "total_samples": len(samples),
            "total_chunks": len(chunks),
            "docs_with_multiple_chunks": docs_with_multiple_chunks,
            "avg_chunks_per_doc": avg_chunks,
            "max_chunks_per_doc": max(chunks_per_doc.values()) if chunks_per_doc else 0
        }


class ResultAggregator:
    """
    Aggregates chunk-level search results to document level.

    Strategies:
    - max_score: Use the best-scoring chunk per document
    - average_top_n: Average the top N chunks per document
    - weighted_avg: Weight chunks by their scores
    """

    @staticmethod
    def aggregate_results(
        results: List[Dict[str, Any]],
        strategy: str = "max_score",
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Aggregate chunk results to document level.

        Args:
            results: List of chunk results with:
                - db_id: Original document ID
                - score: Similarity score
                - chunk_index: Chunk position
                - content: Chunk content
                - label: Label
                - Other metadata
            strategy: Aggregation strategy ("max_score", "average_top_n", "weighted_avg")
            top_n: Number of top chunks to consider for averaging

        Returns:
            List of aggregated document-level results
        """
        if not results:
            return []

        # Group results by db_id
        from collections import defaultdict
        doc_chunks = defaultdict(list)

        for result in results:
            db_id = result.get("db_id")
            if db_id is not None:
                doc_chunks[db_id].append(result)

        # Aggregate each document
        aggregated = []

        for db_id, chunks in doc_chunks.items():
            if strategy == "max_score":
                # Use the best chunk
                best_chunk = max(chunks, key=lambda x: x["score"])
                aggregated.append({
                    **best_chunk,
                    "aggregation_strategy": "max_score",
                    "num_chunks": len(chunks),
                    "all_chunk_scores": [c["score"] for c in chunks]
                })

            elif strategy == "average_top_n":
                # Average top N chunks
                sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
                top_chunks = sorted_chunks[:top_n]

                avg_score = sum(c["score"] for c in top_chunks) / len(top_chunks)

                # Use the best chunk's content but average score
                best_chunk = top_chunks[0].copy()
                best_chunk["score"] = avg_score
                best_chunk["aggregation_strategy"] = f"average_top_{top_n}"
                best_chunk["num_chunks"] = len(chunks)
                best_chunk["all_chunk_scores"] = [c["score"] for c in chunks]

                aggregated.append(best_chunk)

            elif strategy == "weighted_avg":
                # Weighted average by scores
                total_weight = sum(c["score"] for c in chunks)
                if total_weight > 0:
                    weighted_score = sum(c["score"] ** 2 for c in chunks) / total_weight
                else:
                    weighted_score = 0

                best_chunk = max(chunks, key=lambda x: x["score"]).copy()
                best_chunk["score"] = weighted_score
                best_chunk["aggregation_strategy"] = "weighted_avg"
                best_chunk["num_chunks"] = len(chunks)
                best_chunk["all_chunk_scores"] = [c["score"] for c in chunks]

                aggregated.append(best_chunk)

        # Sort by aggregated score
        aggregated.sort(key=lambda x: x["score"], reverse=True)

        logger.debug(f"Aggregated {len(results)} chunks to {len(aggregated)} documents")
        return aggregated

    @staticmethod
    def fetch_full_content_batch(
        db_manager,
        aggregated_results: List[Dict[str, Any]],
        replace_content: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch fetch full document content from database (ELIMINATES PAYLOAD TRUNCATION).

        This is the "Fetch-from-Source" architecture:
        - Qdrant returns only metadata (db_id, scores, chunk info)
        - Full, untruncated code is fetched from PostgreSQL (Source of Truth)
        - Ensures Few-Shot prompt gets 100% original code

        Args:
            db_manager: DatasetManager instance
            aggregated_results: Aggregated document-level results with db_id
            replace_content: If True, replace 'content' field with full content;
                           If False, add as 'full_content' (keeps truncated preview)

        Returns:
            Results with full content from database
        """
        if not aggregated_results:
            return []

        # Extract all db_ids
        db_ids = [r.get("db_id") for r in aggregated_results if r.get("db_id") is not None]

        if not db_ids:
            logger.warning("No valid db_ids found in results")
            return aggregated_results

        logger.debug(f"Batch fetching full content for {len(db_ids)} documents from database...")

        # Batch fetch from database (much more efficient than individual queries)
        try:
            from scriptguard.database.db_schema import get_connection, return_connection
            conn = get_connection()
            cursor = conn.cursor()

            # Use parameterized query with IN clause
            placeholders = ','.join(['%s'] * len(db_ids))
            query = f"""
                SELECT id, content, label, source, url, metadata, created_at
                FROM samples
                WHERE id IN ({placeholders})
            """

            cursor.execute(query, tuple(db_ids))
            rows = cursor.fetchall()

            # Build lookup map: db_id -> full sample data
            sample_map = {}
            for row in rows:
                sample_map[row["id"]] = {
                    "id": row["id"],
                    "content": row["content"],
                    "label": row["label"],
                    "source": row["source"],
                    "url": row["url"],
                    "metadata": row["metadata"] or {},
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None
                }

            cursor.close()
            return_connection(conn)

            logger.debug(f"✓ Fetched {len(sample_map)} full samples from database")

        except Exception as e:
            logger.error(f"Failed to batch fetch samples from database: {e}")
            # Fall back to returning results without full content
            for result in aggregated_results:
                result["content_fetch_failed"] = True
            return aggregated_results

        # Enrich results with full content
        enriched = []
        for result in aggregated_results:
            db_id = result.get("db_id")

            if db_id in sample_map:
                full_sample = sample_map[db_id]

                if replace_content:
                    # Replace truncated content with full content
                    result["content"] = full_sample["content"]
                    result["content_truncated"] = False
                else:
                    # Keep both (preview + full)
                    result["full_content"] = full_sample["content"]
                    result["content_truncated"] = False

                # Optionally enrich with other DB fields
                result["url"] = full_sample.get("url")
                result["db_metadata"] = full_sample.get("metadata", {})

            else:
                logger.warning(f"db_id={db_id} not found in database")
                result["content_truncated"] = True
                result["content_fetch_failed"] = True

            enriched.append(result)

        return enriched


if __name__ == "__main__":
    # Test chunking service
    chunker = ChunkingService(
        tokenizer_name="microsoft/unixcoder-base",
        chunk_size=128,
        overlap=16
    )

    # Test with long code
    long_code = "\n".join([f"def function_{i}():\n    return {i}" for i in range(100)])

    chunks = chunker.chunk_code(
        code=long_code,
        db_id=1,
        label="benign",
        source="test"
    )

    print(f"Created {len(chunks)} chunks from long code")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(chunk['content'])} chars")
        print(f"  Chunk index: {chunk['chunk_index']}/{chunk['total_chunks']}")

    # Test aggregation
    mock_results = [
        {"db_id": 1, "score": 0.9, "chunk_index": 0, "content": "chunk 0"},
        {"db_id": 1, "score": 0.7, "chunk_index": 1, "content": "chunk 1"},
        {"db_id": 2, "score": 0.8, "chunk_index": 0, "content": "chunk 0"},
    ]

    aggregated = ResultAggregator.aggregate_results(mock_results, strategy="max_score")
    print(f"\n\nAggregated {len(mock_results)} chunks to {len(aggregated)} docs")
    for doc in aggregated:
        print(f"  db_id={doc['db_id']}, score={doc['score']:.3f}, num_chunks={doc['num_chunks']}")
