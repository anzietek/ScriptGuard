"""
Chunking Service - Sliding Window Chunking for Long Code Files
Handles tokenization-aware chunking with overlap for better context preservation.
"""

import hashlib
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

    def chunk_code(
        self,
        code: str,
        db_id: Optional[int] = None,
        label: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk code into overlapping segments.

        Args:
            code: Source code to chunk
            db_id: Database ID of the original sample
            label: Label (malicious/benign)
            source: Data source
            metadata: Additional metadata

        Returns:
            List of chunk dictionaries with:
                - content: Chunk text
                - db_id: Original document ID
                - chunk_index: Chunk position
                - chunk_id: Unique chunk identifier
                - total_chunks: Total number of chunks for this document
                - label: Label
                - source: Source
                - metadata: Metadata
        """
        if not code or not code.strip():
            return []

        # Quick token count to check if chunking needed
        try:
            test_tokens = self.tokenizer.encode(code, add_special_tokens=False, truncation=False)
        except:
            # If encoding fails on full text, we definitely need chunking
            test_tokens = list(range(self.chunk_size + 1))

        # If code fits in one chunk, return as-is
        if len(test_tokens) <= self.chunk_size:
            return [{
                "content": code,
                "db_id": db_id,
                "chunk_index": 0,
                "chunk_id": self._generate_chunk_id(code, 0),
                "total_chunks": 1,
                "label": label,
                "source": source,
                "metadata": metadata or {}
            }]

        # Split code into lines for better chunking
        lines = code.split('\n')

        chunks = []
        chunk_index = 0
        current_chunk_lines = []
        current_tokens = 0

        for line in lines:
            # Count tokens in this line
            line_tokens = len(self.tokenizer.encode(line, add_special_tokens=False, truncation=False))

            # If adding this line would exceed chunk_size, save current chunk
            if current_tokens + line_tokens > self.chunk_size and current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines)

                chunks.append({
                    "content": chunk_text,
                    "db_id": db_id,
                    "chunk_index": chunk_index,
                    "chunk_id": self._generate_chunk_id(code, chunk_index),
                    "total_chunks": -1,
                    "label": label,
                    "source": source,
                    "metadata": metadata or {}
                })

                chunk_index += 1

                # Keep last few lines for overlap (approximate)
                overlap_lines = max(1, self.overlap // 20)  # ~20 tokens per line estimate
                current_chunk_lines = current_chunk_lines[-overlap_lines:] if overlap_lines > 0 else []
                current_tokens = sum(len(self.tokenizer.encode(l, add_special_tokens=False, truncation=False))
                                   for l in current_chunk_lines)

            # Add line to current chunk
            current_chunk_lines.append(line)
            current_tokens += line_tokens

        # Add final chunk if any lines remain
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append({
                "content": chunk_text,
                "db_id": db_id,
                "chunk_index": chunk_index,
                "chunk_id": self._generate_chunk_id(code, chunk_index),
                "total_chunks": -1,
                "label": label,
                "source": source,
                "metadata": metadata or {}
            })

        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

        logger.debug(f"Chunked document (db_id={db_id}) into {len(chunks)} chunks")
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
    def reconstruct_full_context(
        db_manager,
        aggregated_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct full document context from database.

        Args:
            db_manager: DatasetManager instance
            aggregated_results: Aggregated document-level results

        Returns:
            Results with full content from database
        """
        enriched = []

        for result in aggregated_results:
            db_id = result.get("db_id")
            if db_id is not None:
                try:
                    # Fetch full content from database
                    full_sample = db_manager.get_sample_by_id(db_id)

                    if full_sample:
                        result["full_content"] = full_sample.get("content", "")
                        result["content_truncated"] = False
                    else:
                        result["content_truncated"] = True

                except Exception as e:
                    logger.warning(f"Failed to fetch full content for db_id={db_id}: {e}")
                    result["content_truncated"] = True

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
