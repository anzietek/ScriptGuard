"""
Reranking Service for Few-Shot RAG
Implements heuristic and cross-encoder based reranking to improve retrieval quality.
"""

from typing import List, Dict, Any, Optional
import re
from difflib import SequenceMatcher

from scriptguard.utils.logger import logger


class RerankingService:
    """
    Two-stage retrieval reranker that improves RAG quality by:
    1. Boosting security-relevant code patterns
    2. Penalizing near-duplicate results for diversity
    3. (Optional) Cross-encoder rescoring
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        security_keywords: Optional[List[str]] = None,
        boost_factor: float = 1.2,
        diversity_penalty: float = 0.9,
        similarity_threshold: float = 0.95,
        cross_encoder_model: Optional[str] = None,
        cross_encoder_enabled: bool = False
    ):
        """
        Initialize Reranking Service.

        Args:
            strategy: Reranking strategy ("heuristic", "cross_encoder", "hybrid")
            security_keywords: List of security-relevant keywords to boost
            boost_factor: Score multiplier for results containing security keywords
            diversity_penalty: Score penalty for near-duplicate results
            similarity_threshold: Threshold for considering results as duplicates
            cross_encoder_model: Optional cross-encoder model name
            cross_encoder_enabled: Whether to use cross-encoder reranking
        """
        self.strategy = strategy
        self.boost_factor = boost_factor
        self.diversity_penalty = diversity_penalty
        self.similarity_threshold = similarity_threshold
        self.cross_encoder_enabled = cross_encoder_enabled

        # Default security keywords if none provided
        self.security_keywords = security_keywords or [
            "os.system", "subprocess", "exec", "eval", "compile",
            "__import__", "socket", "requests.get", "urllib",
            "pickle.loads", "base64.b64decode", "cryptography",
            "keylogger", "reverse_shell", "backdoor", "payload", "exploit"
        ]

        # Compile regex patterns for efficient matching
        self.security_patterns = [
            re.compile(re.escape(kw), re.IGNORECASE)
            for kw in self.security_keywords
        ]

        # Initialize cross-encoder if enabled
        self.cross_encoder = None
        if cross_encoder_enabled and cross_encoder_model:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                logger.info(f"✓ Cross-encoder loaded: {cross_encoder_model}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                self.cross_encoder_enabled = False
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                self.cross_encoder_enabled = False

    def rerank(
        self,
        query_code: str,
        results: List[Dict[str, Any]],
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results to improve relevance and diversity.

        Args:
            query_code: Original query code
            results: List of search results with 'score' and 'code' fields
            k: Optional limit on number of results to return

        Returns:
            Reranked list of results
        """
        if not results:
            return results

        logger.debug(f"Reranking {len(results)} results with strategy: {self.strategy}")

        # Apply heuristic reranking
        if self.strategy in ["heuristic", "hybrid"]:
            results = self._heuristic_rerank(results)

        # Apply cross-encoder reranking
        if self.strategy in ["cross_encoder", "hybrid"] and self.cross_encoder_enabled:
            results = self._cross_encoder_rerank(query_code, results)

        # Limit to k results if specified
        if k is not None:
            results = results[:k]

        logger.debug(f"Reranking complete. Top score: {results[0]['score']:.3f}")
        return results

    def _heuristic_rerank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply heuristic-based reranking:
        1. Boost scores for security-relevant patterns
        2. Penalize near-duplicates for diversity
        """
        if not results:
            return results

        # Step 1: Boost security-relevant results
        for result in results:
            code = result.get("code", "")
            original_score = result["score"]

            # Check for security keywords
            has_security_pattern = any(
                pattern.search(code) for pattern in self.security_patterns
            )

            if has_security_pattern:
                result["score"] = original_score * self.boost_factor
                result["boosted"] = True
                logger.debug(
                    f"Boosted result (label={result.get('label')}): "
                    f"{original_score:.3f} → {result['score']:.3f}"
                )

        # Step 2: Diversity penalty for near-duplicates
        selected_results = []
        for candidate in sorted(results, key=lambda x: x["score"], reverse=True):
            # Check similarity to already selected results
            is_duplicate = False
            for selected in selected_results:
                similarity = self._calculate_similarity(
                    candidate.get("code", ""),
                    selected.get("code", "")
                )

                if similarity >= self.similarity_threshold:
                    # Penalize score for near-duplicate
                    candidate["score"] *= self.diversity_penalty
                    candidate["diversity_penalized"] = True
                    is_duplicate = True
                    logger.debug(
                        f"Applied diversity penalty: "
                        f"similarity={similarity:.3f} to existing result"
                    )
                    break

            selected_results.append(candidate)

        # Sort by final score
        selected_results.sort(key=lambda x: x["score"], reverse=True)
        return selected_results

    def _cross_encoder_rerank(
        self,
        query_code: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply cross-encoder reranking for more accurate similarity scoring.
        """
        if not self.cross_encoder or not results:
            return results

        logger.debug("Applying cross-encoder reranking...")

        # Prepare pairs for cross-encoder
        pairs = [(query_code, result.get("code", "")) for result in results]

        try:
            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(pairs)

            # Update scores (blend with original cosine similarity)
            for result, ce_score in zip(results, ce_scores):
                original_score = result["score"]
                # Weighted average: 60% cross-encoder, 40% original
                blended_score = 0.6 * float(ce_score) + 0.4 * original_score
                result["score"] = blended_score
                result["original_score"] = original_score
                result["cross_encoder_score"] = float(ce_score)

            # Sort by new blended score
            results.sort(key=lambda x: x["score"], reverse=True)
            logger.debug("Cross-encoder reranking complete")

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Fall back to original scores
            pass

        return results

    @staticmethod
    def _calculate_similarity(code1: str, code2: str) -> float:
        """
        Calculate similarity between two code snippets using SequenceMatcher.
        Returns value between 0 and 1.
        """
        return SequenceMatcher(None, code1, code2).ratio()


def create_reranking_service(config: Dict[str, Any]) -> Optional[RerankingService]:
    """
    Factory function to create RerankingService from configuration.

    Args:
        config: Configuration dictionary (from config.yaml)

    Returns:
        RerankingService instance or None if disabled
    """
    reranking_config = config.get("code_embedding", {}).get("reranking", {})

    if not reranking_config.get("enabled", False):
        logger.info("Reranking disabled in configuration")
        return None

    strategy = reranking_config.get("strategy", "hybrid")

    # Heuristic configuration
    heuristic_config = reranking_config.get("heuristic", {})
    security_keywords = heuristic_config.get("security_keywords", None)
    boost_factor = heuristic_config.get("boost_factor", 1.2)
    diversity_penalty = heuristic_config.get("diversity_penalty", 0.9)
    similarity_threshold = heuristic_config.get("similarity_threshold", 0.95)

    # Cross-encoder configuration
    ce_config = reranking_config.get("cross_encoder", {})
    ce_enabled = ce_config.get("enabled", False)
    ce_model = ce_config.get("model", None)

    try:
        service = RerankingService(
            strategy=strategy,
            security_keywords=security_keywords,
            boost_factor=boost_factor,
            diversity_penalty=diversity_penalty,
            similarity_threshold=similarity_threshold,
            cross_encoder_model=ce_model,
            cross_encoder_enabled=ce_enabled
        )
        logger.info(f"✓ Reranking service initialized (strategy={strategy})")
        return service
    except Exception as e:
        logger.error(f"Failed to create reranking service: {e}")
        return None
