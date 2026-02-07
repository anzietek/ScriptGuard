"""
Offline Evaluation: Compare Baseline vs Enhanced RAG
Measures MRR@k and Recall@k on a test set to validate improvements.
"""

import yaml
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from scriptguard.database.dataset_manager import DatasetManager
from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.utils.logger import logger


def calculate_mrr_at_k(results: List[Dict[str, Any]], relevant_label: str, k: int = 5) -> float:
    """
    Calculate Mean Reciprocal Rank at K.

    MRR@k = 1/|Q| * sum(1/rank_i) where rank_i is the position of first relevant result

    Args:
        results: Search results with 'label' field
        relevant_label: Label to consider as relevant
        k: Consider only top-k results

    Returns:
        MRR score (0.0 to 1.0)
    """
    results = results[:k]

    for i, result in enumerate(results, 1):
        if result.get("label") == relevant_label:
            return 1.0 / i

    return 0.0


def calculate_recall_at_k(results: List[Dict[str, Any]], relevant_label: str, k: int = 5) -> float:
    """
    Calculate Recall at K.

    Recall@k = 1 if any of top-k results are relevant, else 0

    Args:
        results: Search results with 'label' field
        relevant_label: Label to consider as relevant
        k: Consider only top-k results

    Returns:
        Recall score (0.0 or 1.0 for single query)
    """
    results = results[:k]

    for result in results:
        if result.get("label") == relevant_label:
            return 1.0

    return 0.0


def calculate_precision_at_k(results: List[Dict[str, Any]], relevant_label: str, k: int = 5) -> float:
    """
    Calculate Precision at K.

    Precision@k = (# relevant in top-k) / k
    """
    results = results[:k]
    relevant_count = sum(1 for r in results if r.get("label") == relevant_label)
    return relevant_count / len(results) if results else 0.0


class RAGEvaluator:
    """Evaluates RAG system performance with metrics."""

    def __init__(self, config: Dict[str, Any], baseline: bool = False):
        """
        Initialize evaluator.

        Args:
            config: Configuration dict
            baseline: If True, use baseline settings (no normalization, no chunking)
        """
        self.config = config
        self.baseline = baseline

        # Get configuration
        embedding_config = config.get("code_embedding", {})
        qdrant_config = config.get("qdrant", {})

        if baseline:
            logger.info("Initializing BASELINE RAG system...")
            # Baseline: no normalization, no chunking, cls pooling
            self.store = CodeSimilarityStore(
                host=qdrant_config.get("host", "localhost"),
                port=qdrant_config.get("port", 6333),
                collection_name="code_samples_baseline",
                embedding_model=embedding_config.get("model", "microsoft/unixcoder-base"),
                pooling_strategy="cls",
                normalize=False,
                max_length=512,
                enable_chunking=False,
                chunk_overlap=0
            )
        else:
            logger.info("Initializing ENHANCED RAG system...")
            # Enhanced: with normalization, chunking, mean pooling
            self.store = CodeSimilarityStore(
                host=qdrant_config.get("host", "localhost"),
                port=qdrant_config.get("port", 6333),
                collection_name="code_samples_enhanced",
                embedding_model=embedding_config.get("model", "microsoft/unixcoder-base"),
                pooling_strategy=embedding_config.get("pooling_strategy", "mean_pooling"),
                normalize=embedding_config.get("normalize", True),
                max_length=embedding_config.get("max_code_length", 512),
                enable_chunking=embedding_config.get("enable_chunking", True),
                chunk_overlap=embedding_config.get("chunk_overlap", 64)
            )

    def prepare_data(self, samples: List[Dict[str, Any]]):
        """Prepare data by upserting to Qdrant."""
        logger.info(f"Upserting {len(samples)} samples to collection...")
        self.store.clear_collection()
        self.store.upsert_code_samples(samples)
        logger.info("✓ Data preparation complete")

    def evaluate(
        self,
        test_queries: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system on test queries.

        Args:
            test_queries: List of dicts with:
                - code: Query code
                - expected_label: Expected label (malicious/benign)
            k_values: List of k values to evaluate

        Returns:
            Evaluation metrics
        """
        results = defaultdict(list)

        for query in test_queries:
            query_code = query["code"]
            expected_label = query["expected_label"]

            # Search
            search_results = self.store.search_similar_code(
                query_code=query_code,
                k=max(k_values),
                balance_labels=False,
                score_threshold=0.0,  # Get all results for evaluation
                aggregate_chunks=not self.baseline  # Only aggregate if enhanced
            )

            # Calculate metrics for each k
            for k in k_values:
                mrr = calculate_mrr_at_k(search_results, expected_label, k)
                recall = calculate_recall_at_k(search_results, expected_label, k)
                precision = calculate_precision_at_k(search_results, expected_label, k)

                results[f"MRR@{k}"].append(mrr)
                results[f"Recall@{k}"].append(recall)
                results[f"Precision@{k}"].append(precision)

        # Calculate averages
        metrics = {}
        for metric_name, values in results.items():
            metrics[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }

        return metrics


def create_test_set(db_manager: DatasetManager, n_per_class: int = 25) -> Tuple[List[Dict], List[Dict]]:
    """
    Create training and test sets from database.

    Args:
        db_manager: DatasetManager instance
        n_per_class: Number of samples per class for test set

    Returns:
        Tuple of (training_samples, test_queries)
    """
    logger.info(f"Creating test set ({n_per_class} per class)...")

    # Get all samples
    malicious_samples = db_manager.get_all_samples(label="malicious", limit=None)
    benign_samples = db_manager.get_all_samples(label="benign", limit=None)

    logger.info(f"Total samples: {len(malicious_samples)} malicious, {len(benign_samples)} benign")

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(malicious_samples)
    np.random.shuffle(benign_samples)

    # Split: use first n_per_class for testing, rest for training
    test_malicious = malicious_samples[:n_per_class]
    test_benign = benign_samples[:n_per_class]

    train_malicious = malicious_samples[n_per_class:]
    train_benign = benign_samples[n_per_class:]

    # Create test queries
    test_queries = []
    for sample in test_malicious:
        test_queries.append({
            "code": sample["content"],
            "expected_label": "malicious"
        })
    for sample in test_benign:
        test_queries.append({
            "code": sample["content"],
            "expected_label": "benign"
        })

    # Training samples
    training_samples = train_malicious + train_benign

    logger.info(f"✓ Test set: {len(test_queries)} queries ({n_per_class} malicious, {n_per_class} benign)")
    logger.info(f"✓ Training set: {len(training_samples)} samples")

    return training_samples, test_queries


def print_comparison(baseline_metrics: Dict, enhanced_metrics: Dict):
    """Print side-by-side comparison of metrics."""
    print("\n" + "=" * 90)
    print("BASELINE vs ENHANCED RAG - METRIC COMPARISON")
    print("=" * 90)
    print(f"{'Metric':<20} {'Baseline Mean':<15} {'Enhanced Mean':<15} {'Improvement':<20}")
    print("-" * 90)

    improvements = []

    for metric_name in sorted(baseline_metrics.keys()):
        baseline_mean = baseline_metrics[metric_name]["mean"]
        enhanced_mean = enhanced_metrics[metric_name]["mean"]

        if baseline_mean > 0:
            improvement = ((enhanced_mean - baseline_mean) / baseline_mean) * 100
        else:
            improvement = float('inf') if enhanced_mean > 0 else 0

        improvements.append(improvement)

        print(f"{metric_name:<20} {baseline_mean:<15.4f} {enhanced_mean:<15.4f} {improvement:+.2f}%")

    print("-" * 90)
    print(f"{'Average Improvement':<20} {'':<15} {'':<15} {np.mean([i for i in improvements if i != float('inf')]):+.2f}%")
    print("=" * 90)


def main():
    """Run offline evaluation comparing baseline vs enhanced RAG."""
    logger.info("Starting Offline RAG Evaluation...")

    # Load config
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize database
    db_manager = DatasetManager()

    # Create test set
    training_samples, test_queries = create_test_set(db_manager, n_per_class=25)

    if len(test_queries) < 10:
        logger.error("Not enough samples in database. Run data ingestion first.")
        return

    # Evaluate BASELINE
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING BASELINE RAG")
    logger.info("=" * 70)
    baseline_evaluator = RAGEvaluator(config, baseline=True)
    baseline_evaluator.prepare_data(training_samples)
    baseline_metrics = baseline_evaluator.evaluate(test_queries, k_values=[1, 3, 5])

    logger.info("\nBaseline Results:")
    for metric, values in baseline_metrics.items():
        logger.info(f"  {metric}: {values['mean']:.4f} (±{values['std']:.4f})")

    # Evaluate ENHANCED
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING ENHANCED RAG")
    logger.info("=" * 70)
    enhanced_evaluator = RAGEvaluator(config, baseline=False)
    enhanced_evaluator.prepare_data(training_samples)
    enhanced_metrics = enhanced_evaluator.evaluate(test_queries, k_values=[1, 3, 5])

    logger.info("\nEnhanced Results:")
    for metric, values in enhanced_metrics.items():
        logger.info(f"  {metric}: {values['mean']:.4f} (±{values['std']:.4f})")

    # Print comparison
    print_comparison(baseline_metrics, enhanced_metrics)

    # Verify stability
    logger.info("\n" + "=" * 70)
    logger.info("STABILITY CHECK")
    logger.info("=" * 70)

    for metric in ["MRR@5", "Recall@5"]:
        baseline_val = baseline_metrics[metric]["mean"]
        enhanced_val = enhanced_metrics[metric]["mean"]

        if enhanced_val >= baseline_val:
            logger.info(f"✅ {metric}: Enhanced ({enhanced_val:.4f}) >= Baseline ({baseline_val:.4f})")
        else:
            logger.warning(f"⚠️  {metric}: Enhanced ({enhanced_val:.4f}) < Baseline ({baseline_val:.4f})")

    logger.info("\n✅ Offline evaluation complete!")


if __name__ == "__main__":
    main()
