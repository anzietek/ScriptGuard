"""
Calibration Tool for Score Thresholds
Determines optimal score thresholds for Few-Shot RAG retrieval on validation set.
"""

import os
import sys
import argparse
import yaml
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scriptguard.rag.code_similarity_store import CodeSimilarityStore
from src.scriptguard.database.dataset_manager import DatasetManager
from scriptguard.utils.logger import logger


class ThresholdCalibrator:
    """
    Calibrates score thresholds for optimal retrieval quality.

    Methodology:
    1. For each query in validation set, retrieve top-k candidates
    2. Compute relevance: 1 if retrieved sample has same label as query, 0 otherwise
    3. Find threshold that maximizes F1 score (balance of precision and recall)
    4. Generate report with recommended thresholds per model
    """

    def __init__(
        self,
        store: CodeSimilarityStore,
        db_manager: DatasetManager,
        validation_size: int = 100
    ):
        """
        Initialize calibrator.

        Args:
            store: CodeSimilarityStore instance
            db_manager: DatasetManager for fetching validation samples
            validation_size: Number of validation queries to use
        """
        self.store = store
        self.db_manager = db_manager
        self.validation_size = validation_size

    def calibrate(
        self,
        k: int = 5,
        threshold_range: Tuple[float, float] = (0.0, 0.9),
        num_thresholds: int = 50
    ) -> Dict[str, Any]:
        """
        Run calibration and find optimal threshold.

        Args:
            k: Number of results to retrieve per query
            threshold_range: (min, max) range of thresholds to test
            num_thresholds: Number of threshold values to test

        Returns:
            Calibration results with optimal thresholds
        """
        logger.info("=" * 70)
        logger.info("THRESHOLD CALIBRATION")
        logger.info("=" * 70)
        logger.info(f"Model: {self.store.embedding_model}")
        logger.info(f"Validation samples: {self.validation_size}")
        logger.info(f"k: {k}")
        logger.info(f"Threshold range: {threshold_range}")
        logger.info("")

        # Step 1: Get validation samples
        logger.info("Fetching validation samples...")
        validation_samples = self._get_validation_samples()

        if not validation_samples:
            logger.error("No validation samples found!")
            return {}

        logger.info(f"✓ Loaded {len(validation_samples)} validation samples")

        # Step 2: Collect scores and relevance for each query
        logger.info("Running retrieval evaluation...")
        all_scores = []
        all_relevances = []

        for sample in tqdm(validation_samples, desc="Evaluating queries"):
            query_code = sample["content"]
            query_label = sample["label"].lower()

            try:
                # Search without threshold (get all results with scores)
                results = self.store.search_similar_code(
                    query_code=query_code,
                    k=k * 3,  # Get more candidates
                    balance_labels=False,
                    score_threshold=0.0,  # No threshold
                    enable_reranking=False  # Disable for calibration
                )

                for result in results:
                    score = result["score"]
                    retrieved_label = result["label"].lower()

                    # Relevance: 1 if labels match, 0 otherwise
                    relevance = 1 if retrieved_label == query_label else 0

                    all_scores.append(score)
                    all_relevances.append(relevance)

            except Exception as e:
                logger.warning(f"Failed to evaluate query: {e}")
                continue

        if not all_scores:
            logger.error("No scores collected. Cannot calibrate.")
            return {}

        all_scores = np.array(all_scores)
        all_relevances = np.array(all_relevances)

        logger.info(f"✓ Collected {len(all_scores)} (score, relevance) pairs")
        logger.info(f"  Relevant: {all_relevances.sum()} ({100*all_relevances.mean():.1f}%)")
        logger.info(f"  Score range: [{all_scores.min():.3f}, {all_scores.max():.3f}]")
        logger.info("")

        # Step 3: Test different thresholds
        logger.info("Testing thresholds...")
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            # Filter by threshold
            mask = all_scores >= threshold

            if mask.sum() == 0:
                # No results above threshold
                precisions.append(0.0)
                recalls.append(0.0)
                f1_scores.append(0.0)
                continue

            filtered_relevances = all_relevances[mask]

            # Metrics
            precision = filtered_relevances.mean()  # TP / (TP + FP)
            recall = filtered_relevances.sum() / all_relevances.sum()  # TP / (TP + FN)

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1_scores = np.array(f1_scores)

        # Step 4: Find optimal thresholds
        # Default: maximize F1
        f1_max_idx = np.argmax(f1_scores)
        optimal_default = float(thresholds[f1_max_idx])

        # Strict: high precision (e.g., precision >= 0.8, maximize F1)
        high_precision_mask = precisions >= 0.8
        if high_precision_mask.any():
            strict_idx = np.argmax(f1_scores * high_precision_mask)
            optimal_strict = float(thresholds[strict_idx])
        else:
            # Fallback: top 10% by precision
            strict_idx = np.argsort(precisions)[-max(1, len(precisions) // 10)]
            optimal_strict = float(thresholds[strict_idx])

        # Lenient: high recall (e.g., recall >= 0.8, maximize F1)
        high_recall_mask = recalls >= 0.8
        if high_recall_mask.any():
            lenient_idx = np.argmax(f1_scores * high_recall_mask)
            optimal_lenient = float(thresholds[lenient_idx])
        else:
            # Fallback: top 10% by recall
            lenient_idx = np.argsort(recalls)[-max(1, len(recalls) // 10)]
            optimal_lenient = float(thresholds[lenient_idx])

        # Step 5: Generate report
        results = {
            "model": self.store.embedding_model,
            "validation_samples": len(validation_samples),
            "total_pairs": len(all_scores),
            "optimal_thresholds": {
                "default": optimal_default,
                "strict": optimal_strict,
                "lenient": optimal_lenient
            },
            "metrics_at_optimal": {
                "default": {
                    "threshold": optimal_default,
                    "precision": float(precisions[f1_max_idx]),
                    "recall": float(recalls[f1_max_idx]),
                    "f1": float(f1_scores[f1_max_idx])
                },
                "strict": {
                    "threshold": optimal_strict,
                    "precision": float(precisions[strict_idx]),
                    "recall": float(recalls[strict_idx]),
                    "f1": float(f1_scores[strict_idx])
                },
                "lenient": {
                    "threshold": optimal_lenient,
                    "precision": float(precisions[lenient_idx]),
                    "recall": float(recalls[lenient_idx]),
                    "f1": float(f1_scores[lenient_idx])
                }
            },
            "calibration_curve": {
                "thresholds": thresholds.tolist(),
                "precisions": precisions.tolist(),
                "recalls": recalls.tolist(),
                "f1_scores": f1_scores.tolist()
            }
        }

        # Print report
        self._print_report(results)

        # Plot calibration curve
        self._plot_calibration_curve(results)

        return results

    def _get_validation_samples(self) -> List[Dict[str, Any]]:
        """Get validation samples from database."""
        try:
            # Get balanced samples (50% malicious, 50% benign)
            malicious = self.db_manager.get_all_samples(
                label="malicious",
                limit=self.validation_size // 2
            )
            benign = self.db_manager.get_all_samples(
                label="benign",
                limit=self.validation_size // 2
            )

            samples = malicious + benign

            # Shuffle
            import random
            random.shuffle(samples)

            return samples

        except Exception as e:
            logger.error(f"Failed to fetch validation samples: {e}")
            return []

    def _print_report(self, results: Dict[str, Any]):
        """Print calibration report."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("CALIBRATION RESULTS")
        logger.info("=" * 70)
        logger.info("")
        logger.info(f"Model: {results['model']}")
        logger.info(f"Validation pairs: {results['total_pairs']}")
        logger.info("")
        logger.info("RECOMMENDED THRESHOLDS:")
        logger.info("-" * 70)

        for mode in ["default", "strict", "lenient"]:
            threshold = results["optimal_thresholds"][mode]
            metrics = results["metrics_at_optimal"][mode]

            logger.info(f"\n{mode.upper()}:")
            logger.info(f"  Threshold: {threshold:.3f}")
            logger.info(f"  Precision: {metrics['precision']:.3f}")
            logger.info(f"  Recall:    {metrics['recall']:.3f}")
            logger.info(f"  F1 Score:  {metrics['f1']:.3f}")

        logger.info("")
        logger.info("=" * 70)
        logger.info("")
        logger.info("To update config.yaml, add these values under:")
        logger.info(f"  code_embedding.score_thresholds.\"{results['model']}\":")
        logger.info(f"    default: {results['optimal_thresholds']['default']:.3f}")
        logger.info(f"    strict:  {results['optimal_thresholds']['strict']:.3f}")
        logger.info(f"    lenient: {results['optimal_thresholds']['lenient']:.3f}")
        logger.info("")

    def _plot_calibration_curve(self, results: Dict[str, Any]):
        """Plot precision, recall, and F1 vs threshold."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use('Agg')  # Non-interactive backend

            thresholds = results["calibration_curve"]["thresholds"]
            precisions = results["calibration_curve"]["precisions"]
            recalls = results["calibration_curve"]["recalls"]
            f1_scores = results["calibration_curve"]["f1_scores"]

            plt.figure(figsize=(12, 6))

            plt.plot(thresholds, precisions, label="Precision", linewidth=2)
            plt.plot(thresholds, recalls, label="Recall", linewidth=2)
            plt.plot(thresholds, f1_scores, label="F1 Score", linewidth=2, linestyle='--')

            # Mark optimal points
            for mode, color in [("default", "red"), ("strict", "green"), ("lenient", "blue")]:
                threshold = results["optimal_thresholds"][mode]
                metrics = results["metrics_at_optimal"][mode]
                plt.axvline(threshold, color=color, linestyle=':', alpha=0.7, label=f"{mode.capitalize()}: {threshold:.3f}")

            plt.xlabel("Score Threshold")
            plt.ylabel("Metric Value")
            plt.title(f"Threshold Calibration Curve\nModel: {results['model']}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            output_path = f"threshold_calibration_{results['model'].replace('/', '_')}.png"
            plt.savefig(output_path, dpi=150)
            logger.info(f"✓ Calibration curve saved to: {output_path}")

        except Exception as e:
            logger.warning(f"Failed to plot calibration curve: {e}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate score thresholds for RAG retrieval")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--validation-size", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve per query")
    parser.add_argument("--output", default="calibration_results.yaml", help="Output file for results")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Initialize components
    logger.info("Initializing Code Similarity Store...")
    store = CodeSimilarityStore(config_path=args.config)

    logger.info("Connecting to database...")
    db_manager = DatasetManager()

    # Run calibration
    calibrator = ThresholdCalibrator(
        store=store,
        db_manager=db_manager,
        validation_size=args.validation_size
    )

    results = calibrator.calibrate(k=args.k)

    # Save results
    if results:
        with open(args.output, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False)
        logger.info(f"✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
