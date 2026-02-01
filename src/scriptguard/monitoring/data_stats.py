"""
Dataset Statistics
Tools for analyzing dataset quality, balance, and source distribution.
"""

import logging
from typing import Dict, List
from collections import Counter

logger = logging.getLogger(__name__)


class DatasetStatistics:
    """Comprehensive dataset statistics analyzer."""

    def __init__(self, samples: List[Dict]):
        """
        Initialize statistics analyzer.

        Args:
            samples: List of sample dictionaries
        """
        self.samples = samples
        self.total = len(samples)

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels."""
        labels = [s.get("label", "unknown") for s in self.samples]
        return dict(Counter(labels))

    def get_source_distribution(self) -> Dict[str, int]:
        """Get distribution of sources."""
        sources = [s.get("source", "unknown") for s in self.samples]
        return dict(Counter(sources))

    def get_content_length_stats(self) -> Dict[str, float]:
        """Get statistics about content lengths."""
        lengths = [len(s.get("content", "")) for s in self.samples]

        if not lengths:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0
            }

        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)

        return {
            "min": sorted_lengths[0],
            "max": sorted_lengths[-1],
            "mean": sum(lengths) / n,
            "median": sorted_lengths[n // 2] if n % 2 else (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2
        }

    def get_balance_ratio(self) -> float:
        """
        Calculate class balance ratio.

        Returns:
            Ratio of malicious to benign samples
        """
        label_dist = self.get_label_distribution()
        malicious = label_dist.get("malicious", 0)
        benign = label_dist.get("benign", 0)

        if benign == 0:
            return float("inf") if malicious > 0 else 0

        return malicious / benign

    def get_full_report(self) -> Dict:
        """
        Get comprehensive statistics report.

        Returns:
            Dictionary with all statistics
        """
        label_dist = self.get_label_distribution()
        source_dist = self.get_source_distribution()
        length_stats = self.get_content_length_stats()
        balance = self.get_balance_ratio()

        return {
            "total_samples": self.total,
            "label_distribution": label_dist,
            "source_distribution": source_dist,
            "content_length_stats": length_stats,
            "balance_ratio": balance,
            "balance_quality": self._evaluate_balance(balance)
        }

    def _evaluate_balance(self, ratio: float) -> str:
        """
        Evaluate balance quality.

        Args:
            ratio: Balance ratio

        Returns:
            Quality description
        """
        if ratio == 0 or ratio == float("inf"):
            return "CRITICAL - Only one class present"
        elif 0.8 <= ratio <= 1.25:
            return "EXCELLENT - Well balanced"
        elif 0.5 <= ratio <= 2.0:
            return "GOOD - Acceptable balance"
        elif 0.25 <= ratio <= 4.0:
            return "FAIR - Consider rebalancing"
        else:
            return "POOR - Severe imbalance"

    def print_report(self):
        """Print formatted statistics report."""
        report = self.get_full_report()

        logger.info("=" * 60)
        logger.info("DATASET STATISTICS REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Samples: {report['total_samples']}")
        logger.info("")

        logger.info("Label Distribution:")
        for label, count in report['label_distribution'].items():
            percentage = (count / report['total_samples']) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        logger.info("")

        logger.info("Source Distribution:")
        for source, count in report['source_distribution'].items():
            percentage = (count / report['total_samples']) * 100
            logger.info(f"  {source}: {count} ({percentage:.1f}%)")
        logger.info("")

        logger.info("Content Length Statistics:")
        for stat, value in report['content_length_stats'].items():
            logger.info(f"  {stat}: {value:.0f} characters")
        logger.info("")

        logger.info(f"Balance Ratio: {report['balance_ratio']:.2f}")
        logger.info(f"Balance Quality: {report['balance_quality']}")
        logger.info("=" * 60)


def check_balance(samples: List[Dict], target_ratio: float = 1.0, tolerance: float = 0.2) -> bool:
    """
    Check if dataset is balanced within tolerance.

    Args:
        samples: List of sample dictionaries
        target_ratio: Target balance ratio (default 1.0 for equal)
        tolerance: Acceptable deviation from target (default 0.2)

    Returns:
        True if balanced within tolerance
    """
    stats = DatasetStatistics(samples)
    ratio = stats.get_balance_ratio()

    if ratio == 0 or ratio == float("inf"):
        logger.warning("Dataset has only one class!")
        return False

    deviation = abs(ratio - target_ratio)

    if deviation <= tolerance:
        logger.info(f"Dataset is balanced (ratio: {ratio:.2f}, target: {target_ratio:.2f})")
        return True
    else:
        logger.warning(f"Dataset is imbalanced (ratio: {ratio:.2f}, target: {target_ratio:.2f})")
        return False


def analyze_sources(samples: List[Dict]) -> Dict[str, Dict]:
    """
    Analyze samples by source.

    Args:
        samples: List of sample dictionaries

    Returns:
        Dictionary with per-source statistics
    """
    source_samples = {}

    for sample in samples:
        source = sample.get("source", "unknown")
        if source not in source_samples:
            source_samples[source] = []
        source_samples[source].append(sample)

    source_stats = {}

    for source, source_samples_list in source_samples.items():
        stats = DatasetStatistics(source_samples_list)
        source_stats[source] = {
            "total": len(source_samples_list),
            "label_distribution": stats.get_label_distribution(),
            "avg_length": stats.get_content_length_stats()["mean"]
        }

    return source_stats
