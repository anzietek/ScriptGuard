#!/usr/bin/env python3
"""
Benchmark embedding models for RAG system.

Compares UniXcoder-base vs Jina-embeddings-v3 on:
- Encoding speed
- Similarity score distribution
- Retrieval quality (Precision@K, Recall@K, NDCG@K)
- Memory usage

Usage:
    python scripts/benchmark_embeddings.py --samples 1000 --output benchmark_report.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import psutil
import torch
from dataclasses import dataclass, asdict
from collections import defaultdict

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scriptguard.database.postgres_manager import PostgresManager
from scriptguard.utils.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Metrics for a single model."""
    model_name: str
    embedding_dim: int

    # Performance metrics
    avg_encode_time_ms: float
    samples_per_second: float

    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float

    # Retrieval metrics (averaged across queries)
    precision_at_3: float
    recall_at_3: float
    ndcg_at_3: float

    # Score distribution
    similarity_score_mean: float
    similarity_score_std: float
    similarity_score_min: float
    similarity_score_max: float

    # Detailed results
    detailed_results: Dict[str, Any] = None


class EmbeddingBenchmark:
    """Benchmark different embedding models."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.db_manager = PostgresManager(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_samples(self, n_samples: int = 1000) -> Tuple[List[Dict], List[Dict]]:
        """Load balanced sample of code from PostgreSQL.

        Returns:
            malicious_samples, benign_samples (each n_samples/2)
        """
        logger.info(f"Loading {n_samples} samples from PostgreSQL...")

        with self.db_manager.get_connection() as conn:
            # Load malicious samples
            malicious_query = """
                SELECT id, content, label, source
                FROM code_samples
                WHERE label = 'malicious'
                AND content IS NOT NULL
                AND LENGTH(content) > 100
                ORDER BY RANDOM()
                LIMIT %s
            """
            malicious = conn.execute(malicious_query, (n_samples // 2,)).fetchall()

            # Load benign samples
            benign_query = """
                SELECT id, content, label, source
                FROM code_samples
                WHERE label = 'benign'
                AND content IS NOT NULL
                AND LENGTH(content) > 100
                ORDER BY RANDOM()
                LIMIT %s
            """
            benign = conn.execute(benign_query, (n_samples // 2,)).fetchall()

        malicious_samples = [
            {"id": r[0], "content": r[1], "label": r[2], "source": r[3]}
            for r in malicious
        ]
        benign_samples = [
            {"id": r[0], "content": r[1], "label": r[2], "source": r[3]}
            for r in benign
        ]

        logger.info(f"Loaded {len(malicious_samples)} malicious, {len(benign_samples)} benign samples")
        return malicious_samples, benign_samples

    def init_unixcoder(self) -> Tuple[Any, Any]:
        """Initialize UniXcoder model."""
        from transformers import AutoTokenizer, AutoModel

        logger.info("Loading UniXcoder model...")
        model_name = "microsoft/unixcoder-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(self.device)
        model.eval()

        return tokenizer, model

    def init_jina_v3(self, task_adapter: str = "retrieval.v2", output_dim: int = 768) -> Tuple[Any, Any]:
        """Initialize Jina-embeddings-v3 model.

        Args:
            task_adapter: One of "retrieval.v2", "classification.v2", or None (base)
            output_dim: Output dimension (512, 768, or 1024 for Matryoshka)
        """
        from transformers import AutoModel

        logger.info(f"Loading Jina-v3 model (adapter={task_adapter}, dim={output_dim})...")
        model_name = "jinaai/jina-embeddings-v3"

        # Jina-v3 doesn't use traditional tokenizer - uses internal processing
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(self.device)
        model.eval()

        # Configure task adapter and output dimension
        if hasattr(model, 'set_adapter'):
            if task_adapter:
                model.set_adapter(task_adapter)
                logger.info(f"Set task adapter: {task_adapter}")

        if hasattr(model, 'set_output_dim'):
            model.set_output_dim(output_dim)
            logger.info(f"Set output dimension: {output_dim}")

        return None, model  # Jina handles tokenization internally

    def encode_unixcoder(self, texts: List[str], tokenizer, model, batch_size: int = 8) -> np.ndarray:
        """Encode texts with UniXcoder."""
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Tokenize
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Encode
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def encode_jina_v3(self, texts: List[str], model, batch_size: int = 8) -> np.ndarray:
        """Encode texts with Jina-v3."""
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Jina-v3 encode method handles tokenization internally
                batch_embeddings = model.encode(
                    batch,
                    max_length=8192,  # Jina-v3 supports up to 8192 tokens
                    task="retrieval.query",  # Use retrieval task
                    device=self.device
                )

                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def benchmark_encoding_speed(
        self,
        samples: List[Dict],
        encode_fn,
        *model_args,
        batch_size: int = 8,
        n_runs: int = 3
    ) -> Dict[str, float]:
        """Benchmark encoding speed."""
        texts = [s["content"] for s in samples]

        # Warmup
        _ = encode_fn(texts[:batch_size], *model_args, batch_size=batch_size)

        # Benchmark
        times = []
        memory_usage = []

        for run in range(n_runs):
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            start_time = time.time()
            _ = encode_fn(texts, *model_args, batch_size=batch_size)
            end_time = time.time()

            end_memory = process.memory_info().rss / 1024 / 1024  # MB

            elapsed = (end_time - start_time) * 1000  # ms
            times.append(elapsed)
            memory_usage.append(end_memory - start_memory)

        avg_time_ms = np.mean(times)
        samples_per_sec = len(samples) / (avg_time_ms / 1000)

        return {
            "avg_encode_time_ms": avg_time_ms,
            "samples_per_second": samples_per_sec,
            "peak_memory_mb": np.max(memory_usage),
            "avg_memory_mb": np.mean(memory_usage)
        }

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix."""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        # Compute similarity matrix
        similarity = normalized @ normalized.T

        return similarity

    def evaluate_retrieval_quality(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
        query_samples: List[Dict],
        corpus_samples: List[Dict],
        k: int = 3
    ) -> Dict[str, float]:
        """Evaluate retrieval quality using label relevance.

        For each query:
        - Retrieve top-k most similar corpus samples
        - Relevant = same label as query
        - Compute Precision@K, Recall@K, NDCG@K
        """
        # Compute query-corpus similarity
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        corpus_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-8)
        similarity = query_norm @ corpus_norm.T  # (n_queries, n_corpus)

        precisions = []
        recalls = []
        ndcgs = []
        all_scores = []

        for i, query_sample in enumerate(query_samples):
            query_label = query_sample["label"]

            # Get top-k corpus indices
            scores = similarity[i]
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_scores = scores[top_k_indices]

            all_scores.extend(top_k_scores.tolist())

            # Compute relevance
            relevant_retrieved = sum(
                1 for idx in top_k_indices
                if corpus_samples[idx]["label"] == query_label
            )
            total_relevant = sum(
                1 for cs in corpus_samples
                if cs["label"] == query_label
            )

            # Precision@K
            precision = relevant_retrieved / k if k > 0 else 0.0
            precisions.append(precision)

            # Recall@K
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
            recalls.append(recall)

            # NDCG@K
            dcg = sum(
                (1 if corpus_samples[idx]["label"] == query_label else 0) / np.log2(rank + 2)
                for rank, idx in enumerate(top_k_indices)
            )
            idcg = sum(1 / np.log2(rank + 2) for rank in range(min(k, total_relevant)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        return {
            "precision_at_k": np.mean(precisions),
            "recall_at_k": np.mean(recalls),
            "ndcg_at_k": np.mean(ndcgs),
            "score_distribution": {
                "mean": np.mean(all_scores),
                "std": np.std(all_scores),
                "min": np.min(all_scores),
                "max": np.max(all_scores)
            }
        }

    def benchmark_model(
        self,
        model_name: str,
        samples: List[Dict],
        init_fn,
        encode_fn,
        output_dim: int,
        **init_kwargs
    ) -> BenchmarkMetrics:
        """Benchmark a single model configuration."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {model_name} (dim={output_dim})")
        logger.info(f"{'='*60}")

        # Initialize model
        model_args = init_fn(**init_kwargs)

        # Split samples for retrieval evaluation
        # Use 200 as queries, rest as corpus
        n_queries = min(200, len(samples) // 5)
        query_samples = samples[:n_queries]
        corpus_samples = samples[n_queries:]

        logger.info(f"Queries: {n_queries}, Corpus: {len(corpus_samples)}")

        # Benchmark encoding speed
        logger.info("Benchmarking encoding speed...")
        speed_metrics = self.benchmark_encoding_speed(
            samples, encode_fn, *model_args, batch_size=8, n_runs=3
        )

        logger.info(f"  Avg time: {speed_metrics['avg_encode_time_ms']:.2f}ms")
        logger.info(f"  Throughput: {speed_metrics['samples_per_second']:.2f} samples/sec")
        logger.info(f"  Memory: {speed_metrics['peak_memory_mb']:.2f}MB")

        # Encode all samples for retrieval evaluation
        logger.info("Encoding samples for retrieval evaluation...")
        all_texts = [s["content"] for s in samples]
        all_embeddings = encode_fn(all_texts, *model_args, batch_size=8)

        query_embeddings = all_embeddings[:n_queries]
        corpus_embeddings = all_embeddings[n_queries:]

        # Evaluate retrieval quality
        logger.info("Evaluating retrieval quality...")
        retrieval_metrics = self.evaluate_retrieval_quality(
            query_embeddings, corpus_embeddings,
            query_samples, corpus_samples, k=3
        )

        logger.info(f"  Precision@3: {retrieval_metrics['precision_at_k']:.4f}")
        logger.info(f"  Recall@3: {retrieval_metrics['recall_at_k']:.4f}")
        logger.info(f"  NDCG@3: {retrieval_metrics['ndcg_at_k']:.4f}")

        score_dist = retrieval_metrics['score_distribution']
        logger.info(f"  Score: {score_dist['mean']:.4f} ± {score_dist['std']:.4f} "
                   f"[{score_dist['min']:.4f}, {score_dist['max']:.4f}]")

        # Clean up
        del model_args
        torch.cuda.empty_cache()

        return BenchmarkMetrics(
            model_name=model_name,
            embedding_dim=output_dim,
            avg_encode_time_ms=speed_metrics['avg_encode_time_ms'],
            samples_per_second=speed_metrics['samples_per_second'],
            peak_memory_mb=speed_metrics['peak_memory_mb'],
            avg_memory_mb=speed_metrics['avg_memory_mb'],
            precision_at_3=retrieval_metrics['precision_at_k'],
            recall_at_3=retrieval_metrics['recall_at_k'],
            ndcg_at_3=retrieval_metrics['ndcg_at_k'],
            similarity_score_mean=score_dist['mean'],
            similarity_score_std=score_dist['std'],
            similarity_score_min=score_dist['min'],
            similarity_score_max=score_dist['max']
        )

    def run_full_benchmark(self, n_samples: int = 1000) -> Dict[str, BenchmarkMetrics]:
        """Run comprehensive benchmark of all models."""
        # Load samples
        malicious_samples, benign_samples = self.load_samples(n_samples)
        all_samples = malicious_samples + benign_samples

        results = {}

        # Benchmark 1: UniXcoder (baseline)
        results['unixcoder_768d'] = self.benchmark_model(
            model_name="microsoft/unixcoder-base",
            samples=all_samples,
            init_fn=self.init_unixcoder,
            encode_fn=self.encode_unixcoder,
            output_dim=768
        )

        # Benchmark 2: Jina-v3 @ 768d (backward compatible)
        results['jina_v3_768d_retrieval'] = self.benchmark_model(
            model_name="jinaai/jina-embeddings-v3 (retrieval.v2)",
            samples=all_samples,
            init_fn=self.init_jina_v3,
            encode_fn=self.encode_jina_v3,
            output_dim=768,
            task_adapter="retrieval.v2"
        )

        # Benchmark 3: Jina-v3 @ 1024d (full capacity)
        results['jina_v3_1024d_retrieval'] = self.benchmark_model(
            model_name="jinaai/jina-embeddings-v3 (retrieval.v2)",
            samples=all_samples,
            init_fn=self.init_jina_v3,
            encode_fn=self.encode_jina_v3,
            output_dim=1024,
            task_adapter="retrieval.v2"
        )

        # Optional: Benchmark 4: Jina-v3 @ 768d with classification adapter
        results['jina_v3_768d_classification'] = self.benchmark_model(
            model_name="jinaai/jina-embeddings-v3 (classification.v2)",
            samples=all_samples,
            init_fn=self.init_jina_v3,
            encode_fn=self.encode_jina_v3,
            output_dim=768,
            task_adapter="classification.v2"
        )

        return results

    def generate_report(self, results: Dict[str, BenchmarkMetrics], output_path: str):
        """Generate benchmark report."""
        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK SUMMARY")
        logger.info(f"{'='*60}\n")

        # Convert to dict for JSON serialization
        results_dict = {k: asdict(v) for k, v in results.items()}

        # Print comparison table
        print(f"{'Model':<40} {'P@3':<8} {'R@3':<8} {'NDCG@3':<8} {'Speed (s/s)':<12} {'Memory (MB)':<12}")
        print("-" * 100)

        for model_name, metrics in results.items():
            print(f"{model_name:<40} "
                  f"{metrics.precision_at_3:<8.4f} "
                  f"{metrics.recall_at_3:<8.4f} "
                  f"{metrics.ndcg_at_3:<8.4f} "
                  f"{metrics.samples_per_second:<12.2f} "
                  f"{metrics.peak_memory_mb:<12.2f}")

        # GO/NO-GO decision
        baseline = results['unixcoder_768d']
        jina_768d = results['jina_v3_768d_retrieval']

        print(f"\n{'='*60}")
        print("GO/NO-GO DECISION CRITERIA")
        print(f"{'='*60}\n")

        # Compare metrics
        precision_delta = (jina_768d.precision_at_3 - baseline.precision_at_3) / baseline.precision_at_3 * 100
        recall_delta = (jina_768d.recall_at_3 - baseline.recall_at_3) / baseline.recall_at_3 * 100
        ndcg_delta = (jina_768d.ndcg_at_3 - baseline.ndcg_at_3) / baseline.ndcg_at_3 * 100
        speed_delta = (jina_768d.samples_per_second - baseline.samples_per_second) / baseline.samples_per_second * 100

        print(f"Precision@3 Change: {precision_delta:+.2f}%")
        print(f"Recall@3 Change: {recall_delta:+.2f}%")
        print(f"NDCG@3 Change: {ndcg_delta:+.2f}%")
        print(f"Speed Change: {speed_delta:+.2f}%")

        # Decision logic
        retrieval_improved = (precision_delta >= 0 or recall_delta >= 0 or ndcg_delta >= 0)
        speed_acceptable = speed_delta > -20  # Not more than 20% slower

        decision = "✅ GO" if (retrieval_improved and speed_acceptable) else "❌ NO-GO"

        print(f"\n{'='*60}")
        print(f"RECOMMENDATION: {decision}")
        print(f"{'='*60}\n")

        if decision == "✅ GO":
            print("Jina-v3 shows improvement or comparable performance.")
            print("Proceed with Phase 1B: Configuration Migration")
        else:
            print("Jina-v3 does not meet performance criteria.")
            print("Reasons:")
            if not retrieval_improved:
                print("  - No improvement in retrieval metrics")
            if not speed_acceptable:
                print(f"  - Speed degradation > 20% ({speed_delta:.2f}%)")

        # Save to file
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results_dict,
            "comparison": {
                "precision_delta_pct": precision_delta,
                "recall_delta_pct": recall_delta,
                "ndcg_delta_pct": ndcg_delta,
                "speed_delta_pct": speed_delta
            },
            "decision": decision,
            "recommendation": "Proceed with Phase 1B" if decision == "✅ GO" else "Do not migrate to Jina-v3"
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"\nBenchmark report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models for RAG")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to benchmark")
    parser.add_argument("--output", type=str, default="benchmark_report.json", help="Output report path")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")

    args = parser.parse_args()

    benchmark = EmbeddingBenchmark(config_path=args.config)
    results = benchmark.run_full_benchmark(n_samples=args.samples)
    benchmark.generate_report(results, output_path=args.output)


if __name__ == "__main__":
    main()
