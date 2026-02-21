#!/usr/bin/env python3
"""
Phase 2: Benchmark Jina-v3 vs UniXcoder

Compare embedding quality between UniXcoder and Jina-v3 on the same test samples.

Decision Criteria:
- ✅ GO: If Jina-v3 similarity(benign-malicious) < 0.50 AND UniXcoder >= 0.65
  → Proceed to Phase 3 (End-to-End RAG test)
- ❌ NO-GO: If Jina-v3 shows similar poor separation
  → Investigate other root causes (training data, chunking, prompting)
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Import test samples
from comprehensive_test_samples import get_samples_up_to_level


def load_model(model_name, max_length=512, is_jina_v3=False):
    """Load embedding model."""
    print(f"Loading {model_name}...")

    if is_jina_v3:
        # Jina-v3 requires trust_remote_code for task adapters
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        # Set task adapter
        if hasattr(model, 'set_adapter'):
            model.set_adapter('retrieval.v2')
            print("  ✓ Task adapter set: retrieval.v2")

        # Set output dimension
        if hasattr(model, 'set_output_dim'):
            model.set_output_dim(1024)
            print("  ✓ Output dimension: 1024")

        print(f"  ✓ Model loaded on {device}")
        return None, model, device, 1024, is_jina_v3

    else:
        # Standard transformers model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, use_safetensors=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        print(f"  ✓ Model loaded on {device}")
        return tokenizer, model, device, 768, is_jina_v3


def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling with attention mask."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def normalize_embeddings(embeddings):
    """L2 normalization."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return embeddings / norms


def encode_code_samples(tokenizer, model, device, code_samples, max_length, is_jina_v3):
    """Encode code samples using the specified model."""
    if is_jina_v3:
        # Jina-v3 native encoding
        codes = [sample["code"] for sample in code_samples]

        with torch.no_grad():
            embeddings = model.encode(
                codes,
                max_length=max_length,
                task="retrieval.query",
                batch_size=8,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False
            )

        # Manual normalization
        embeddings = normalize_embeddings(embeddings)
        return embeddings

    else:
        # Standard transformers encoding
        embeddings = []

        for sample in code_samples:
            code = sample["code"]

            # Tokenize
            inputs = tokenizer(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = model(**inputs)
                # Use mean pooling
                embedding = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                embedding = embedding.cpu().numpy()
                embeddings.append(embedding[0])

        # Stack and normalize
        embeddings = np.vstack(embeddings)
        embeddings = normalize_embeddings(embeddings)

        return embeddings


def analyze_model_performance(model_name, benign_embeddings, malicious_embeddings, benign_samples, malicious_samples):
    """Analyze embedding quality for a single model."""

    print(f"\n{'='*80}")
    print(f"{model_name.upper()} EMBEDDING ANALYSIS")
    print(f"{'='*80}")

    # Calculate similarities
    benign_sim_matrix = cosine_similarity(benign_embeddings, benign_embeddings)
    malicious_sim_matrix = cosine_similarity(malicious_embeddings, malicious_embeddings)
    cross_sim_matrix = cosine_similarity(benign_embeddings, malicious_embeddings)

    # Extract values (exclude diagonal for intra-class)
    benign_sim_values = benign_sim_matrix[np.triu_indices_from(benign_sim_matrix, k=1)]
    malicious_sim_values = malicious_sim_matrix[np.triu_indices_from(malicious_sim_matrix, k=1)]
    cross_sim_values = cross_sim_matrix.flatten()

    # Calculate statistics
    benign_mean = np.mean(benign_sim_values)
    benign_std = np.std(benign_sim_values)
    malicious_mean = np.mean(malicious_sim_values)
    malicious_std = np.std(malicious_sim_values)
    cross_mean = np.mean(cross_sim_values)
    cross_std = np.std(cross_sim_values)

    print(f"\n1. INTRA-CLASS SIMILARITY (Same Label)")
    print(f"   Benign-Benign:       {benign_mean:.4f} ± {benign_std:.4f}")
    print(f"   Malicious-Malicious: {malicious_mean:.4f} ± {malicious_std:.4f}")

    print(f"\n2. INTER-CLASS SIMILARITY (Different Labels)")
    print(f"   Benign-Malicious:    {cross_mean:.4f} ± {cross_std:.4f}")

    # Separation metrics
    separation_gap = benign_mean - cross_mean
    malicious_separation = malicious_mean - cross_mean

    print(f"\n3. SEPARATION METRICS")
    print(f"   Benign Separation:    {separation_gap:.4f} (benign_intra - cross)")
    print(f"   Malicious Separation: {malicious_separation:.4f} (malicious_intra - cross)")
    print(f"   Avg Separation:       {(separation_gap + malicious_separation)/2:.4f}")

    # Quality assessment
    if cross_mean < 0.50:
        quality = "✅ EXCELLENT"
    elif cross_mean < 0.55:
        quality = "✓ GOOD"
    elif cross_mean < 0.65:
        quality = "⚠️  WEAK"
    else:
        quality = "❌ POOR"

    print(f"\n4. SEPARATION QUALITY: {quality}")

    return {
        "benign_intra": benign_mean,
        "malicious_intra": malicious_mean,
        "cross": cross_mean,
        "separation_gap": separation_gap,
        "quality": quality
    }


def compare_models(unixcoder_results, jina_results):
    """Compare the two models and make GO/NO-GO decision."""

    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    # Calculate improvement
    cross_sim_improvement = unixcoder_results["cross"] - jina_results["cross"]
    separation_improvement = jina_results["separation_gap"] - unixcoder_results["separation_gap"]

    print(f"\n1. CROSS-SIMILARITY (Benign-Malicious)")
    print(f"   UniXcoder:  {unixcoder_results['cross']:.4f}")
    print(f"   Jina-v3:    {jina_results['cross']:.4f}")
    print(f"   Improvement: {cross_sim_improvement:.4f} ({cross_sim_improvement/unixcoder_results['cross']*100:+.1f}%)")

    print(f"\n2. SEPARATION GAP (Benign Clustering - Cross)")
    print(f"   UniXcoder:  {unixcoder_results['separation_gap']:.4f}")
    print(f"   Jina-v3:    {jina_results['separation_gap']:.4f}")
    print(f"   Improvement: {separation_improvement:+.4f} ({separation_improvement/abs(unixcoder_results['separation_gap'])*100:+.1f}%)")

    # Decision logic
    print("\n" + "="*80)
    print("GO/NO-GO DECISION")
    print("="*80)

    unixcoder_poor = unixcoder_results["cross"] >= 0.65
    jina_good = jina_results["cross"] < 0.50

    print(f"\nCriteria:")
    print(f"  [{'✓' if unixcoder_poor else '✗'}] UniXcoder cross-similarity >= 0.65: {unixcoder_results['cross']:.4f}")
    print(f"  [{'✓' if jina_good else '✗'}] Jina-v3 cross-similarity < 0.50: {jina_results['cross']:.4f}")

    if unixcoder_poor and jina_good:
        print(f"\n✅ GO: Proceed to Phase 3 (End-to-End RAG Test)")
        print(f"   Jina-v3 shows significant improvement (+{cross_sim_improvement/unixcoder_results['cross']*100:.1f}% separation)")
        print(f"   Expected F1 score improvement: 71.79% → 85%+")
        print(f"\nNext steps:")
        print(f"   python scripts/test_rag_with_jina.py")
        return True

    elif jina_results["cross"] < unixcoder_results["cross"] - 0.10:
        print(f"\n⚠️  PARTIAL GO: Jina-v3 shows improvement but not meeting full criteria")
        print(f"   Improvement: {cross_sim_improvement/unixcoder_results['cross']*100:+.1f}%")
        print(f"   Consider Phase 3 testing to validate real-world performance")
        return True

    else:
        print(f"\n❌ NO-GO: Jina-v3 does not show sufficient improvement")
        print(f"   Jina-v3 cross-similarity ({jina_results['cross']:.4f}) not significantly better")
        print(f"\nInvestigate alternative root causes:")
        print(f"   - Training data domain mismatch (missing benign csv/json/database samples)")
        print(f"   - Chunk-level retrieval loss (context truncation)")
        print(f"   - Label distribution imbalance (56.2% malicious bias)")
        print(f"   - Prompt engineering (label leakage, few-shot selection)")
        return False


def main():
    """Main benchmark function."""
    print("="*80)
    print("PHASE 2: UNIXCODER VS JINA-V3 BENCHMARK")
    print("="*80)
    print("\nObjective: Test if Jina-v3 improves separation over UniXcoder")
    print("Decision: GO if Jina-v3 cross-sim < 0.50 AND UniXcoder >= 0.65\n")

    # Load test samples
    print("Loading test samples (Level 1-3)...")
    benign_samples, malicious_samples = get_samples_up_to_level(3)

    print(f"✓ Loaded {len(benign_samples)} benign samples")
    print(f"✓ Loaded {len(malicious_samples)} malicious samples")

    # Test UniXcoder
    print("\n" + "="*80)
    print("TESTING UNIXCODER")
    print("="*80)

    unixcoder_tokenizer, unixcoder_model, unixcoder_device, _, _ = load_model(
        "microsoft/unixcoder-base",
        max_length=512,
        is_jina_v3=False
    )

    print("\nGenerating UniXcoder embeddings...")
    unixcoder_benign = encode_code_samples(
        unixcoder_tokenizer, unixcoder_model, unixcoder_device,
        benign_samples, 512, False
    )
    unixcoder_malicious = encode_code_samples(
        unixcoder_tokenizer, unixcoder_model, unixcoder_device,
        malicious_samples, 512, False
    )

    unixcoder_results = analyze_model_performance(
        "UniXcoder",
        unixcoder_benign,
        unixcoder_malicious,
        benign_samples,
        malicious_samples
    )

    # Free memory
    del unixcoder_model, unixcoder_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test Jina-v3
    print("\n" + "="*80)
    print("TESTING JINA-V3")
    print("="*80)

    jina_tokenizer, jina_model, jina_device, _, _ = load_model(
        "jinaai/jina-embeddings-v3",
        max_length=8192,
        is_jina_v3=True
    )

    print("\nGenerating Jina-v3 embeddings...")
    jina_benign = encode_code_samples(
        jina_tokenizer, jina_model, jina_device,
        benign_samples, 8192, True
    )
    jina_malicious = encode_code_samples(
        jina_tokenizer, jina_model, jina_device,
        malicious_samples, 8192, True
    )

    jina_results = analyze_model_performance(
        "Jina-v3",
        jina_benign,
        jina_malicious,
        benign_samples,
        malicious_samples
    )

    # Compare and decide
    go_decision = compare_models(unixcoder_results, jina_results)

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)

    if go_decision:
        print("\n✅ Jina-v3 shows promise - proceed to Phase 3 testing")
    else:
        print("\n❌ Jina-v3 does not solve the problem - investigate alternatives")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
