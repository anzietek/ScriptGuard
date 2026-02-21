#!/usr/bin/env python3
"""
Phase 1: Diagnose UniXcoder Bottleneck

Test if UniXcoder embeddings fail to distinguish benign utility code from malicious code.

Expected Results:
- High similarity within benign samples (good clustering)
- High similarity within malicious samples (good clustering)
- LOW similarity between benign and malicious (good separation)

If benign-malicious similarity is HIGH (>0.65), this confirms UniXcoder is the problem.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from dotenv import load_dotenv

load_dotenv()

# Import test samples
from comprehensive_test_samples import get_samples_up_to_level


def load_unixcoder_model():
    """Load UniXcoder model for embedding generation."""
    print("Loading UniXcoder model...")
    model_name = "microsoft/unixcoder-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded on {device}")
    return tokenizer, model, device


def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling with attention mask."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def normalize_embeddings(embeddings):
    """L2 normalization."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return embeddings / norms


def encode_code_samples(tokenizer, model, device, code_samples, max_length=512):
    """Encode code samples using UniXcoder."""
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


def analyze_embeddings(benign_embeddings, malicious_embeddings, benign_samples, malicious_samples):
    """Analyze embedding similarity within and between classes."""

    print("\n" + "="*80)
    print("UNIXCODER EMBEDDING ANALYSIS")
    print("="*80)

    # Calculate similarities
    benign_sim_matrix = cosine_similarity(benign_embeddings, benign_embeddings)
    malicious_sim_matrix = cosine_similarity(malicious_embeddings, malicious_embeddings)
    cross_sim_matrix = cosine_similarity(benign_embeddings, malicious_embeddings)

    # Extract upper triangle (exclude self-similarity diagonal)
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

    print(f"\n1. INTRA-CLASS SIMILARITY (Within Same Label)")
    print(f"   Benign-Benign:       {benign_mean:.4f} ± {benign_std:.4f}")
    print(f"   Malicious-Malicious: {malicious_mean:.4f} ± {malicious_std:.4f}")

    print(f"\n2. INTER-CLASS SIMILARITY (Between Different Labels)")
    print(f"   Benign-Malicious:    {cross_mean:.4f} ± {cross_std:.4f}")

    # Calculate separation metric
    separation_quality = benign_mean - cross_mean

    print(f"\n3. SEPARATION QUALITY")
    print(f"   Benign clustering - Cross similarity: {separation_quality:.4f}")

    if cross_mean >= 0.65:
        print(f"\n   [FAIL] POOR SEPARATION: Cross-similarity ({cross_mean:.4f}) >= 0.65")
        print(f"      UniXcoder CANNOT distinguish benign from malicious code!")
        print(f"      This confirms the bottleneck hypothesis.")
        return False
    elif cross_mean >= 0.55:
        print(f"\n   [WARNING] WEAK SEPARATION: Cross-similarity ({cross_mean:.4f}) >= 0.55")
        print(f"      UniXcoder has difficulty separating benign from malicious.")
        print(f"      Jina-v3 upgrade recommended.")
        return False
    else:
        print(f"\n   [OK] GOOD SEPARATION: Cross-similarity ({cross_mean:.4f}) < 0.55")
        print(f"      UniXcoder can somewhat distinguish the classes.")
        print(f"      Problem may be elsewhere (training data, prompting).")
        return True

    # Find most confusing pairs (benign samples similar to malicious)
    print(f"\n4. MOST CONFUSING BENIGN SAMPLES (High Similarity to Malicious)")
    confusion_threshold = 0.75

    confusing_pairs = []
    for i, benign_sample in enumerate(benign_samples):
        for j, malicious_sample in enumerate(malicious_samples):
            sim = cross_sim_matrix[i, j]
            if sim >= confusion_threshold:
                confusing_pairs.append((i, j, sim, benign_sample, malicious_sample))

    confusing_pairs.sort(key=lambda x: x[2], reverse=True)

    if confusing_pairs:
        print(f"\n   Found {len(confusing_pairs)} highly confusing pairs (similarity >= {confusion_threshold}):")
        for i, (benign_idx, mal_idx, sim, benign_sample, mal_sample) in enumerate(confusing_pairs[:5]):
            print(f"\n   [{i+1}] Similarity: {sim:.4f}")
            print(f"       Benign:    {benign_sample.get('category', 'unknown')} - {benign_sample.get('description', '')}")
            print(f"       Malicious: {mal_sample.get('category', 'unknown')} - {mal_sample.get('description', '')}")
            print(f"       Benign code:    {benign_sample['code'][:100]}...")
            print(f"       Malicious code: {mal_sample['code'][:100]}...")
    else:
        print(f"\n   No highly confusing pairs found (all similarities < {confusion_threshold})")

    return True


def main():
    """Main diagnostic function."""
    print("="*80)
    print("PHASE 1: UNIXCODER BOTTLENECK DIAGNOSIS")
    print("="*80)
    print("\nObjective: Test if UniXcoder embeddings fail to separate benign from malicious")
    print("Expected: If cross-similarity >= 0.65, UniXcoder is the bottleneck\n")

    # Load test samples from Level 3 (includes failing benign categories)
    print("Loading test samples (Level 1-3: csv, json, database, logging, etc.)...")
    benign_samples, malicious_samples = get_samples_up_to_level(3)

    print(f"[OK] Loaded {len(benign_samples)} benign samples")
    print(f"[OK] Loaded {len(malicious_samples)} malicious samples")

    # Count categories
    benign_categories = {}
    for sample in benign_samples:
        cat = sample.get('category', 'unknown')
        benign_categories[cat] = benign_categories.get(cat, 0) + 1

    print(f"\nBenign categories: {list(benign_categories.keys())}")

    # Load model
    tokenizer, model, device = load_unixcoder_model()

    # Generate embeddings
    print("\nGenerating UniXcoder embeddings...")
    benign_embeddings = encode_code_samples(tokenizer, model, device, benign_samples)
    malicious_embeddings = encode_code_samples(tokenizer, model, device, malicious_samples)

    print(f"[OK] Benign embeddings: {benign_embeddings.shape}")
    print(f"[OK] Malicious embeddings: {malicious_embeddings.shape}")

    # Analyze
    separation_ok = analyze_embeddings(
        benign_embeddings,
        malicious_embeddings,
        benign_samples,
        malicious_samples
    )

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if not separation_ok:
        print("\n[CONFIRMED] HYPOTHESIS CONFIRMED: UniXcoder is the bottleneck!")
        print("   -> Proceed to Phase 2: Benchmark Jina-v3")
        print("\nNext steps:")
        print("   python scripts/benchmark_jina_vs_unixcoder.py")
    else:
        print("\n[REJECTED] HYPOTHESIS REJECTED: UniXcoder separation is acceptable")
        print("   -> Problem may be elsewhere:")
        print("     - Training data domain mismatch")
        print("     - Prompt engineering issues")
        print("     - RAG retrieval configuration")
        print("     - Label distribution imbalance")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
