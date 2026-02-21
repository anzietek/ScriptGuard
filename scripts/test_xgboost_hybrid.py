#!/usr/bin/env python3
"""
Test XGBoost hybrid approach: Features + Embedding similarity as inputs.

Uses production data to train a classifier that combines:
1. Static features (entropy, dangerous APIs, etc.)
2. Embedding-based similarity scores from RAG

This tests if a learned model can beat hand-crafted boosting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

from scriptguard.rag.code_similarity_store import CodeSimilarityStore
from scriptguard.steps.feature_extraction import (
    extract_ast_features, calculate_entropy,
    extract_api_patterns, extract_string_features
)
from level3_expansion import LEVEL3_BENIGN_EXPANSION, LEVEL3_MALICIOUS_EXPANSION

def extract_features_vector(code):
    """Extract numerical feature vector from code."""
    features = []

    # 1. Entropy
    entropy = calculate_entropy(code)
    features.append(entropy)

    # 2. AST features
    ast_feats = extract_ast_features(code)
    features.append(ast_feats.get("complexity_score", 0))
    features.append(len(ast_feats.get("dangerous_patterns", [])))
    features.append(len(ast_feats.get("imports", [])))
    features.append(len(ast_feats.get("function_calls", [])))

    # 3. API patterns (boolean flags)
    api_feats = extract_api_patterns(code)
    features.append(1 if api_feats.get("network_apis") else 0)
    features.append(1 if api_feats.get("file_apis") else 0)
    features.append(1 if api_feats.get("process_apis") else 0)
    features.append(1 if api_feats.get("crypto_apis") else 0)
    features.append(len(api_feats.get("suspicious_combinations", [])))

    # 4. String features
    string_feats = extract_string_features(code)
    features.append(1 if string_feats.get("has_urls") else 0)
    features.append(1 if string_feats.get("has_ips") else 0)
    features.append(1 if string_feats.get("has_base64") else 0)
    features.append(1 if string_feats.get("has_hex") else 0)
    features.append(len(string_feats.get("suspicious_strings", [])))

    # 5. Code length
    features.append(len(code))
    features.append(code.count('\n') + 1)  # line count

    return np.array(features)

def get_rag_scores(store, code, k=10):
    """Get top-k RAG similarity scores."""
    results = store.search_similar_code(
        query_code=code,
        k=k,
        filter_label=None,
        enable_feature_boosting=False  # Pure embeddings
    )

    # Extract scores and labels
    scores = [r.get('score', 0.0) for r in results]
    labels = [1 if r.get('label') == 'malicious' else 0 for r in results]

    # Pad if needed
    while len(scores) < k:
        scores.append(0.0)
        labels.append(0)

    return scores, labels

def main():
    print("="*80)
    print("XGBoost Hybrid Model Test")
    print("="*80)

    # Prepare dataset
    print("\n[1/5] Preparing dataset...")
    X_features = []
    X_rag_scores = []
    X_rag_labels = []
    y = []

    # Benign samples
    for sample in LEVEL3_BENIGN_EXPANSION:
        X_features.append(extract_features_vector(sample['code']))
        y.append(0)  # Benign

    # Malicious samples
    for sample in LEVEL3_MALICIOUS_EXPANSION:
        X_features.append(extract_features_vector(sample['code']))
        y.append(1)  # Malicious

    X_features = np.array(X_features)
    y = np.array(y)

    print(f"   Dataset: {len(y)} samples ({sum(y)} malicious, {len(y)-sum(y)} benign)")
    print(f"   Features per sample: {X_features.shape[1]}")

    # Initialize RAG
    print("\n[2/5] Initializing RAG store...")
    store = CodeSimilarityStore(
        host="localhost",
        port=6333,
        collection_name="code_samples"
    )

    # Get RAG scores for each sample
    print("\n[3/5] Computing RAG scores...")
    all_samples = list(LEVEL3_BENIGN_EXPANSION) + list(LEVEL3_MALICIOUS_EXPANSION)

    for idx, sample in enumerate(all_samples):
        scores, labels = get_rag_scores(store, sample['code'], k=10)
        X_rag_scores.append(scores)
        X_rag_labels.append(labels)

        if (idx + 1) % 10 == 0:
            print(f"   Progress: {idx+1}/{len(all_samples)}...")

    X_rag_scores = np.array(X_rag_scores)
    X_rag_labels = np.array(X_rag_labels)

    # Combine features: static features + RAG scores + RAG labels
    X_combined = np.hstack([X_features, X_rag_scores, X_rag_labels])

    print(f"\n   Combined features: {X_combined.shape[1]} dimensions")
    print(f"     - Static features: {X_features.shape[1]}")
    print(f"     - RAG scores: {X_rag_scores.shape[1]}")
    print(f"     - RAG labels: {X_rag_labels.shape[1]}")

    # Split train/test
    print("\n[4/5] Training XGBoost model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train XGBoost
    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict
    print("\n[5/5] Evaluating...")
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\nXGBoost Hybrid Model:")
    print(f"  Accuracy:  {acc:.2%}")
    print(f"  Precision: {prec:.2%}")
    print(f"  Recall:    {rec:.2%}")
    print(f"  F1 Score:  {f1:.2%}")

    # Compare to baseline (majority vote on RAG)
    print(f"\nBaseline (k=10 majority vote): 88.37% F1")
    print(f"XGBoost improvement: {(f1 - 0.8837) * 100:+.2f}%")

    # Feature importance
    print("\n" + "="*80)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*80)

    feature_names = [
        'entropy', 'complexity', 'dangerous_count', 'imports_count', 'func_calls_count',
        'has_network', 'has_file', 'has_process', 'has_crypto', 'suspicious_combo_count',
        'has_urls', 'has_ips', 'has_base64', 'has_hex', 'suspicious_str_count',
        'code_length', 'line_count'
    ] + [f'rag_score_{i}' for i in range(10)] + [f'rag_label_{i}' for i in range(10)]

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names[idx]:20s}: {importances[idx]:.4f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
