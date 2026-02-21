#!/usr/bin/env python3
"""
FAIR XGBoost Test: Train on Qdrant production data, test on Level 3 samples.

Training set: Samples FROM Qdrant (different from test)
Test set: Level 3 expansion (40 samples NOT in database)

This ensures no data leakage and fair comparison with baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from qdrant_client import QdrantClient
import random

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

    # 5. Code metrics
    features.append(len(code))
    features.append(code.count('\n') + 1)

    return np.array(features)

def get_rag_features(store, code, k=10):
    """Get RAG-based features (scores and labels of top-k neighbors)."""
    results = store.search_similar_code(
        query_code=code,
        k=k,
        filter_label=None,
        enable_feature_boosting=False
    )

    # Extract scores and labels
    scores = [r.get('score', 0.0) for r in results]
    labels = [1 if r.get('label') == 'malicious' else 0 for r in results]

    # Pad if needed
    while len(scores) < k:
        scores.append(0.0)
        labels.append(0)

    return scores, labels

def sample_training_data(client, collection_name, n_samples=500):
    """Sample training data from Qdrant (parent docs only)."""
    print(f"Sampling {n_samples} training samples from Qdrant...")

    samples_by_label = {'malicious': [], 'benign': []}
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            offset=offset,
            scroll_filter={
                "must": [
                    {"key": "chunk_index", "match": {"value": 0}}
                ]
            }
        )

        points, next_offset = result
        if not points:
            break

        for point in points:
            label = point.payload.get("label")
            if label in ["malicious", "benign"]:
                code = point.payload.get("code_preview", "")
                if code and len(code) > 50:  # Skip very short samples
                    samples_by_label[label].append(code)

        offset = next_offset
        if offset is None:
            break

        # Early stop if we have enough
        if len(samples_by_label['malicious']) >= n_samples and len(samples_by_label['benign']) >= n_samples:
            break

    # Balance and sample
    n_per_label = min(n_samples // 2, len(samples_by_label['malicious']), len(samples_by_label['benign']))

    train_samples = (
        random.sample(samples_by_label['malicious'], n_per_label) +
        random.sample(samples_by_label['benign'], n_per_label)
    )
    train_labels = [1] * n_per_label + [0] * n_per_label

    print(f"  Sampled {len(train_samples)} samples ({n_per_label} per label)")

    return train_samples, train_labels

def main():
    print("="*80)
    print("XGBoost Production Test (No Data Leakage)")
    print("="*80)

    # Connect to Qdrant
    import os
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(host="localhost", port=6333, api_key=api_key, https=False)
    collection_name = "code_samples"

    # Initialize RAG store
    print("\n[1/6] Initializing RAG store...")
    store = CodeSimilarityStore(
        host="localhost",
        port=6333,
        collection_name=collection_name
    )

    # Sample TRAINING data from Qdrant (different from test set)
    print("\n[2/6] Sampling training data from Qdrant...")
    train_codes, train_labels = sample_training_data(client, collection_name, n_samples=500)

    # Extract features for training data
    print("\n[3/6] Extracting features for training data...")
    X_train_static = []
    X_train_rag_scores = []
    X_train_rag_labels = []

    for idx, code in enumerate(train_codes):
        # Static features
        X_train_static.append(extract_features_vector(code))

        # RAG features
        scores, labels = get_rag_features(store, code, k=10)
        X_train_rag_scores.append(scores)
        X_train_rag_labels.append(labels)

        if (idx + 1) % 50 == 0:
            print(f"   Progress: {idx+1}/{len(train_codes)}...")

    X_train_static = np.array(X_train_static)
    X_train_rag_scores = np.array(X_train_rag_scores)
    X_train_rag_labels = np.array(X_train_rag_labels)
    X_train = np.hstack([X_train_static, X_train_rag_scores, X_train_rag_labels])
    y_train = np.array(train_labels)

    print(f"\n   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Prepare TEST data (Level 3 expansion - NOT in database)
    print("\n[4/6] Preparing test data (Level 3 expansion)...")
    test_samples = []
    test_labels = []

    for sample in LEVEL3_BENIGN_EXPANSION:
        test_samples.append(sample['code'])
        test_labels.append(0)

    for sample in LEVEL3_MALICIOUS_EXPANSION:
        test_samples.append(sample['code'])
        test_labels.append(1)

    print(f"   Test set: {len(test_samples)} samples (NOT in database)")

    # Extract features for test data
    print("\n[5/6] Extracting features for test data...")
    X_test_static = []
    X_test_rag_scores = []
    X_test_rag_labels = []

    for idx, code in enumerate(test_samples):
        X_test_static.append(extract_features_vector(code))
        scores, labels = get_rag_features(store, code, k=10)
        X_test_rag_scores.append(scores)
        X_test_rag_labels.append(labels)

        if (idx + 1) % 10 == 0:
            print(f"   Progress: {idx+1}/{len(test_samples)}...")

    X_test_static = np.array(X_test_static)
    X_test_rag_scores = np.array(X_test_rag_scores)
    X_test_rag_labels = np.array(X_test_rag_labels)
    X_test = np.hstack([X_test_static, X_test_rag_scores, X_test_rag_labels])
    y_test = np.array(test_labels)

    # Train XGBoost
    print("\n[6/6] Training and evaluating XGBoost...")
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict on TEST set (40 Level 3 samples)
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    tp = sum((y_test == 1) & (y_pred == 1))
    tn = sum((y_test == 0) & (y_pred == 0))
    fp = sum((y_test == 0) & (y_pred == 1))
    fn = sum((y_test == 1) & (y_pred == 0))

    print("\n" + "="*80)
    print("RESULTS (Fair Test - No Data Leakage)")
    print("="*80)

    print(f"\nXGBoost Hybrid (trained on {len(train_labels)} Qdrant samples):")
    print(f"  Accuracy:  {acc:.2%}")
    print(f"  Precision: {prec:.2%}")
    print(f"  Recall:    {rec:.2%}")
    print(f"  F1 Score:  {f1:.2%}")

    print(f"\nConfusion Matrix (on 40 test samples):")
    print(f"  True Positives:  {tp:3d}")
    print(f"  True Negatives:  {tn:3d}")
    print(f"  False Positives: {fp:3d}")
    print(f"  False Negatives: {fn:3d}")

    print(f"\nBaseline (k=10 majority vote, same 40 test samples):")
    print(f"  Accuracy:  87.50%")
    print(f"  F1 Score:  88.37%")

    improvement = (f1 - 0.8837) * 100
    print(f"\nXGBoost vs Baseline: {improvement:+.2f}%")

    if improvement > 0:
        print("✅ XGBoost BEATS baseline!")
    else:
        print("❌ XGBoost does NOT beat baseline")

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
