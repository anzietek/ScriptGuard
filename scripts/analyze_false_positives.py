#!/usr/bin/env python3
"""
Analyze why false positives are happening.

This script helps understand:
1. What features distinguish benign from malicious at each level
2. Why simple code like "Hello World" gets flagged
3. What the model is seeing in the training data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_test_samples import LEVEL1_BENIGN, LEVEL1_MALICIOUS, LEVEL2_BENIGN, LEVEL2_MALICIOUS
from test_hybrid_balanced import extract_features_for_sample
import json


def analyze_sample(sample, label):
    """Analyze features of a single sample."""
    code = sample['code']
    features = extract_features_for_sample(code)

    return {
        'label': label,
        'category': sample['category'],
        'description': sample.get('description', 'N/A'),
        'code_length': len(code),
        'code_lines': code.count('\n') + 1,
        'features': features
    }


def compare_features(benign_samples, malicious_samples, level_name):
    """Compare feature distributions between benign and malicious."""
    print(f"\n{'='*80}")
    print(f"FEATURE ANALYSIS: {level_name}")
    print('='*80)

    benign_analyses = [analyze_sample(s, 'benign') for s in benign_samples]
    malicious_analyses = [analyze_sample(s, 'malicious') for s in malicious_samples]

    # Calculate feature statistics
    def avg_feature(analyses, feature_name):
        values = [a['features'].get(feature_name, 0) for a in analyses]
        return sum(values) / len(values) if values else 0

    print(f"\n{'Metric':<30} {'Benign Avg':<15} {'Malicious Avg':<15} {'Difference':<15}")
    print("-" * 75)

    # Code length
    benign_len = sum(a['code_length'] for a in benign_analyses) / len(benign_analyses)
    malicious_len = sum(a['code_length'] for a in malicious_analyses) / len(malicious_analyses)
    print(f"{'Code Length':<30} {benign_len:<15.1f} {malicious_len:<15.1f} {abs(benign_len - malicious_len):<15.1f}")

    # Entropy
    benign_entropy = avg_feature(benign_analyses, 'entropy')
    malicious_entropy = avg_feature(malicious_analyses, 'entropy')
    print(f"{'Entropy':<30} {benign_entropy:<15.2f} {malicious_entropy:<15.2f} {abs(benign_entropy - malicious_entropy):<15.2f}")

    # Complexity
    benign_complexity = avg_feature(benign_analyses, 'complexity_score')
    malicious_complexity = avg_feature(malicious_analyses, 'complexity_score')
    print(f"{'Complexity Score':<30} {benign_complexity:<15.1f} {malicious_complexity:<15.1f} {abs(benign_complexity - malicious_complexity):<15.1f}")

    # Dangerous patterns count (use dangerous_api_calls key)
    benign_dangerous = sum(len(a['features'].get('dangerous_api_calls', [])) for a in benign_analyses) / len(benign_analyses)
    malicious_dangerous = sum(len(a['features'].get('dangerous_api_calls', [])) for a in malicious_analyses) / len(malicious_analyses)
    print(f"{'Dangerous API Calls':<30} {benign_dangerous:<15.2f} {malicious_dangerous:<15.2f} {abs(benign_dangerous - malicious_dangerous):<15.2f}")

    # Show specific dangerous patterns in malicious
    print(f"\nDangerous API calls in malicious code:")
    all_patterns = {}
    for a in malicious_analyses:
        for pattern in a['features'].get('dangerous_api_calls', []):
            all_patterns[pattern] = all_patterns.get(pattern, 0) + 1

    for pattern, count in sorted(all_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {pattern}: {count} occurrences")

    # Check for false positive candidates
    print(f"\n{'='*80}")
    print("POTENTIAL FALSE POSITIVE CANDIDATES (Simple Benign Code)")
    print('='*80)

    simple_benign = sorted(benign_analyses, key=lambda x: x['code_length'])[:5]
    for i, analysis in enumerate(simple_benign, 1):
        print(f"\n{i}. {analysis['category']} - {analysis['description']}")
        print(f"   Length: {analysis['code_length']} chars, Lines: {analysis['code_lines']}")
        print(f"   Entropy: {analysis['features'].get('entropy', 0):.2f}")
        print(f"   Dangerous API calls: {analysis['features'].get('dangerous_api_calls', [])}")

        # Show if it looks suspicious
        if analysis['features'].get('dangerous_api_calls'):
            print(f"   [WARN]  HAS dangerous API calls! Might confuse model!")


def find_similar_in_training():
    """Check if training data has similar simple malicious samples."""
    print(f"\n{'='*80}")
    print("CHECKING FOR SIMPLE MALICIOUS SAMPLES IN TRAINING")
    print('='*80)

    malicious_analyses = [analyze_sample(s, 'malicious') for s in LEVEL1_MALICIOUS + LEVEL2_MALICIOUS]

    # Find very short malicious samples
    short_malicious = sorted(malicious_analyses, key=lambda x: x['code_length'])[:10]

    print("\nShortest malicious samples (could confuse with simple benign):")
    for i, analysis in enumerate(short_malicious, 1):
        print(f"\n{i}. {analysis['category']}")
        print(f"   Length: {analysis['code_length']} chars, Lines: {analysis['code_lines']}")
        print(f"   Dangerous APIs: {analysis['features'].get('dangerous_api_calls', [])}")
        print(f"   Entropy: {analysis['features'].get('entropy', 0):.2f}")


def main():
    """Run false positive analysis."""
    print("="*80)
    print("FALSE POSITIVE ANALYSIS")
    print("="*80)
    print("\nAnalyzing why simple benign code gets marked as malicious...")

    # Level 1 analysis
    compare_features(LEVEL1_BENIGN, LEVEL1_MALICIOUS, "LEVEL 1 (Very Simple)")

    # Level 2 analysis
    compare_features(LEVEL2_BENIGN, LEVEL2_MALICIOUS, "LEVEL 2 (Simple)")

    # Check for similar samples
    find_similar_in_training()

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
1. ISSUE: Simple benign code (print, math) similar to simple malicious (exec, eval)
   FIX: Add MORE diverse simple benign samples (calculations, string ops, loops)

2. ISSUE: Training data may have short malicious samples confusing the model
   FIX: Ensure training data has clear distinction in features

3. ISSUE: Feature extraction may not distinguish well at low complexity
   FIX: Add more discriminative features (AST depth, function calls, imports)

4. ISSUE: Level 3 has 71% precision - too many complex benign marked as malicious
   FIX: Add MORE diverse complex benign samples (web apps, data processing)
""")


if __name__ == '__main__':
    main()
