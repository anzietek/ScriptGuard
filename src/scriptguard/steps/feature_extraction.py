"""
Feature Extraction Step
AST-based features, entropy analysis, and API call pattern extraction.
"""

import ast
import math
from scriptguard.utils.logger import logger
from typing import Dict, List, Set
from collections import Counter
from zenml import step

def extract_ast_features(code: str) -> Dict:
    """
    Extract AST-based features from code.

    Args:
        code: Python code string

    Returns:
        Dictionary of AST features
    """
    features = {
        "function_calls": [],
        "imports": [],
        "dangerous_patterns": [],
        "ast_node_counts": {},
        "complexity_score": 0
    }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return features

    # Dangerous function calls
    dangerous_funcs = {
        'eval', 'exec', 'compile', '__import__',
        'system', 'popen', 'spawn', 'call',
        'loads', 'load', 'decode'
    }

    # Count different node types
    node_counts = Counter()

    for node in ast.walk(tree):
        node_type = type(node).__name__
        node_counts[node_type] += 1

        # Extract function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                features["function_calls"].append(func_name)

                if func_name in dangerous_funcs:
                    features["dangerous_patterns"].append(func_name)

            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                features["function_calls"].append(attr_name)

                if attr_name in dangerous_funcs:
                    features["dangerous_patterns"].append(attr_name)

        # Extract imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                features["imports"].append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                features["imports"].append(f"{module}.{alias.name}")

    features["ast_node_counts"] = dict(node_counts)

    # Calculate complexity score (based on node diversity)
    features["complexity_score"] = len(node_counts)

    return features

def calculate_entropy(code: str) -> float:
    """
    Calculate Shannon entropy of code.

    Args:
        code: Code string

    Returns:
        Entropy value
    """
    if not code:
        return 0.0

    # Count character frequencies
    freq = Counter(code)
    length = len(code)

    # Calculate entropy
    entropy = 0.0
    for count in freq.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy

def extract_api_patterns(code: str) -> Dict:
    """
    Extract API call patterns and suspicious combinations.

    Args:
        code: Python code string

    Returns:
        Dictionary of API patterns
    """
    patterns = {
        "network_apis": [],
        "file_apis": [],
        "process_apis": [],
        "crypto_apis": [],
        "suspicious_combinations": []
    }

    # Define API categories
    network_keywords = ['socket', 'requests', 'urllib', 'http', 'ftp', 'smtp']
    file_keywords = ['open', 'read', 'write', 'file', 'path', 'mkdir', 'rmdir']
    process_keywords = ['subprocess', 'os.system', 'popen', 'exec', 'eval', 'spawn']
    crypto_keywords = ['base64', 'hashlib', 'crypt', 'encode', 'decode', 'encrypt']

    code_lower = code.lower()

    # Check for each category
    for keyword in network_keywords:
        if keyword in code_lower:
            patterns["network_apis"].append(keyword)

    for keyword in file_keywords:
        if keyword in code_lower:
            patterns["file_apis"].append(keyword)

    for keyword in process_keywords:
        if keyword in code_lower:
            patterns["process_apis"].append(keyword)

    for keyword in crypto_keywords:
        if keyword in code_lower:
            patterns["crypto_apis"].append(keyword)

    # Detect suspicious combinations
    if patterns["network_apis"] and patterns["process_apis"]:
        patterns["suspicious_combinations"].append("network_and_process")

    if patterns["crypto_apis"] and patterns["network_apis"]:
        patterns["suspicious_combinations"].append("crypto_and_network")

    if patterns["file_apis"] and patterns["network_apis"]:
        patterns["suspicious_combinations"].append("file_and_network")

    if 'eval' in code_lower and ('input' in code_lower or 'recv' in code_lower):
        patterns["suspicious_combinations"].append("eval_with_input")

    return patterns

def extract_string_features(code: str) -> Dict:
    """
    Extract features from string literals.

    Args:
        code: Python code string

    Returns:
        Dictionary of string features
    """
    features = {
        "has_urls": False,
        "has_ips": False,
        "has_base64": False,
        "has_hex": False,
        "suspicious_strings": []
    }

    # URL pattern
    import re
    if re.search(r'https?://', code):
        features["has_urls"] = True

    # IP address pattern
    if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', code):
        features["has_ips"] = True

    # Base64-like strings (long alphanumeric)
    if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', code):
        features["has_base64"] = True

    # Hex strings
    if re.search(r'\\x[0-9a-fA-F]{2}', code) or re.search(r'0x[0-9a-fA-F]+', code):
        features["has_hex"] = True

    # Suspicious keywords
    suspicious_keywords = [
        'password', 'passwd', 'token', 'key', 'secret',
        'backdoor', 'shell', 'payload', 'exploit',
        'ransomware', 'keylog', 'steal'
    ]

    code_lower = code.lower()
    for keyword in suspicious_keywords:
        if keyword in code_lower:
            features["suspicious_strings"].append(keyword)

    return features

@step
def extract_features(data: List[Dict]) -> List[Dict]:
    """
    Extract features from code samples.

    Args:
        data: List of code samples

    Returns:
        Samples with extracted features added
    """
    logger.info(f"Extracting features from {len(data)} samples...")

    enriched_samples = []

    for sample in data:
        code = sample.get("content", "")

        # Extract all features
        ast_features = extract_ast_features(code)
        entropy = calculate_entropy(code)
        api_patterns = extract_api_patterns(code)
        string_features = extract_string_features(code)

        # Add features to sample
        enriched = {
            **sample,
            "features": {
                "ast": ast_features,
                "entropy": entropy,
                "api_patterns": api_patterns,
                "strings": string_features,
                "length": len(code),
                "lines": code.count("\n") + 1
            }
        }

        enriched_samples.append(enriched)

    logger.info(f"Feature extraction completed for {len(enriched_samples)} samples")

    return enriched_samples

@step
def analyze_feature_importance(data: List[Dict]) -> Dict:
    """
    Analyze which features are most indicative of malicious code.

    Args:
        data: List of samples with extracted features

    Returns:
        Feature importance analysis
    """
    logger.info("Analyzing feature importance...")

    malicious = [s for s in data if s.get("label") == "malicious"]
    benign = [s for s in data if s.get("label") == "benign"]

    analysis = {
        "malicious_stats": {},
        "benign_stats": {},
        "discriminative_features": []
    }

    # Analyze entropy
    mal_entropies = [s.get("features", {}).get("entropy", 0) for s in malicious]
    ben_entropies = [s.get("features", {}).get("entropy", 0) for s in benign]

    if mal_entropies and ben_entropies:
        analysis["malicious_stats"]["avg_entropy"] = sum(mal_entropies) / len(mal_entropies)
        analysis["benign_stats"]["avg_entropy"] = sum(ben_entropies) / len(ben_entropies)

    # Analyze dangerous patterns
    mal_dangerous = [len(s.get("features", {}).get("ast", {}).get("dangerous_patterns", [])) for s in malicious]
    ben_dangerous = [len(s.get("features", {}).get("ast", {}).get("dangerous_patterns", [])) for s in benign]

    if mal_dangerous:
        analysis["malicious_stats"]["avg_dangerous_patterns"] = sum(mal_dangerous) / len(mal_dangerous)
    if ben_dangerous:
        analysis["benign_stats"]["avg_dangerous_patterns"] = sum(ben_dangerous) / len(ben_dangerous)

    # Analyze suspicious combinations
    mal_suspicious = sum(1 for s in malicious if s.get("features", {}).get("api_patterns", {}).get("suspicious_combinations"))
    ben_suspicious = sum(1 for s in benign if s.get("features", {}).get("api_patterns", {}).get("suspicious_combinations"))

    analysis["malicious_stats"]["pct_with_suspicious_combinations"] = (mal_suspicious / len(malicious) * 100) if malicious else 0
    analysis["benign_stats"]["pct_with_suspicious_combinations"] = (ben_suspicious / len(benign) * 100) if benign else 0

    logger.info("Feature importance analysis:")
    logger.info(f"  Malicious avg entropy: {analysis['malicious_stats'].get('avg_entropy', 0):.2f}")
    logger.info(f"  Benign avg entropy: {analysis['benign_stats'].get('avg_entropy', 0):.2f}")
    logger.info(f"  Malicious avg dangerous patterns: {analysis['malicious_stats'].get('avg_dangerous_patterns', 0):.2f}")
    logger.info(f"  Benign avg dangerous patterns: {analysis['benign_stats'].get('avg_dangerous_patterns', 0):.2f}")

    return analysis
