"""
Data Validation Step
Validates code syntax, filters invalid samples, and performs quality checks.
"""

import ast
import warnings
import re
from scriptguard.utils.logger import logger
from scriptguard.schemas import CodeSample, validate_data_batch, ValidatedCodeSample
from typing import Dict, List
from zenml import step

def validate_python_syntax(code: str, allow_python2: bool = True) -> bool:
    """
    Validate Python code syntax using AST parsing.
    Supports both Python 3 and optionally Python 2 syntax patterns.

    Args:
        code: Python code string
        allow_python2: If True, accept Python 2 syntax patterns

    Returns:
        True if valid syntax, False otherwise
    """
    # First, try Python 3 syntax validation
    try:
        # Suppress SyntaxWarning for invalid escape sequences in analyzed code
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=SyntaxWarning)
            ast.parse(code)
        return True
    except SyntaxError:
        # If Python 3 fails and Python 2 support is enabled, check for Python 2 patterns
        if not allow_python2:
            return False

        # Detect common Python 2 syntax patterns
        python2_patterns = [
            re.compile(r'\bprint\s+[^(]'),  # print without parentheses
            re.compile(r'except\s+\w+\s*,\s*\w+:'),  # except E, e:
            re.compile(r'<>'),  # != operator in Python 2
            re.compile(r'\bexec\s+'),  # exec statement (not function)
        ]

        for pattern in python2_patterns:
            if pattern.search(code):
                logger.debug("Python 2 syntax detected - accepting sample")
                return True

        return False
    except Exception:
        return False

def check_code_length(code: str, min_length: int = 50, max_length: int = 50000) -> bool:
    """
    Check if code length is within acceptable range.

    Args:
        code: Code string
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length

    Returns:
        True if length is acceptable
    """
    length = len(code)
    return min_length <= length <= max_length

def check_encoding(code: str) -> bool:
    """
    Check if code is valid UTF-8.

    Args:
        code: Code string

    Returns:
        True if valid encoding
    """
    try:
        code.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False

def verify_label(label: str) -> bool:
    """
    Verify label is valid.

    Args:
        label: Label string

    Returns:
        True if label is "malicious" or "benign"
    """
    return label.lower() in ["malicious", "benign"]

def is_mostly_comments(code: str, threshold: float = 0.8) -> bool:
    """
    Check if code is mostly comments (low quality).

    Args:
        code: Code string
        threshold: Fraction of lines that are comments to reject

    Returns:
        True if mostly comments
    """
    lines = code.split("\n")
    if not lines:
        return True

    comment_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            comment_lines += 1

    ratio = comment_lines / len(lines)
    return ratio > threshold

def has_minimum_code_content(code: str, min_non_whitespace: int = 100) -> bool:
    """
    Check if code has minimum non-whitespace content.

    Args:
        code: Code string
        min_non_whitespace: Minimum non-whitespace characters

    Returns:
        True if meets minimum
    """
    non_whitespace = len("".join(code.split()))
    return non_whitespace >= min_non_whitespace

@step
def validate_samples(
    data: List[Dict],
    config: Dict = None,
    validate_syntax: bool = True,
    min_length: int = 50,
    max_length: int = 50000,
    skip_syntax_errors: bool = True
) -> List[Dict]:
    """
    Validate and filter code samples using Pydantic schemas.

    Args:
        data: List of code sample dictionaries
        config: Configuration dictionary from config.yaml
        validate_syntax: Whether to validate Python syntax
        min_length: Minimum code length
        max_length: Maximum code length
        skip_syntax_errors: Whether to skip samples with syntax errors

    Returns:
        Filtered list of valid samples

    Raises:
        ValueError: If data schema validation fails critically
    """
    logger.info(f"Validating {len(data)} samples...")

    # First, validate data schema
    try:
        validated_schemas = validate_data_batch(data, CodeSample)
        logger.info(f"Schema validation passed for {len(validated_schemas)} samples")
    except ValueError as e:
        logger.warning(f"Schema validation encountered errors: {e}")
        # Continue with samples that passed validation
        validated_schemas = []
        for item in data:
            try:
                validated_schemas.append(CodeSample(**item))
            except Exception:
                pass
        logger.info(f"Proceeding with {len(validated_schemas)} schema-valid samples")

    valid_samples = []
    stats = {
        "total": len(validated_schemas),
        "invalid_syntax": 0,
        "invalid_length": 0,
        "invalid_encoding": 0,
        "invalid_label": 0,
        "mostly_comments": 0,
        "insufficient_content": 0,
        "valid": 0
    }

    for schema_sample in validated_schemas:
        # Convert Pydantic model back to dict for compatibility
        sample = schema_sample.model_dump()
        content = sample.get("content", "")
        label = sample.get("label", "")

        # Check label
        if not verify_label(label):
            stats["invalid_label"] += 1
            logger.debug(f"Invalid label: {label}")
            continue

        # Check encoding
        if not check_encoding(content):
            stats["invalid_encoding"] += 1
            logger.debug("Invalid encoding")
            continue

        # Check length
        if not check_code_length(content, min_length, max_length):
            stats["invalid_length"] += 1
            logger.debug(f"Invalid length: {len(content)}")
            continue

        # Check if mostly comments
        if is_mostly_comments(content):
            stats["mostly_comments"] += 1
            logger.debug("Sample is mostly comments")
            continue

        # Check minimum content
        if not has_minimum_code_content(content):
            stats["insufficient_content"] += 1
            logger.debug("Insufficient code content")
            continue

        # Validate syntax (skip validation for malicious samples - they're often obfuscated)
        if validate_syntax:
            # Get allow_python2 from config (default: True for historical malware)
            validation_config = config.get("validation", {}) if config else {}
            allow_python2 = validation_config.get("allow_python2", True)

            if not validate_python_syntax(content, allow_python2=allow_python2):
                stats["invalid_syntax"] += 1

                # Allow malicious samples with syntax errors (obfuscation is common)
                if label == "malicious" and skip_syntax_errors:
                    logger.debug(f"Malicious sample has syntax errors (likely obfuscated) - keeping anyway")
                    sample["validation_warning"] = "syntax_error_malicious"
                    # KEEP THE SAMPLE (don't continue)
                elif skip_syntax_errors:
                    # Skip benign samples with syntax errors
                    preview = content[:100].replace('\n', ' ')
                    logger.debug(f"Syntax error - skipping: {preview}... (label: {label}, source: {sample.get('source', 'unknown')})")
                    continue
                else:
                    # Keep sample but add warning
                    sample["validation_warning"] = "syntax_error"

        # Sample is valid
        stats["valid"] += 1
        valid_samples.append(sample)

    # Log statistics
    logger.info("=" * 60)
    logger.info("VALIDATION STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total samples: {stats['total']}")
    logger.info(f"Valid samples: {stats['valid']}")
    logger.info(f"Invalid syntax: {stats['invalid_syntax']}")
    logger.info(f"Invalid length: {stats['invalid_length']}")
    logger.info(f"Invalid encoding: {stats['invalid_encoding']}")
    logger.info(f"Invalid label: {stats['invalid_label']}")
    logger.info(f"Mostly comments: {stats['mostly_comments']}")
    logger.info(f"Insufficient content: {stats['insufficient_content']}")
    logger.info(f"Validation pass rate: {(stats['valid'] / stats['total'] * 100):.1f}%")
    logger.info("=" * 60)

    # Apply deduplication if enabled in config
    validation_config = config.get("validation", {}) if config else {}

    if validation_config.get("deduplicate", True):
        from scriptguard.database.deduplication import deduplicate_samples

        threshold = validation_config.get("dedup_threshold", 0.85)
        enable_exact = validation_config.get("dedup_exact_first", True)
        method = validation_config.get("dedup_method", "auto")

        # Jaccard fallback settings
        batch_size = validation_config.get("dedup_batch_size", 1000)
        max_memory_mb = validation_config.get("dedup_max_memory_mb", 500)

        logger.info(
            f"Applying two-stage deduplication "
            f"(method={method}, threshold={threshold})"
        )

        valid_samples = deduplicate_samples(
            samples=valid_samples,
            threshold=threshold,
            enable_exact=enable_exact,
            enable_fuzzy=True,
            method=method,
            batch_size=batch_size,
            max_memory_mb=max_memory_mb
        )

    return valid_samples

@step
def filter_by_quality(
    data: List[Dict],
    min_code_lines: int = 5,
    max_comment_ratio: float = 0.5
) -> List[Dict]:
    """
    Filter samples by code quality metrics.

    Args:
        data: List of code samples
        min_code_lines: Minimum number of code lines
        max_comment_ratio: Maximum ratio of comment lines

    Returns:
        Filtered samples
    """
    logger.info(f"Filtering {len(data)} samples by quality...")

    filtered_samples = []

    for sample in data:
        content = sample.get("content", "")
        lines = content.split("\n")

        # Count code vs comment lines
        code_lines = 0
        comment_lines = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            elif stripped.startswith("#"):
                comment_lines += 1
            else:
                code_lines += 1

        # Check minimum code lines
        if code_lines < min_code_lines:
            continue

        # Check comment ratio
        total_lines = code_lines + comment_lines
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            if comment_ratio > max_comment_ratio:
                continue

        filtered_samples.append(sample)

    logger.info(f"Quality filtering: {len(data)} -> {len(filtered_samples)} samples")

    return filtered_samples
