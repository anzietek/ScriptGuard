"""
Early data quality filtering for data sources.
Rejects binary/garbage data BEFORE it enters the pipeline.
"""

import re
from typing import Optional, Tuple
from scriptguard.utils.logger import logger


def is_valid_source_code(content: str, file_extension: Optional[str] = None) -> Tuple[bool, str]:
    """
    Early validation to reject binary/garbage data at data source level.

    This is a FAST check done before expensive operations like sanitization,
    embedding, or database insertion.

    Args:
        content: Raw file content
        file_extension: Optional file extension (e.g., ".py", ".ps1")

    Returns:
        Tuple of (is_valid, rejection_reason)
        - is_valid: True if content appears to be source code
        - rejection_reason: Empty string if valid, otherwise describes why rejected
    """

    # 1. Empty check
    if not content or not content.strip():
        return False, "empty_content"

    # 2. Minimum length (10 bytes)
    if len(content) < 10:
        return False, "too_short"

    # 3. Null byte check (binary file indicator)
    if '\x00' in content:
        return False, "null_bytes_detected"

    # 4. Non-printable character ratio
    # Allow up to 5% non-printable (stricter than sanitization's 10%)
    non_printable = sum(1 for c in content if ord(c) < 32 and c not in '\n\r\t')
    non_printable_ratio = non_printable / len(content)

    if non_printable_ratio > 0.05:
        return False, f"too_many_non_printable_{non_printable_ratio:.1%}"

    # 5. Check for executable signatures (PE, ELF, Mach-O)
    # These are magic bytes indicating compiled binaries
    if content.startswith(b'MZ'.decode('latin-1', errors='ignore')):  # Windows PE
        return False, "windows_executable"
    if content.startswith(b'\x7fELF'.decode('latin-1', errors='ignore')):  # Linux ELF
        return False, "linux_executable"
    if content.startswith(b'\xfe\xed\xfa'.decode('latin-1', errors='ignore')):  # Mach-O
        return False, "macos_executable"
    if content.startswith(b'PK\x03\x04'.decode('latin-1', errors='ignore')):  # ZIP/JAR
        return False, "compressed_archive"

    # 6. Check for image/media files
    if content.startswith(b'\x89PNG'.decode('latin-1', errors='ignore')):
        return False, "png_image"
    if content.startswith(b'\xff\xd8\xff'.decode('latin-1', errors='ignore')):
        return False, "jpeg_image"
    if content.startswith(b'GIF'.decode('latin-1', errors='ignore')):
        return False, "gif_image"

    # 7. Extension-based validation (if provided)
    if file_extension:
        valid_extensions = {'.py', '.ps1', '.js', '.vbs', '.sh', '.bat', '.cmd', '.rb', '.pl'}
        if file_extension.lower() not in valid_extensions:
            return False, f"invalid_extension_{file_extension}"

    # 8. Check for excessive base64 (>60% of content)
    # More aggressive than sanitization's 40%
    base64_pattern = re.compile(r'[A-Za-z0-9+/]{40,}={0,2}')
    base64_matches = base64_pattern.findall(content)

    if base64_matches:
        total_base64_len = sum(len(m) for m in base64_matches)
        base64_ratio = total_base64_len / len(content)

        if base64_ratio > 0.60:
            return False, f"excessive_base64_{base64_ratio:.1%}"

    # 9. Minimum printable ASCII ratio
    # Source code should be mostly ASCII printable characters
    printable_ascii = sum(1 for c in content if 32 <= ord(c) < 127 or c in '\n\r\t')
    ascii_ratio = printable_ascii / len(content)

    if ascii_ratio < 0.5:  # At least 50% should be printable ASCII
        return False, f"low_ascii_ratio_{ascii_ratio:.1%}"

    # 10. Check for obvious code patterns (positive indicators)
    # If we see these, it's likely legitimate code even if it fails some checks above
    code_indicators = [
        # Python
        r'\bimport\s+\w+',
        r'\bdef\s+\w+\s*\(',
        r'\bclass\s+\w+',
        r'\bif\s+\w+\s*:',
        r'\bfor\s+\w+\s+in\s+',
        # PowerShell
        r'\$\w+\s*=',
        r'\bFunction\s+\w+',
        r'\bParam\s*\(',
        # JavaScript
        r'\bfunction\s+\w+\s*\(',
        r'\bconst\s+\w+\s*=',
        r'\bvar\s+\w+\s*=',
        # Shell script
        r'#!/bin/(bash|sh)',
        r'\becho\s+',
    ]

    has_code_pattern = any(re.search(pattern, content, re.IGNORECASE) for pattern in code_indicators)

    if has_code_pattern:
        # Override some earlier failures if we detect clear code patterns
        # This allows obfuscated but valid code to pass
        logger.debug(f"Content has code patterns, allowing despite marginal quality metrics")
        return True, ""

    # All checks passed
    return True, ""


def quick_binary_check(content: bytes) -> bool:
    """
    Ultra-fast binary detection for filtering before decoding.

    Use this when you have raw bytes (e.g., from HTTP response)
    before converting to string.

    Args:
        content: Raw bytes content

    Returns:
        True if content appears to be binary (should be rejected)
    """
    if not content or len(content) < 10:
        return True

    # Check first 512 bytes for null bytes (binary indicator)
    sample = content[:512]
    if b'\x00' in sample:
        return True

    # Check magic bytes for common binary formats
    magic_bytes = [
        b'MZ',           # Windows PE
        b'\x7fELF',      # Linux ELF
        b'\xfe\xed\xfa', # Mach-O
        b'PK\x03\x04',   # ZIP
        b'\x89PNG',      # PNG
        b'\xff\xd8\xff', # JPEG
        b'GIF8',         # GIF
        b'%PDF',         # PDF
    ]

    for magic in magic_bytes:
        if content.startswith(magic):
            return True

    return False


def log_rejection_stats(total_fetched: int, rejected_counts: dict):
    """
    Log statistics about rejected samples for debugging.

    Args:
        total_fetched: Total number of samples fetched
        rejected_counts: Dict mapping rejection_reason -> count
    """
    if not rejected_counts:
        logger.info(f"✓ All {total_fetched} samples passed early quality filter")
        return

    total_rejected = sum(rejected_counts.values())
    rejection_rate = (total_rejected / total_fetched * 100) if total_fetched > 0 else 0

    logger.warning(f"Early quality filter rejected {total_rejected}/{total_fetched} samples ({rejection_rate:.1f}%)")

    # Show top rejection reasons
    sorted_reasons = sorted(rejected_counts.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons[:5]:  # Top 5 reasons
        logger.warning(f"  - {reason}: {count} samples")
