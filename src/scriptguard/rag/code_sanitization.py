"""
Code Sanitization Service
Validates and cleans code samples before ingestion to prevent index pollution.

Implements AST-based validation, entropy filtering, and metadata enrichment.
"""

import ast
import re
import math
import warnings
from typing import Dict, Any, Optional, Tuple
from collections import Counter
from scriptguard.utils.logger import logger


class CodeSanitizer:
    """
    Validates and cleans code samples before embedding.

    Filters out:
    - Binary/base64 data
    - Minified/obfuscated code (low entropy)
    - Invalid syntax
    - License headers and excessive comments
    """

    def __init__(
        self,
        min_entropy: float = 3.5,
        max_line_length: int = 500,
        min_valid_lines: int = 3,
        max_empty_line_ratio: float = 0.5,
        remove_license_headers: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize code sanitizer.

        Args:
            min_entropy: Minimum Shannon entropy (bits/char) to accept
            max_line_length: Maximum characters per line (minified detection)
            min_valid_lines: Minimum non-empty lines required
            max_empty_line_ratio: Maximum ratio of empty/whitespace lines
            remove_license_headers: Remove common license/copyright headers
            strict_mode: Enable strict AST validation (slower)
        """
        self.min_entropy = min_entropy
        self.max_line_length = max_line_length
        self.min_valid_lines = min_valid_lines
        self.max_empty_line_ratio = max_empty_line_ratio
        self.remove_license_headers = remove_license_headers
        self.strict_mode = strict_mode

        # Patterns for detection
        self.base64_pattern = re.compile(
            r'[A-Za-z0-9+/]{40,}={0,2}',  # Long base64-like strings
            re.MULTILINE
        )
        self.license_patterns = [
            re.compile(r'^\s*#.*?copyright.*?\n', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*#.*?license.*?\n', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*#.*?MIT License.*?\n', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*#.*?Apache License.*?\n', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*""".*?copyright.*?"""', re.IGNORECASE | re.DOTALL),
            re.compile(r'^\s*""".*?license.*?"""', re.IGNORECASE | re.DOTALL),
        ]

        logger.info("Code Sanitizer initialized:")
        logger.info(f"  Min entropy: {min_entropy} bits/char")
        logger.info(f"  Max line length: {max_line_length}")
        logger.info(f"  Strict mode: {strict_mode}")

    def sanitize(
        self,
        content: str,
        language: str = "python",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Sanitize and validate code sample.

        Args:
            content: Raw code content
            language: Programming language (default: "python")
            metadata: Optional metadata dict to enrich

        Returns:
            Tuple of (cleaned_content, validation_report)
            - cleaned_content: Sanitized code or None if invalid
            - validation_report: Dict with validation metrics and flags
        """
        if not content or not content.strip():
            return None, {"valid": False, "reason": "empty_content"}

        report = {
            "valid": True,
            "language": language,
            "original_length": len(content),
            "original_lines": content.count('\n') + 1,
            "warnings": []
        }

        # Step 1: Check for binary/base64 data
        if self._is_binary_data(content):
            report["valid"] = False
            report["reason"] = "binary_data_detected"
            return None, report

        # Step 2: Calculate entropy
        entropy = self._calculate_entropy(content)
        report["entropy"] = entropy

        if entropy < self.min_entropy:
            report["valid"] = False
            report["reason"] = f"low_entropy_{entropy:.2f}"
            report["warnings"].append(f"Entropy {entropy:.2f} < {self.min_entropy}")
            return None, report

        # Step 3: Check for minified code (excessively long lines)
        lines = content.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        max_line_len = max(len(line) for line in lines) if lines else 0

        report["avg_line_length"] = avg_line_length
        report["max_line_length"] = max_line_len

        if max_line_len > self.max_line_length:
            report["warnings"].append(f"Very long line detected: {max_line_len} chars")
            # Don't reject, but flag as potentially minified
            report["possibly_minified"] = True

        # Step 4: Check empty line ratio
        non_empty_lines = [line for line in lines if line.strip()]
        empty_line_ratio = 1.0 - (len(non_empty_lines) / len(lines)) if lines else 0

        report["non_empty_lines"] = len(non_empty_lines)
        report["empty_line_ratio"] = empty_line_ratio

        if len(non_empty_lines) < self.min_valid_lines:
            report["valid"] = False
            report["reason"] = "too_few_valid_lines"
            return None, report

        if empty_line_ratio > self.max_empty_line_ratio:
            report["warnings"].append(f"High empty line ratio: {empty_line_ratio:.2%}")

        # Step 5: Language-specific validation
        cleaned_content = content

        if language.lower() == "python":
            is_valid, syntax_report = self._validate_python_syntax(content)
            report["syntax_valid"] = is_valid
            report["syntax_report"] = syntax_report

            if not is_valid and self.strict_mode:
                report["valid"] = False
                report["reason"] = "invalid_python_syntax"
                return None, report

            # Remove license headers if enabled
            if self.remove_license_headers:
                cleaned_content = self._remove_license_headers(content)
                if len(cleaned_content) < len(content) * 0.5:
                    # If we removed more than 50%, something is wrong
                    report["warnings"].append("Excessive header removal detected")

        # Step 6: Remove excessive empty lines (normalize)
        cleaned_content = self._normalize_whitespace(cleaned_content)

        report["cleaned_length"] = len(cleaned_content)
        report["cleaned_lines"] = cleaned_content.count('\n') + 1
        report["compression_ratio"] = len(cleaned_content) / len(content) if content else 0

        # Final check: ensure we haven't over-cleaned
        if len(cleaned_content) < 50:  # Minimum viable code length
            report["valid"] = False
            report["reason"] = "content_too_short_after_cleaning"
            return None, report

        return cleaned_content, report

    def _calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of text.

        Higher entropy = more information density.
        Binary/base64 data tends to have high entropy (>5.5).
        Natural code has moderate entropy (3.5-5.0).
        Repeated patterns have low entropy (<3.0).

        Returns:
            Entropy in bits per character
        """
        if not text:
            return 0.0

        # Count character frequencies
        counter = Counter(text)
        length = len(text)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in counter.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def _is_binary_data(self, content: str) -> bool:
        """
        Detect if content is likely binary data or base64 encoded.

        Heuristics:
        - High ratio of non-ASCII characters
        - Long base64-like strings
        - Null bytes or control characters
        """
        # Check for null bytes
        if '\x00' in content:
            return True

        # Check non-printable character ratio
        non_printable = sum(1 for c in content if ord(c) < 32 and c not in '\n\r\t')
        if non_printable / len(content) > 0.1:  # >10% non-printable
            return True

        # Check for long base64 strings (likely embedded data)
        base64_matches = self.base64_pattern.findall(content)
        if base64_matches:
            total_base64_len = sum(len(m) for m in base64_matches)
            if total_base64_len / len(content) > 0.4:  # >40% base64
                logger.debug(f"Detected {len(base64_matches)} base64 blocks "
                           f"({total_base64_len / len(content):.1%} of content)")
                return True

        return False

    def _validate_python_syntax(self, code: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate Python syntax using AST parsing.

        Returns:
            Tuple of (is_valid, report_dict)
        """
        try:
            # Suppress SyntaxWarning for invalid escape sequences in analyzed code
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=SyntaxWarning)
                ast.parse(code)
            return True, {"valid": True}
        except SyntaxError as e:
            return False, {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset
            }
        except Exception as e:
            return False, {
                "valid": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def _remove_license_headers(self, content: str) -> str:
        """
        Remove common license and copyright headers.

        Targets:
        - Multi-line comment blocks at file start
        - Repeated license text (MIT, Apache, etc.)
        """
        cleaned = content

        for pattern in self.license_patterns:
            cleaned = pattern.sub('', cleaned, count=10)  # Max 10 removals

        # Remove leading docstrings (often contain license text)
        lines = cleaned.split('\n')
        in_docstring = False
        docstring_start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = True
                    docstring_start = i
                elif stripped and not stripped.startswith('#'):
                    # Found actual code, stop
                    break
            else:
                if '"""' in line or "'''" in line:
                    # End of docstring - remove if it's at file start
                    if docstring_start == 0 or all(
                        not l.strip() or l.strip().startswith('#')
                        for l in lines[:docstring_start]
                    ):
                        # Remove docstring lines
                        lines = lines[:docstring_start] + lines[i+1:]
                    break

        return '\n'.join(lines)

    def _normalize_whitespace(self, content: str) -> str:
        """
        Normalize excessive whitespace while preserving code structure.

        Rules:
        - Maximum 2 consecutive empty lines
        - Remove trailing whitespace
        - Preserve indentation
        """
        lines = content.split('\n')
        normalized = []
        consecutive_empty = 0

        for line in lines:
            stripped = line.rstrip()  # Remove trailing whitespace

            if not stripped:
                consecutive_empty += 1
                if consecutive_empty <= 2:
                    normalized.append('')
            else:
                consecutive_empty = 0
                normalized.append(stripped)

        # Remove leading/trailing empty lines
        while normalized and not normalized[0]:
            normalized.pop(0)
        while normalized and not normalized[-1]:
            normalized.pop()

        return '\n'.join(normalized)


class ContextEnricher:
    """
    Enriches code samples with contextual metadata for better embedding quality.

    Injects structured metadata into the text before embedding to improve
    semantic understanding and retrieval accuracy.
    """

    def __init__(self, injection_format: str = "structured"):
        """
        Initialize context enricher.

        Args:
            injection_format: Format for context injection
                - "structured": Formal header with metadata
                - "inline": Inline comments
                - "minimal": Just filename
        """
        self.injection_format = injection_format
        logger.info(f"Context Enricher initialized (format: {injection_format})")

    def enrich(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enrich code content with contextual metadata.

        Args:
            content: Raw code content
            metadata: Metadata dict with optional keys:
                - file_path: Source file path
                - repository: Repository name
                - language: Programming language
                - source: Data source
                - label: Code label (malicious/benign)
                - cve_id: Associated CVE (if applicable)

        Returns:
            Enriched content with metadata header
        """
        if not metadata:
            return content

        if self.injection_format == "structured":
            return self._structured_enrichment(content, metadata)
        elif self.injection_format == "inline":
            return self._inline_enrichment(content, metadata)
        else:  # minimal
            return self._minimal_enrichment(content, metadata)

    def _structured_enrichment(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Create structured header with metadata.

        Format:
        ```
        # === CODE METADATA ===
        # File: path/to/file.py
        # Repository: owner/repo
        # Language: python
        # Source: github
        # ====================

        <actual code>
        ```
        """
        header_lines = ["# === CODE METADATA ==="]

        if metadata.get('file_path'):
            header_lines.append(f"# File: {metadata['file_path']}")

        if metadata.get('repository'):
            header_lines.append(f"# Repository: {metadata['repository']}")

        if metadata.get('language'):
            header_lines.append(f"# Language: {metadata['language']}")

        if metadata.get('source'):
            header_lines.append(f"# Source: {metadata['source']}")

        if metadata.get('label'):
            header_lines.append(f"# Classification: {metadata['label']}")

        if metadata.get('cve_id'):
            header_lines.append(f"# CVE: {metadata['cve_id']}")

        header_lines.append("# " + "=" * 20)
        header_lines.append("")  # Empty line separator

        return '\n'.join(header_lines) + '\n' + content

    def _inline_enrichment(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Add inline comment at the start of code.

        Format:
        ```
        # [file.py from owner/repo]
        <actual code>
        ```
        """
        parts = []

        if metadata.get('file_path'):
            parts.append(metadata['file_path'])

        if metadata.get('repository'):
            parts.append(f"from {metadata['repository']}")

        if parts:
            inline = f"# [{' '.join(parts)}]\n"
            return inline + content

        return content

    def _minimal_enrichment(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Minimal enrichment - just filename if available.

        Format:
        ```
        # file.py
        <actual code>
        ```
        """
        if metadata.get('file_path'):
            filename = metadata['file_path'].split('/')[-1]
            return f"# {filename}\n{content}"

        return content


# Factory functions
def create_sanitizer(config: Optional[Dict[str, Any]] = None) -> CodeSanitizer:
    """
    Create CodeSanitizer from configuration.

    Args:
        config: Configuration dict with optional keys:
            - min_entropy
            - max_line_length
            - strict_mode
            - etc.

    Returns:
        Configured CodeSanitizer instance
    """
    if not config:
        return CodeSanitizer()

    return CodeSanitizer(
        min_entropy=config.get('min_entropy', 3.5),
        max_line_length=config.get('max_line_length', 500),
        min_valid_lines=config.get('min_valid_lines', 3),
        max_empty_line_ratio=config.get('max_empty_line_ratio', 0.5),
        remove_license_headers=config.get('remove_license_headers', True),
        strict_mode=config.get('strict_mode', False)
    )


def create_enricher(config: Optional[Dict[str, Any]] = None) -> ContextEnricher:
    """
    Create ContextEnricher from configuration.

    Args:
        config: Configuration dict with 'injection_format' key

    Returns:
        Configured ContextEnricher instance
    """
    if not config:
        return ContextEnricher()

    return ContextEnricher(
        injection_format=config.get('injection_format', 'structured')
    )


__all__ = [
    'CodeSanitizer',
    'ContextEnricher',
    'create_sanitizer',
    'create_enricher'
]
