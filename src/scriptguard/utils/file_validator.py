"""
File Validation Module - Strict Python Gatekeeper
Enforces strict validation rules for Python files before ingestion.
"""

import os
import ast
import warnings
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from scriptguard.utils.logger import logger

# Constants
MAX_PY_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_CODE_DENSITY = 0.1  # Minimum ratio of code to comments/whitespace
ALLOWED_EXTENSION = ".py"


class FileValidationError(Exception):
    """Custom exception for file validation errors."""
    pass


class PythonFileValidator:
    """
    Strict validator for Python source files.

    Validation Rules:
    1. Extension must be .py
    2. File size must not exceed MAX_PY_FILE_SIZE
    3. Must be valid UTF-8 encoding
    4. Must have valid Python AST (no syntax errors)
    5. Must meet minimum code density threshold
    """

    def __init__(
        self,
        max_size: int = MAX_PY_FILE_SIZE,
        min_code_density: float = MIN_CODE_DENSITY,
        strict_encoding: bool = True
    ):
        """
        Initialize validator with configurable thresholds.

        Args:
            max_size: Maximum file size in bytes
            min_code_density: Minimum code density (0.0-1.0)
            strict_encoding: Require strict UTF-8 encoding
        """
        self.max_size = max_size
        self.min_code_density = min_code_density
        self.strict_encoding = strict_encoding

        logger.info(f"Initialized PythonFileValidator:")
        logger.info(f"  Max size: {self.max_size / (1024*1024):.1f}MB")
        logger.info(f"  Min code density: {self.min_code_density}")
        logger.info(f"  Strict UTF-8: {self.strict_encoding}")

    def validate_extension(self, file_path: str) -> None:
        """
        Validate file has .py extension.

        Args:
            file_path: Path to file

        Raises:
            FileValidationError: If extension is not .py
        """
        ext = Path(file_path).suffix.lower()
        if ext != ALLOWED_EXTENSION:
            raise FileValidationError(
                f"Invalid extension '{ext}'. Only {ALLOWED_EXTENSION} files are allowed."
            )

    def validate_size(self, file_path: str) -> None:
        """
        Validate file size is within limits.

        Args:
            file_path: Path to file

        Raises:
            FileValidationError: If file exceeds size limit
        """
        try:
            file_size = os.path.getsize(file_path)
        except OSError as e:
            raise FileValidationError(f"Cannot access file: {e}")

        if file_size > self.max_size:
            raise FileValidationError(
                f"File size {file_size / (1024*1024):.2f}MB exceeds "
                f"maximum allowed {self.max_size / (1024*1024):.1f}MB"
            )

        if file_size == 0:
            raise FileValidationError("File is empty")

    def validate_encoding(self, content: str) -> None:
        """
        Validate content is valid UTF-8.

        Args:
            content: File content as string

        Raises:
            FileValidationError: If encoding is invalid
        """
        # Content is already decoded if we got here, so just validate it's not corrupted
        try:
            # Try to encode back to UTF-8 to verify
            content.encode('utf-8')
        except UnicodeEncodeError as e:
            raise FileValidationError(f"Invalid UTF-8 encoding: {e}")

    def validate_ast(self, content: str) -> ast.AST:
        """
        Validate Python syntax using AST parser.

        Args:
            content: Python source code

        Returns:
            Parsed AST tree

        Raises:
            FileValidationError: If syntax is invalid
        """
        try:
            # Suppress SyntaxWarning for invalid escape sequences in analyzed code
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=SyntaxWarning)
                tree = ast.parse(content)
            return tree
        except SyntaxError as e:
            raise FileValidationError(
                f"Python syntax error at line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            raise FileValidationError(f"AST parsing failed: {e}")

    def calculate_code_density(self, content: str) -> float:
        """
        Calculate code density (ratio of code lines to total lines).

        Args:
            content: Python source code

        Returns:
            Code density ratio (0.0-1.0)
        """
        lines = content.split('\n')
        if not lines:
            return 0.0

        code_lines = 0
        for line in lines:
            stripped = line.strip()
            # Skip empty lines and pure comment lines
            if stripped and not stripped.startswith('#'):
                code_lines += 1

        density = code_lines / len(lines)
        return density

    def validate_code_density(self, content: str) -> None:
        """
        Validate code meets minimum density threshold.

        Args:
            content: Python source code

        Raises:
            FileValidationError: If code density is too low
        """
        density = self.calculate_code_density(content)

        if density < self.min_code_density:
            raise FileValidationError(
                f"Code density {density:.2%} below minimum threshold "
                f"{self.min_code_density:.2%} (likely all comments)"
            )

    def validate_file(self, file_path: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Perform complete validation on a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Tuple of (is_valid, content, metadata)
            - is_valid: True if file passes all validation
            - content: File content if valid, None otherwise
            - metadata: Validation metadata (size, density, etc.)

        Note:
            This method does NOT raise exceptions - it returns validation status.
            Use validate_file_strict() for exception-based validation.
        """
        try:
            # 1. Extension check
            self.validate_extension(file_path)

            # 2. Size check (before reading into memory)
            self.validate_size(file_path)

            # 3. Read file with UTF-8 encoding
            try:
                if self.strict_encoding:
                    with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
                        content = f.read()
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
            except UnicodeDecodeError as e:
                raise FileValidationError(f"UTF-8 decoding failed: {e}")
            except Exception as e:
                raise FileValidationError(f"Failed to read file: {e}")

            # 4. Encoding validation
            self.validate_encoding(content)

            # 5. AST validation
            ast_tree = self.validate_ast(content)

            # 6. Code density check
            self.validate_code_density(content)

            # Calculate metadata
            file_size = os.path.getsize(file_path)
            code_density = self.calculate_code_density(content)

            metadata = {
                "file_size": file_size,
                "code_density": code_density,
                "line_count": len(content.split('\n')),
                "validation_passed": True
            }

            return True, content, metadata

        except FileValidationError as e:
            logger.debug(f"Validation failed for {file_path}: {e}")
            return False, None, {"validation_passed": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected validation error for {file_path}: {e}")
            return False, None, {"validation_passed": False, "error": f"Unexpected error: {e}"}

    def validate_file_strict(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Perform strict validation that raises exceptions on failure.

        Args:
            file_path: Path to Python file

        Returns:
            Tuple of (content, metadata)

        Raises:
            FileValidationError: If any validation step fails
        """
        is_valid, content, metadata = self.validate_file(file_path)

        if not is_valid:
            error_msg = metadata.get("error", "Unknown validation error")
            raise FileValidationError(error_msg)

        return content, metadata

    def validate_content(self, content: str, source_label: str = "memory") -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate Python code content (without file system access).

        Args:
            content: Python source code
            source_label: Label for logging (e.g., "memory", "url")

        Returns:
            Tuple of (is_valid, metadata)
        """
        try:
            # 1. Encoding validation
            self.validate_encoding(content)

            # 2. AST validation
            self.validate_ast(content)

            # 3. Code density check
            self.validate_code_density(content)

            # Calculate metadata
            code_density = self.calculate_code_density(content)

            metadata = {
                "content_length": len(content),
                "code_density": code_density,
                "line_count": len(content.split('\n')),
                "validation_passed": True
            }

            return True, metadata

        except FileValidationError as e:
            logger.debug(f"Content validation failed for {source_label}: {e}")
            return False, {"validation_passed": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected content validation error for {source_label}: {e}")
            return False, {"validation_passed": False, "error": f"Unexpected error: {e}"}


# Global validator instance
_default_validator: Optional[PythonFileValidator] = None


def get_default_validator() -> PythonFileValidator:
    """Get or create default validator instance."""
    global _default_validator
    if _default_validator is None:
        _default_validator = PythonFileValidator()
    return _default_validator


def validate_python_file(file_path: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Convenience function to validate a Python file using default validator.

    Args:
        file_path: Path to Python file

    Returns:
        Tuple of (is_valid, content, metadata)
    """
    validator = get_default_validator()
    return validator.validate_file(file_path)


def validate_python_content(content: str, source_label: str = "memory") -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Convenience function to validate Python code content using default validator.

    Args:
        content: Python source code
        source_label: Label for logging

    Returns:
        Tuple of (is_valid, metadata)
    """
    validator = get_default_validator()
    return validator.validate_content(content, source_label)
