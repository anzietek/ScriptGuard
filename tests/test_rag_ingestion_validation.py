"""
Test Suite for RAG Pipeline Ingestion Validation
Tests strict Python gatekeeper, AST validation, and Qdrant parent-child structure.
"""

import pytest
import tempfile
import os
from pathlib import Path
from scriptguard.utils.file_validator import (
    PythonFileValidator,
    FileValidationError,
    validate_python_file,
    validate_python_content,
    MAX_PY_FILE_SIZE
)
from scriptguard.rag.chunking_service import ChunkingService


class TestPythonGatekeeper:
    """Test strict file validation rules."""

    def test_valid_python_file(self):
        """Test that valid Python file passes validation."""
        validator = PythonFileValidator()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def hello():\n    return 'world'\n")
            temp_path = f.name

        try:
            is_valid, content, metadata = validator.validate_file(temp_path)

            assert is_valid is True
            assert content is not None
            assert "hello" in content
            assert metadata["validation_passed"] is True
            assert metadata["code_density"] > 0
        finally:
            os.unlink(temp_path)

    def test_reject_non_python_extension(self):
        """Test that non-.py files are rejected."""
        validator = PythonFileValidator()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("print('hello')")
            temp_path = f.name

        try:
            is_valid, content, metadata = validator.validate_file(temp_path)

            assert is_valid is False
            assert content is None
            assert "Invalid extension" in metadata.get("error", "")
        finally:
            os.unlink(temp_path)

    def test_reject_large_file(self):
        """Test that files exceeding MAX_PY_FILE_SIZE are rejected."""
        validator = PythonFileValidator(max_size=1024)  # 1KB limit for testing

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write more than 1KB of valid Python
            f.write("# " + "x" * 2000 + "\n")
            f.write("def test():\n    pass\n")
            temp_path = f.name

        try:
            is_valid, content, metadata = validator.validate_file(temp_path)

            assert is_valid is False
            assert content is None
            assert "exceeds maximum" in metadata.get("error", "")
        finally:
            os.unlink(temp_path)

    def test_reject_empty_file(self):
        """Test that empty files are rejected."""
        validator = PythonFileValidator()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_path = f.name

        try:
            is_valid, content, metadata = validator.validate_file(temp_path)

            assert is_valid is False
            assert "empty" in metadata.get("error", "").lower()
        finally:
            os.unlink(temp_path)

    def test_reject_invalid_utf8(self):
        """Test that files with invalid UTF-8 are rejected."""
        validator = PythonFileValidator(strict_encoding=True)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as f:
            # Write invalid UTF-8 bytes
            f.write(b'\xff\xfe\x00\x00\x01\x02')
            temp_path = f.name

        try:
            is_valid, content, metadata = validator.validate_file(temp_path)

            assert is_valid is False
            assert "UTF-8" in metadata.get("error", "") or "decoding" in metadata.get("error", "").lower()
        finally:
            os.unlink(temp_path)

    def test_size_check_before_load(self):
        """Test that size validation happens before loading into memory."""
        validator = PythonFileValidator(max_size=100)  # 100 bytes

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write 200 bytes
            f.write("# " + "x" * 200)
            temp_path = f.name

        try:
            # This should fail at size check, not during content read
            with pytest.raises(FileValidationError) as exc_info:
                validator.validate_file_strict(temp_path)

            assert "exceeds maximum" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


class TestASTValidation:
    """Test AST-based syntax validation."""

    def test_valid_syntax_passes(self):
        """Test that valid Python syntax passes AST validation."""
        validator = PythonFileValidator()

        valid_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
"""

        is_valid, metadata = validator.validate_content(valid_code)
        assert is_valid is True
        assert metadata["validation_passed"] is True

    def test_syntax_error_rejected(self):
        """Test that Python syntax errors are caught and rejected."""
        validator = PythonFileValidator()

        # Unclosed parenthesis
        invalid_code = """
def broken_function(:
    return "unclosed"
"""

        is_valid, metadata = validator.validate_content(invalid_code)
        assert is_valid is False
        assert "syntax error" in metadata.get("error", "").lower()

    def test_multiple_syntax_errors(self):
        """Test various syntax errors are caught."""
        validator = PythonFileValidator()

        test_cases = [
            "def foo(:\n    pass",  # Invalid parameter
            "if True\n    pass",     # Missing colon
            "return = 5",            # Invalid assignment
            "for i in range(10\n    print(i)",  # Unclosed paren
        ]

        for invalid_code in test_cases:
            is_valid, metadata = validator.validate_content(invalid_code)
            assert is_valid is False, f"Should reject: {invalid_code[:30]}"

    def test_low_code_density_rejected(self):
        """Test that files with only comments are rejected."""
        validator = PythonFileValidator(min_code_density=0.1)

        # File with 90% comments
        mostly_comments = """
# This is a comment
# Another comment
# Yet another comment
# So many comments
# More comments
# Even more comments
# Still commenting
# Won't stop commenting
# Comments forever
print('one line of code')
"""

        is_valid, metadata = validator.validate_content(mostly_comments)
        # This should pass because we have some code
        assert is_valid is True

        # But this should fail (all comments)
        all_comments = "\n".join([f"# Comment {i}" for i in range(100)])
        is_valid, metadata = validator.validate_content(all_comments)
        assert is_valid is False
        assert "density" in metadata.get("error", "").lower()


class TestQdrantParentChildStructure:
    """Test parent-child structure in Qdrant payloads."""

    def test_chunk_contains_parent_id(self):
        """Test that chunks contain parent_id field."""
        chunker = ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            chunk_size=128,
            overlap=16
        )

        code = """
import os
import sys

def main():
    print("Hello, world!")
    for i in range(100):
        print(i)

if __name__ == "__main__":
    main()
"""

        chunks = chunker.chunk_code(
            code=code,
            db_id=123,
            label="benign",
            source="test"
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert "parent_id" in chunk
            assert chunk["parent_id"] is not None
            assert len(chunk["parent_id"]) == 64  # SHA256 hex digest

    def test_chunk_contains_parent_context(self):
        """Test that chunks contain parent_context field."""
        chunker = ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            chunk_size=128,
            overlap=16
        )

        code = """
'''Module for testing parent context.'''
import requests
from typing import List

class DataProcessor:
    def process(self, data: List[str]) -> None:
        pass

def helper_function(x, y, z):
    return x + y + z
"""

        chunks = chunker.chunk_code(
            code=code,
            db_id=456,
            label="benign",
            source="test"
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert "parent_context" in chunk
            assert chunk["parent_context"] is not None

            # Parent context should contain module-level info
            context = chunk["parent_context"]
            assert len(context) > 0
            # Should contain imports or class/function names
            assert any(keyword in context.lower() for keyword in
                      ["import", "class", "def", "module"])

    def test_all_chunks_same_parent_id(self):
        """Test that all chunks from same document have same parent_id."""
        chunker = ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            chunk_size=50,  # Small chunks to force multiple
            overlap=10
        )

        # Long code to ensure multiple chunks
        code = "\n".join([f"def function_{i}():\n    return {i}\n" for i in range(50)])

        chunks = chunker.chunk_code(
            code=code,
            db_id=789,
            label="malicious",
            source="test"
        )

        assert len(chunks) > 1, "Should generate multiple chunks"

        # All chunks should have same parent_id
        parent_ids = [chunk["parent_id"] for chunk in chunks]
        assert len(set(parent_ids)) == 1, "All chunks should share same parent_id"

    def test_parent_context_extraction(self):
        """Test parent context extraction from various code structures."""
        chunker = ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            chunk_size=512,
            overlap=64
        )

        test_cases = [
            # Module with docstring
            ('''"""This is a module."""\nimport os\n''', "module"),

            # Module with imports
            ('''import numpy as np\nimport pandas as pd\n''', "import"),

            # Module with class
            ('''class MyClass:\n    def method(self):\n        pass\n''', "class"),

            # Module with function
            ('''def my_function(a, b, c):\n    return a + b + c\n''', "def"),
        ]

        for code, expected_keyword in test_cases:
            chunks = chunker.chunk_code(code=code, db_id=1, label="benign", source="test")
            assert len(chunks) > 0

            context = chunks[0]["parent_context"].lower()
            assert expected_keyword in context, f"Expected '{expected_keyword}' in context: {context}"

    def test_single_chunk_has_parent_structure(self):
        """Test that even single-chunk documents have parent_id and parent_context."""
        chunker = ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            chunk_size=512,
            overlap=64
        )

        # Small code that fits in one chunk
        code = "def small():\n    return 42\n"

        chunks = chunker.chunk_code(
            code=code,
            db_id=999,
            label="benign",
            source="test"
        )

        assert len(chunks) == 1
        chunk = chunks[0]

        assert "parent_id" in chunk
        assert "parent_context" in chunk
        assert chunk["parent_id"] is not None
        assert chunk["parent_context"] is not None
        assert chunk["total_chunks"] == 1


class TestIntegration:
    """Integration tests for complete validation pipeline."""

    def test_valid_file_through_pipeline(self):
        """Test valid Python file goes through entire pipeline."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import math

class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2
""")
            temp_path = f.name

        try:
            # Step 1: Validate file
            is_valid, content, metadata = validate_python_file(temp_path)
            assert is_valid is True

            # Step 2: Chunk code
            chunker = ChunkingService()
            chunks = chunker.chunk_code(
                code=content,
                db_id=1,
                label="benign",
                source=temp_path
            )

            # Step 3: Verify chunk structure
            assert len(chunks) > 0
            for chunk in chunks:
                assert "parent_id" in chunk
                assert "parent_context" in chunk
                assert "content" in chunk
                assert chunk["db_id"] == 1
                assert chunk["label"] == "benign"
        finally:
            os.unlink(temp_path)

    def test_invalid_file_rejected_early(self):
        """Test that invalid files are rejected before chunking."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("not python")
            temp_path = f.name

        try:
            # Validation should fail
            is_valid, content, metadata = validate_python_file(temp_path)
            assert is_valid is False

            # Should not proceed to chunking
            assert content is None
        finally:
            os.unlink(temp_path)

    def test_syntax_error_rejected_early(self):
        """Test that syntax errors are rejected before chunking."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def broken(\n    pass")
            temp_path = f.name

        try:
            is_valid, content, metadata = validate_python_file(temp_path)
            assert is_valid is False
            assert "syntax" in metadata.get("error", "").lower()
        finally:
            os.unlink(temp_path)


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_validate_python_content_wrapper(self):
        """Test validate_python_content convenience function."""
        valid_code = "def test():\n    pass\n"
        is_valid, metadata = validate_python_content(valid_code, "test")
        assert is_valid is True

        invalid_code = "def test(\n    pass"
        is_valid, metadata = validate_python_content(invalid_code, "test")
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
