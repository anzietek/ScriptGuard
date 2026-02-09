"""
Unit Tests for Hierarchical Chunking (AST-aware)

Tests the hierarchical chunking functionality that completes the parent-child
architecture by using AST boundaries for chunk creation.
"""

import pytest
from scriptguard.rag.chunking_service import ChunkingService


class TestHierarchicalChunking:
    """Test suite for hierarchical (AST-aware) chunking."""

    @pytest.fixture
    def chunker(self):
        """Create chunking service configured for hierarchical chunking."""
        return ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            chunk_size=256,  # Smaller for testing
            overlap=32,
            max_function_tokens=512,
            strategy="hierarchical"
        )

    @pytest.fixture
    def sliding_chunker(self):
        """Create chunking service configured for sliding window."""
        return ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            chunk_size=256,
            overlap=32,
            strategy="sliding_window"
        )

    def test_complete_functions_no_mixing(self, chunker):
        """Test: Each chunk contains complete function, no mixing."""
        code = """
import os
import sys

def authenticate_user(username, password):
    # Authentication logic
    if username and password:
        return True
    return False

def process_payment(amount, card_number):
    # Payment processing
    if amount > 0:
        print(f"Processing ${amount}")
        return True
    return False
"""
        chunks = chunker.chunk_code(code, db_id=1, label="benign", language="python")

        # Should have at least 2 function chunks (+ possibly module chunk)
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

        # Find function chunks
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        assert len(func_chunks) == 2, f"Expected 2 function chunks, got {len(func_chunks)}"

        # Verify function names
        names = [c["function_name"] for c in func_chunks]
        assert "authenticate_user" in names, "authenticate_user not found"
        assert "process_payment" in names, "process_payment not found"

        # Verify no mixing: each function chunk should not contain the other function
        for chunk in func_chunks:
            if chunk["function_name"] == "authenticate_user":
                assert "process_payment" not in chunk["content"], "Functions are mixed!"
                assert "authenticate_user" in chunk["content"]
            elif chunk["function_name"] == "process_payment":
                assert "authenticate_user" not in chunk["content"], "Functions are mixed!"
                assert "process_payment" in chunk["content"]

    def test_large_function_fallback(self, chunker):
        """Test: Large functions fallback to sliding window."""
        # Create a function with >512 tokens (max_function_tokens)
        large_func = "def huge_function():\n" + "    x = 1\n" * 1000

        chunks = chunker.chunk_code(large_func, db_id=2, label="benign", language="python")

        # Should be multiple chunks (sliding window fallback)
        assert len(chunks) > 1, "Large function should be split into multiple chunks"

        # All chunks should have token_count <= 256 (chunk_size for fallback)
        for chunk in chunks:
            assert chunk["token_count"] <= 256, f"Chunk token count {chunk['token_count']} exceeds limit"

        # Should be marked as sliding_window_fallback
        fallback_chunks = [c for c in chunks if c["chunk_type"] == "sliding_window_fallback"]
        assert len(fallback_chunks) > 0, "Should have sliding_window_fallback chunks"

        # Function name should be preserved with "(partial)" marker
        for chunk in fallback_chunks:
            assert "huge_function" in chunk.get("function_name", ""), "Function name not preserved"

    def test_syntax_error_fallback(self, chunker):
        """Test: Syntax errors trigger sliding window fallback."""
        invalid_code = """
def broken_function(
    # missing closing parenthesis
    x = 1
"""

        chunks = chunker.chunk_code(invalid_code, db_id=3, label="benign", language="python")

        # Should fallback to sliding window
        assert len(chunks) >= 1, "Should have at least one chunk"
        assert all(c["chunk_type"] == "sliding_window" for c in chunks), \
            "Should use sliding_window on syntax error"

    def test_non_python_fallback(self, chunker):
        """Test: Non-Python code uses sliding window."""
        js_code = """
function hello() {
    console.log('Hello, world!');
    return true;
}

function goodbye() {
    console.log('Goodbye!');
}
"""

        chunks = chunker.chunk_code(js_code, db_id=4, label="benign", language="javascript")

        # Should fallback to sliding window
        assert len(chunks) >= 1, "Should have at least one chunk"
        assert all(c["chunk_type"] == "sliding_window" for c in chunks), \
            "Non-Python should use sliding_window"

    def test_class_chunking(self, chunker):
        """Test: Classes are chunked as complete units."""
        code = """
class MalwareScanner:
    def __init__(self, config):
        self.config = config

    def scan(self, file_path):
        # Scanning logic
        print(f"Scanning {file_path}")
        return "safe"

    def report(self):
        return {"status": "complete"}
"""
        chunks = chunker.chunk_code(code, db_id=5, label="benign", language="python")

        # Should have at least one class chunk
        class_chunks = [c for c in chunks if c["chunk_type"] == "class"]
        assert len(class_chunks) >= 1, "Should have at least one class chunk"
        assert class_chunks[0]["function_name"] == "MalwareScanner", \
            f"Expected MalwareScanner, got {class_chunks[0]['function_name']}"

        # Class chunk should contain all methods
        class_content = class_chunks[0]["content"]
        assert "__init__" in class_content, "Class should contain __init__"
        assert "scan" in class_content, "Class should contain scan method"
        assert "report" in class_content, "Class should contain report method"

    def test_module_level_code_preserved(self, chunker):
        """Test: Module-level code (imports, globals) preserved."""
        code = """
import os
import sys
import base64

API_KEY = "secret"
DEBUG = True

def main():
    print("Hello")
    return 0
"""
        chunks = chunker.chunk_code(code, db_id=6, label="benign", language="python")

        # Should have module chunk + function chunk
        module_chunks = [c for c in chunks if c["chunk_type"] == "module"]
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]

        assert len(module_chunks) >= 1, "Should have module chunk"
        assert len(func_chunks) >= 1, "Should have function chunk"

        # Module chunk should contain imports and globals
        module_content = module_chunks[0]["content"]
        assert "import os" in module_content, "Module should contain imports"
        assert "API_KEY" in module_content, "Module should contain globals"

        # Function chunk should NOT contain imports (no duplication)
        func_content = func_chunks[0]["content"]
        assert "def main" in func_content, "Function chunk should contain function"
        # It's OK if imports are referenced, but they shouldn't be the main content

    def test_parent_child_metadata_preserved(self, chunker):
        """Test: Parent-child metadata is correctly preserved."""
        code = """
def func_a():
    return 1

def func_b():
    return 2
"""
        chunks = chunker.chunk_code(code, db_id=7, label="malicious", source="test", language="python")

        assert len(chunks) >= 2, "Should have multiple chunks"

        # All chunks should have same parent_id
        parent_ids = [c["parent_id"] for c in chunks]
        assert len(set(parent_ids)) == 1, "All chunks should have same parent_id"

        # All chunks should have parent_context
        for chunk in chunks:
            assert "parent_context" in chunk, "Chunk missing parent_context"
            assert chunk["parent_context"], "parent_context should not be empty"

        # Check chunk indices
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i, f"Chunk {i} has wrong index: {chunk['chunk_index']}"
            assert chunk["total_chunks"] == len(chunks), "total_chunks mismatch"

        # Check metadata propagation
        for chunk in chunks:
            assert chunk["db_id"] == 7, "db_id not preserved"
            assert chunk["label"] == "malicious", "label not preserved"
            assert chunk["source"] == "test", "source not preserved"

    def test_async_functions_handled(self, chunker):
        """Test: Async functions are correctly extracted."""
        code = """
import asyncio

async def fetch_data(url):
    # Async fetch
    await asyncio.sleep(1)
    return "data"

async def process_async():
    data = await fetch_data("http://example.com")
    return data
"""
        chunks = chunker.chunk_code(code, db_id=8, label="benign", language="python")

        # Should have function chunks for async functions
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        assert len(func_chunks) >= 2, "Should have async function chunks"

        names = [c["function_name"] for c in func_chunks]
        assert "fetch_data" in names, "fetch_data not found"
        assert "process_async" in names, "process_async not found"

    def test_mixed_functions_and_classes(self, chunker):
        """Test: Mix of functions and classes handled correctly."""
        code = """
import os

def helper_function():
    return "helper"

class DataProcessor:
    def process(self, data):
        return helper_function()

def main():
    processor = DataProcessor()
    return processor.process("test")
"""
        chunks = chunker.chunk_code(code, db_id=9, label="benign", language="python")

        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        class_chunks = [c for c in chunks if c["chunk_type"] == "class"]

        assert len(func_chunks) >= 2, "Should have function chunks"
        assert len(class_chunks) >= 1, "Should have class chunk"

        # Verify separation
        func_names = [c["function_name"] for c in func_chunks]
        assert "helper_function" in func_names
        assert "main" in func_names

        class_names = [c["function_name"] for c in class_chunks]
        assert "DataProcessor" in class_names

    def test_comparison_with_sliding_window(self, chunker, sliding_chunker):
        """Test: Compare hierarchical vs sliding window results."""
        code = """
def authenticate(username, password):
    # 100 tokens of logic
    if not username or not password:
        return False
    # More authentication logic
    return True

def authorize(user_id, resource):
    # 100 tokens of logic
    if not user_id:
        return False
    # More authorization logic
    return True
"""
        hierarchical_chunks = chunker.chunk_code(code, db_id=10, label="benign", language="python")
        sliding_chunks = sliding_chunker.chunk_code(code, db_id=10, label="benign")

        # Hierarchical should have fewer chunks (no overlap)
        # and each chunk should be a complete function
        func_chunks = [c for c in hierarchical_chunks if c["chunk_type"] == "function"]
        assert len(func_chunks) == 2, "Should have 2 complete function chunks"

        # Verify no overlap in hierarchical
        for chunk in hierarchical_chunks:
            if chunk["chunk_type"] == "function":
                # Each function chunk should contain only one function definition
                assert chunk["content"].count("def ") == 1, "Function chunk should contain exactly one function"

    def test_empty_file_handling(self, chunker):
        """Test: Empty files handled gracefully."""
        code = ""
        chunks = chunker.chunk_code(code, db_id=11, label="benign", language="python")
        assert len(chunks) == 0, "Empty code should return no chunks"

        code_whitespace = "   \n  \n  "
        chunks = chunker.chunk_code(code_whitespace, db_id=12, label="benign", language="python")
        assert len(chunks) == 0, "Whitespace-only code should return no chunks"

    def test_single_small_function(self, chunker):
        """Test: Single small function handled correctly."""
        code = """
def hello():
    print("Hello, world!")
"""
        chunks = chunker.chunk_code(code, db_id=13, label="benign", language="python")

        assert len(chunks) >= 1, "Should have at least one chunk"
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        assert len(func_chunks) == 1, "Should have exactly one function chunk"
        assert func_chunks[0]["function_name"] == "hello"

    def test_strategy_override(self, chunker):
        """Test: Strategy can be overridden per call."""
        code = """
def test_func():
    return True
"""
        # Override to sliding window
        chunks = chunker.chunk_code(code, db_id=14, label="benign", strategy="sliding_window")
        assert all(c["chunk_type"] == "sliding_window" for c in chunks), \
            "Strategy override should use sliding_window"

        # Use default hierarchical
        chunks = chunker.chunk_code(code, db_id=15, label="benign", language="python")
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        assert len(func_chunks) >= 1, "Default strategy should use hierarchical"


class TestParentChildIntegrity:
    """Test parent-child relationship integrity."""

    @pytest.fixture
    def chunker(self):
        return ChunkingService(
            tokenizer_name="microsoft/unixcoder-base",
            max_function_tokens=512,
            strategy="hierarchical"
        )

    def test_parent_id_consistency(self, chunker):
        """Test: All chunks from same document have same parent_id."""
        code = """
def func1():
    return 1

def func2():
    return 2

def func3():
    return 3
"""
        chunks = chunker.chunk_code(code, db_id=100, label="benign", language="python")

        assert len(chunks) >= 3, "Should have multiple chunks"
        parent_ids = [c["parent_id"] for c in chunks]
        assert len(set(parent_ids)) == 1, "All chunks should share same parent_id"

    def test_parent_context_includes_signatures(self, chunker):
        """Test: Parent context includes function signatures."""
        code = """
import socket

def establish_backdoor(host, port):
    s = socket.socket()
    s.connect((host, port))
    return s

def exfiltrate_data(data):
    # Malicious data exfiltration
    return base64.b64encode(data)
"""
        chunks = chunker.chunk_code(code, db_id=101, label="malicious", language="python")

        # All chunks should have parent_context
        for chunk in chunks:
            parent_ctx = chunk.get("parent_context", "")
            assert parent_ctx, "parent_context should not be empty"
            # Should contain function signatures
            assert "establish_backdoor" in parent_ctx or "exfiltrate" in parent_ctx or "socket" in parent_ctx, \
                f"parent_context missing key info: {parent_ctx}"

    def test_db_id_propagation(self, chunker):
        """Test: db_id correctly propagates to all chunks."""
        code = """
def test():
    pass
"""
        chunks = chunker.chunk_code(code, db_id=999, label="benign", language="python")

        for chunk in chunks:
            assert chunk["db_id"] == 999, "db_id not correctly propagated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
