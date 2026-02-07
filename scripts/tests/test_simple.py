import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from scriptguard.utils.file_validator import validate_python_content
from scriptguard.rag.chunking_service import ChunkingService

# Test validation
code = "def test():\n    return 42"
valid, meta = validate_python_content(code, "test")
print(f"Validation: {valid}")
assert valid

# Test chunking with parent structure
chunker = ChunkingService()
chunks = chunker.chunk_code(code, db_id=1, label="benign", source="test")
print(f"Chunks: {len(chunks)}")
print(f"Has parent_id: {'parent_id' in chunks[0]}")
print(f"Has parent_context: {'parent_context' in chunks[0]}")
assert "parent_id" in chunks[0]
assert "parent_context" in chunks[0]

print("\n✓ ALL TESTS PASSED")
