"""
Test Qdrant connection with API key authentication.
"""

import os
from typing import Tuple

from qdrant_client import QdrantClient


def test_qdrant_with_api_key(
    host: str = "localhost",
    port: int = 6333,
    api_key: str = ""
) -> Tuple[bool, str]:
    """
    Test Qdrant connection with API key authentication.

    Args:
        host: Qdrant host
        port: Qdrant HTTP port
        api_key: Qdrant API key

    Returns:
        Tuple of (success, message)
    """
    try:
        if api_key:
            client = QdrantClient(
                url=f"http://{host}:{port}",
                api_key=api_key,
                timeout=5
            )
        else:
            client = QdrantClient(host=host, port=port, timeout=5)

        # Check health
        collections = client.get_collections()

        return True, f"Qdrant connected successfully. Collections: {len(collections.collections)}"

    except Exception as e:
        return False, f"Qdrant connection failed: {str(e)}"


def main() -> None:
    """Run connectivity test with API key."""
    print("=== Qdrant API Key Authentication Test ===\n")

    # Get configuration from environment
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

    print(f"Host: {qdrant_host}")
    print(f"Port: {qdrant_port}")
    print(f"API Key: {'***' + qdrant_api_key[-8:] if qdrant_api_key else 'Not set'}\n")

    # Test connection
    success, message = test_qdrant_with_api_key(
        host=qdrant_host,
        port=qdrant_port,
        api_key=qdrant_api_key
    )

    if success:
        print(f"✓ {message}")

        # List collections
        if qdrant_api_key:
            client = QdrantClient(
                url=f"http://{qdrant_host}:{qdrant_port}",
                api_key=qdrant_api_key
            )
        else:
            client = QdrantClient(host=qdrant_host, port=qdrant_port)

        collections = client.get_collections()
        print("\nAvailable collections:")
        for collection in collections.collections:
            print(f"  - {collection.name}")
    else:
        print(f"✗ {message}")
        print("\nTroubleshooting:")
        print("  1. Check if QDRANT_API_KEY is set correctly in .env")
        print("  2. Verify Qdrant is running and accessible")
        print("  3. Ensure API key has proper permissions")


if __name__ == "__main__":
    main()
