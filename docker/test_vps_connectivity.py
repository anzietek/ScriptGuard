"""
Test PostgreSQL and Qdrant connectivity on VPS deployment.
"""

import os
import sys
from typing import Tuple

import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http import models


def test_postgres_connection(
    host: str = "localhost",
    port: int = 5432,
    database: str = "scriptguard",
    user: str = "scriptguard",
    password: str = ""
) -> Tuple[bool, str]:
    """
    Test PostgreSQL connection.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: Database name
        user: Database user
        password: Database password

    Returns:
        Tuple of (success, message)
    """
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            connect_timeout=5
        )

        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return True, f"PostgreSQL connected successfully: {version}"

    except Exception as e:
        return False, f"PostgreSQL connection failed: {str(e)}"


def test_qdrant_connection(
    host: str = "localhost",
    port: int = 6333,
    grpc_port: int = 6334
) -> Tuple[bool, str]:
    """
    Test Qdrant connection.

    Args:
        host: Qdrant host
        port: Qdrant HTTP port
        grpc_port: Qdrant gRPC port

    Returns:
        Tuple of (success, message)
    """
    try:
        # Try HTTP connection
        client = QdrantClient(host=host, port=port, timeout=5)

        # Check health
        collections = client.get_collections()

        return True, f"Qdrant connected successfully. Collections: {len(collections.collections)}"

    except Exception as e:
        return False, f"Qdrant connection failed: {str(e)}"


def main() -> None:
    """Run connectivity tests."""
    print("=== ScriptGuard VPS Connectivity Test ===\n")

    # Get configuration from environment
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_port = int(os.getenv("POSTGRES_PORT", "5432"))
    pg_database = os.getenv("POSTGRES_DB", "scriptguard")
    pg_user = os.getenv("POSTGRES_USER", "scriptguard")
    pg_password = os.getenv("POSTGRES_PASSWORD", "")

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
    qdrant_grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

    # Test PostgreSQL
    print(f"Testing PostgreSQL connection to {pg_host}:{pg_port}...")
    pg_success, pg_message = test_postgres_connection(
        host=pg_host,
        port=pg_port,
        database=pg_database,
        user=pg_user,
        password=pg_password
    )

    if pg_success:
        print(f"✓ {pg_message}\n")
    else:
        print(f"✗ {pg_message}\n")

    # Test Qdrant
    print(f"Testing Qdrant connection to {qdrant_host}:{qdrant_port}...")
    qdrant_success, qdrant_message = test_qdrant_connection(
        host=qdrant_host,
        port=qdrant_port,
        grpc_port=qdrant_grpc_port
    )

    if qdrant_success:
        print(f"✓ {qdrant_message}\n")
    else:
        print(f"✗ {qdrant_message}\n")

    # Summary
    print("=== Summary ===")
    print(f"PostgreSQL: {'PASS' if pg_success else 'FAIL'}")
    print(f"Qdrant: {'PASS' if qdrant_success else 'FAIL'}")

    if pg_success and qdrant_success:
        print("\nAll tests passed! VPS deployment is ready.")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please check the configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
