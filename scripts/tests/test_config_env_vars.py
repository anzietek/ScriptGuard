"""
Test configuration loading with environment variable substitution.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from main import load_config


def test_config_env_substitution() -> None:
    """Test that environment variables are properly substituted in config."""
    print("=== Testing Configuration Environment Variable Substitution ===\n")

    # Set test environment variables
    os.environ["QDRANT_HOST"] = "test-host"
    os.environ["QDRANT_PORT"] = "9999"
    os.environ["QDRANT_API_KEY"] = "test-api-key"
    os.environ["POSTGRES_HOST"] = "test-postgres"
    os.environ["POSTGRES_PORT"] = "5555"

    # Load config
    config = load_config(str(Path(__file__).parent.parent.parent / "config.yaml"))

    # Check Qdrant configuration
    qdrant_config = config.get("qdrant", {})
    print("Qdrant Configuration:")
    print(f"  host: {qdrant_config.get('host')}")
    print(f"  port: {qdrant_config.get('port')}")
    print(f"  api_key: {qdrant_config.get('api_key')}")
    print()

    # Check PostgreSQL configuration
    pg_config = config.get("database", {}).get("postgresql", {})
    print("PostgreSQL Configuration:")
    print(f"  host: {pg_config.get('host')}")
    print(f"  port: {pg_config.get('port')}")
    print()

    # Verify substitution worked
    assert qdrant_config.get("host") == "test-host", f"Expected 'test-host', got '{qdrant_config.get('host')}'"
    assert qdrant_config.get("port") == "9999", f"Expected '9999', got '{qdrant_config.get('port')}'"
    assert qdrant_config.get("api_key") == "test-api-key", f"Expected 'test-api-key', got '{qdrant_config.get('api_key')}'"
    assert pg_config.get("host") == "test-postgres", f"Expected 'test-postgres', got '{pg_config.get('host')}'"
    assert pg_config.get("port") == "5555", f"Expected '5555', got '{pg_config.get('port')}'"

    print("✓ All environment variables correctly substituted!")
    print()

    # Test default values
    print("Testing default values...")
    os.environ.pop("QDRANT_HOST", None)
    config = load_config(str(Path(__file__).parent.parent.parent / "config.yaml"))
    qdrant_config = config.get("qdrant", {})

    print(f"  QDRANT_HOST (unset): {qdrant_config.get('host')}")
    assert qdrant_config.get("host") == "localhost", "Default value should be 'localhost'"
    print("✓ Default values work correctly!")
    print()


def test_real_env_values() -> None:
    """Test with actual .env values."""
    print("=== Testing with Real .env Values ===\n")

    # Load .env file
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}\n")

        config = load_config(str(Path(__file__).parent.parent.parent / "config.yaml"))

        qdrant_config = config.get("qdrant", {})
        pg_config = config.get("database", {}).get("postgresql", {})

        print("Qdrant Configuration from .env:")
        print(f"  host: {qdrant_config.get('host')}")
        print(f"  port: {qdrant_config.get('port')}")
        print(f"  api_key: {'***' + str(qdrant_config.get('api_key', ''))[-8:] if qdrant_config.get('api_key') else 'Not set'}")
        print()

        print("PostgreSQL Configuration from .env:")
        print(f"  host: {pg_config.get('host')}")
        print(f"  port: {pg_config.get('port')}")
        print(f"  database: {pg_config.get('database')}")
        print(f"  user: {pg_config.get('user')}")
        print()

        # Verify VPS configuration is loaded
        if os.getenv("QDRANT_HOST"):
            assert qdrant_config.get("host") == os.getenv("QDRANT_HOST")
            print("✓ QDRANT_HOST correctly loaded from .env")

        if os.getenv("QDRANT_API_KEY"):
            assert qdrant_config.get("api_key") == os.getenv("QDRANT_API_KEY")
            print("✓ QDRANT_API_KEY correctly loaded from .env")

        print()
    else:
        print(f"⚠ .env file not found at {env_path}")
        print("Skipping real environment test")
        print()


if __name__ == "__main__":
    try:
        test_config_env_substitution()
        test_real_env_values()
        print("=== All Tests Passed! ===")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
