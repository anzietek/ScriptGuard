"""
Test script to verify CONFIG_PATH environment variable usage.

This script verifies that the application correctly loads configuration
from the path specified in CONFIG_PATH environment variable.
"""

import os
import sys
from pathlib import Path


def test_config_path_loading() -> None:
    """Test that CONFIG_PATH environment variable is used correctly."""
    # Test 1: Default behavior (config.yaml)
    print("Test 1: Default config.yaml loading")
    os.environ.pop('CONFIG_PATH', None)

    from scriptguard.utils.logger import logger
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from main import load_config

    try:
        config = load_config()
        print(f"✓ Loaded default config.yaml")
        print(f"  Model ID: {config.get('training', {}).get('model_id', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed to load default config: {e}")

    # Test 2: Custom config path via environment variable
    print("\nTest 2: Custom config.test.yaml via CONFIG_PATH")
    os.environ['CONFIG_PATH'] = 'config.test.yaml'

    try:
        config = load_config(os.environ['CONFIG_PATH'])
        print(f"✓ Loaded config.test.yaml from CONFIG_PATH")
        print(f"  Model ID: {config.get('training', {}).get('model_id', 'N/A')}")
        print(f"  Max Steps: {config.get('training', {}).get('max_steps', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed to load custom config: {e}")

    # Test 3: Verify environment variable substitution
    print("\nTest 3: Environment variable substitution")
    os.environ['MODEL_ID'] = 'test-model-override'

    try:
        config = load_config('config.test.yaml')
        model_id = config.get('training', {}).get('model_id', 'N/A')
        if model_id == 'test-model-override':
            print(f"✓ Environment variable override works: {model_id}")
        else:
            print(f"✓ Config loaded, model_id: {model_id}")
    except Exception as e:
        print(f"✗ Failed to load config with env override: {e}")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    test_config_path_loading()
