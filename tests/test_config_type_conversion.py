"""
Quick test to verify type conversion in config loading.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import load_config


def test_type_conversion():
    """Test that environment variables are properly converted to correct types."""
    print("Testing type conversion in config.test.yaml...\n")

    # Set some test environment variables
    os.environ['MAX_STEPS'] = '20'
    os.environ['BATCH_SIZE'] = '4'
    os.environ['LEARNING_RATE'] = '0.001'
    os.environ['QDRANT_PORT'] = '6334'

    config = load_config('config.test.yaml')

    # Test integer conversion
    max_steps = config['training']['max_steps']
    batch_size = config['training']['batch_size']
    qdrant_port = config['qdrant']['port']

    print(f"max_steps: {max_steps} (type: {type(max_steps).__name__})")
    assert isinstance(max_steps, int), f"Expected int, got {type(max_steps)}"
    assert max_steps == 20, f"Expected 20, got {max_steps}"

    print(f"batch_size: {batch_size} (type: {type(batch_size).__name__})")
    assert isinstance(batch_size, int), f"Expected int, got {type(batch_size)}"
    assert batch_size == 4, f"Expected 4, got {batch_size}"

    print(f"qdrant_port: {qdrant_port} (type: {type(qdrant_port).__name__})")
    assert isinstance(qdrant_port, int), f"Expected int, got {type(qdrant_port)}"
    assert qdrant_port == 6334, f"Expected 6334, got {qdrant_port}"

    # Test float conversion
    learning_rate = config['training']['learning_rate']
    print(f"learning_rate: {learning_rate} (type: {type(learning_rate).__name__})")
    assert isinstance(learning_rate, float), f"Expected float, got {type(learning_rate)}"
    assert learning_rate == 0.001, f"Expected 0.001, got {learning_rate}"

    # Test string values (API keys should remain strings)
    github_token = config['api_keys']['github_token']
    print(f"github_token: {github_token[:20]}... (type: {type(github_token).__name__})")
    assert isinstance(github_token, str), f"Expected str, got {type(github_token)}"

    # Test default values (without env vars)
    os.environ.pop('MAX_SAMPLES_PER_KEYWORD', None)
    config2 = load_config('config.test.yaml')
    max_samples = config2['data_sources']['github']['max_samples_per_keyword']
    print(f"max_samples_per_keyword (default): {max_samples} (type: {type(max_samples).__name__})")
    assert isinstance(max_samples, int), f"Expected int, got {type(max_samples)}"
    assert max_samples == 2, f"Expected 2, got {max_samples}"

    print("\n✅ All type conversion tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_type_conversion()
