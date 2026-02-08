"""
Test suite for config.test.yaml validation.

This module validates that the test configuration is optimized for speed
and contains all required fields for pipeline execution.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict

import pytest


def load_test_config() -> Dict[str, Any]:
    """
    Load the test configuration file.

    Returns:
        Parsed YAML configuration dictionary.
    """
    config_path = Path(__file__).parent.parent / "config.test.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestConfigStructure:
    """Test configuration structure and required fields."""

    def test_config_loads(self) -> None:
        """Test that config.test.yaml loads without errors."""
        config = load_test_config()
        assert config is not None
        assert isinstance(config, dict)

    def test_required_sections_exist(self) -> None:
        """Test that all required configuration sections exist."""
        config = load_test_config()
        required_sections = [
            'pipeline',
            'api_keys',
            'data_sources',
            'validation',
            'augmentation',
            'qdrant',
            'code_embedding',
            'training',
            'wandb',
            'database'
        ]
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"


class TestSpeedOptimizations:
    """Test that configuration is optimized for speed."""

    def test_cache_disabled(self) -> None:
        """Test that caching is disabled for fresh runs."""
        config = load_test_config()
        assert config['pipeline']['enable_cache'] is False

    def test_minimal_data_sources(self) -> None:
        """Test that data sources use minimal samples."""
        config = load_test_config()
        github = config['data_sources']['github']
        max_samples = int(os.getenv('TEST_MAX_SAMPLES_PER_KEYWORD', '2'))
        max_files = int(os.getenv('TEST_MAX_FILES_PER_REPO', '2'))
        assert github['max_samples_per_keyword'] <= max_samples
        assert github['max_files_per_repo'] <= max_files

    def test_non_essential_sources_disabled(self) -> None:
        """Test that non-essential data sources are disabled."""
        config = load_test_config()
        sources = config['data_sources']
        assert sources['malwarebazaar']['enabled'] is False
        assert sources['huggingface']['enabled'] is False
        assert sources['cve_feeds']['enabled'] is False
        assert sources['additional_hf']['enabled'] is False

    def test_validation_relaxed(self) -> None:
        """Test that validation is relaxed for speed."""
        config = load_test_config()
        validation = config['validation']
        expected_syntax_check = os.getenv('TEST_VALIDATE_SYNTAX', 'false').lower() == 'true'
        max_min_length = int(os.getenv('TEST_MAX_MIN_LENGTH', '10'))
        max_min_code_lines = int(os.getenv('TEST_MAX_MIN_CODE_LINES', '1'))
        assert validation['validate_syntax'] is expected_syntax_check
        assert validation['min_length'] <= max_min_length
        assert validation['min_code_lines'] <= max_min_code_lines

    def test_augmentation_disabled(self) -> None:
        """Test that augmentation is disabled."""
        config = load_test_config()
        augmentation = config['augmentation']
        assert augmentation['enabled'] is False
        assert augmentation['balance_dataset'] is False
        assert augmentation['use_qdrant_patterns'] is False

    def test_vectorization_disabled(self) -> None:
        """Test that vectorization is disabled."""
        config = load_test_config()
        assert config['code_embedding']['max_samples_to_vectorize'] == 0

    def test_minimal_training_steps(self) -> None:
        """Test that training uses minimal steps."""
        config = load_test_config()
        training = config['training']
        expected_epochs = int(os.getenv('TEST_NUM_EPOCHS', '1'))
        expected_batch_size = int(os.getenv('TEST_BATCH_SIZE', '1'))
        expected_max_steps = int(os.getenv('TEST_MAX_STEPS', '10'))
        expected_warmup_steps = int(os.getenv('TEST_WARMUP_STEPS', '1'))
        assert training['num_epochs'] == expected_epochs
        assert training['batch_size'] == expected_batch_size
        assert training['max_steps'] == expected_max_steps
        assert training['warmup_steps'] == expected_warmup_steps

    def test_wandb_disabled(self) -> None:
        """Test that W&B logging is disabled."""
        config = load_test_config()
        assert config['wandb']['enabled'] is False

    def test_database_disabled(self) -> None:
        """Test that database is disabled."""
        config = load_test_config()
        assert config['database']['enabled'] is False


class TestTrainingConfiguration:
    """Test training-specific configurations."""

    def test_lora_minimal_config(self) -> None:
        """Test that LoRA uses minimal configuration."""
        config = load_test_config()
        training = config['training']
        expected_lora_r = int(os.getenv('TEST_LORA_R', '8'))
        expected_lora_alpha = int(os.getenv('TEST_LORA_ALPHA', '16'))
        expected_target_modules_count = int(os.getenv('TEST_TARGET_MODULES_COUNT', '2'))
        assert training['lora_r'] == expected_lora_r
        assert training['lora_alpha'] == expected_lora_alpha
        assert len(training['target_modules']) == expected_target_modules_count

    def test_sequence_length_reduced(self) -> None:
        """Test that sequence length is reduced for speed."""
        config = load_test_config()
        expected_max_seq_length = int(os.getenv('TEST_MAX_SEQ_LENGTH', '256'))
        assert config['training']['max_seq_length'] == expected_max_seq_length

    def test_precision_settings(self) -> None:
        """Test that mixed precision is disabled."""
        config = load_test_config()
        training = config['training']
        assert training['fp16'] is False
        assert training['bf16'] is False

    def test_evaluation_minimal(self) -> None:
        """Test that evaluation uses minimal tokens."""
        config = load_test_config()
        training = config['training']
        expected_eval_max_tokens = int(os.getenv('TEST_EVAL_MAX_NEW_TOKENS', '10'))
        expected_eval_max_code_length = int(os.getenv('TEST_EVAL_MAX_CODE_LENGTH', '200'))
        assert training['eval_max_new_tokens'] == expected_eval_max_tokens
        assert training['eval_max_code_length'] == expected_eval_max_code_length

    def test_test_split_size(self) -> None:
        """Test that test split is configured."""
        config = load_test_config()
        assert 0.0 < config['training']['test_split_size'] <= 0.5


class TestAPIKeys:
    """Test API key configuration."""

    def test_api_keys_use_env_vars(self) -> None:
        """Test that API keys reference environment variables."""
        config = load_test_config()
        api_keys = config['api_keys']

        expected_keys = [
            'github_token',
            'nvd_api_key',
            'malwarebazaar_api_key',
            'huggingface_token'
        ]

        for key in expected_keys:
            assert key in api_keys
            assert isinstance(api_keys[key], str)
            assert api_keys[key].startswith('${')


class TestQdrantConfiguration:
    """Test Qdrant vector store configuration."""

    def test_qdrant_connection_params(self) -> None:
        """Test that Qdrant connection parameters are set."""
        config = load_test_config()
        qdrant = config['qdrant']
        expected_host = os.getenv('QDRANT_HOST', 'localhost')
        expected_port = int(os.getenv('QDRANT_PORT', '6333'))
        assert qdrant['host'] == expected_host
        assert qdrant['port'] == expected_port
        assert 'collection_name' in qdrant
        assert 'cve_collection' in qdrant

    def test_qdrant_test_collections(self) -> None:
        """Test that test-specific collection names are used."""
        config = load_test_config()
        qdrant = config['qdrant']
        assert 'test' in qdrant['collection_name'].lower()
        assert 'test' in qdrant['cve_collection'].lower()


@pytest.mark.parametrize("section,key,expected_type", [
    ("training", "batch_size", int),
    ("training", "learning_rate", float),
    ("training", "max_steps", int),
    ("validation", "min_length", int),
    ("validation", "max_length", int),
    ("code_embedding", "chunk_size", int),
    ("code_embedding", "chunk_overlap", int),
])
def test_numeric_field_types(section: str, key: str, expected_type: type) -> None:
    """
    Test that numeric fields have correct types.

    Args:
        section: Configuration section name.
        key: Configuration key within section.
        expected_type: Expected Python type.
    """
    config = load_test_config()
    assert isinstance(config[section][key], expected_type)


@pytest.mark.parametrize("section,key,min_value", [
    ("training", "batch_size", 1),
    ("training", "num_epochs", 1),
    ("training", "max_steps", 1),
    ("validation", "min_length", 1),
    ("code_embedding", "chunk_size", 1),
])
def test_positive_values(section: str, key: str, min_value: int) -> None:
    """
    Test that numeric fields have positive values.

    Args:
        section: Configuration section name.
        key: Configuration key within section.
        min_value: Minimum allowed value.
    """
    config = load_test_config()
    assert config[section][key] >= min_value
