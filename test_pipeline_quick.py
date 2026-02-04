"""
Quick test pipeline with MINIMAL data - only to verify evaluate_model fix
"""
import os
import sys
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from scriptguard.utils.logger import logger
from scriptguard.pipelines.training_pipeline import advanced_training_pipeline

def load_config(config_path: str = "config.test.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    for key, value in config.get("api_keys", {}).items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            config["api_keys"][key] = os.getenv(env_var, "")

    return config

def main():
    logger.info("=" * 60)
    logger.info("QUICK TEST PIPELINE - Minimal data, only verify evaluate_model")
    logger.info("=" * 60)

    # Load test config with minimal data
    config = load_config("config.test.yaml")

    # Get model ID from config
    model_id = config.get("training", {}).get("model_id", "bigcode/starcoder2-3b")

    logger.info("Using MINIMAL test config:")
    logger.info(f"  - GitHub: 1 keyword, 5 samples max")
    logger.info(f"  - Training: 1 epoch only")
    logger.info(f"  - Goal: Verify evaluate_model PEFT loading works")
    logger.info("")

    # Run pipeline with test config
    try:
        run = advanced_training_pipeline.with_options(
            config_path="zenml_config.yaml"  # Use cache settings
        )(
            config=config,
            model_id=model_id
        )
        logger.info("=" * 60)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Results: {run}")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
