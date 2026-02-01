import os
import sys
import signal
import yaml
from dotenv import load_dotenv
import logging

# Disable transformers lazy loading (fixes Python 3.13 Ctrl+C issues)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Graceful shutdown handler for Ctrl+C
def signal_handler(sig, frame):
    logger.info("\n\nShutting down gracefully...")
    logger.info("Cleaning up resources...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Import heavy dependencies after signal handler setup
from scriptguard.pipelines.training_pipeline import (
    malware_detection_training_pipeline,
    advanced_training_pipeline
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    for key, value in config.get("api_keys", {}).items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            config["api_keys"][key] = os.getenv(env_var)

    return config


def main_legacy():
    """Legacy training pipeline (original implementation)."""
    gh_malicious = [
        "https://github.com/example/malware/blob/main/shell.py"
    ]
    gh_benign = [
        "https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py"
    ]

    local_malicious = "./data/malicious_scripts"
    local_benign = "./data/benign_scripts"

    web_urls = ["https://raw.githubusercontent.com/python/cpython/main/Lib/os.py"]
    model_id = "bigcode/starcoder2-3b"

    logger.info("Starting ScriptGuard training pipeline with explicit label separation...")

    run = malware_detection_training_pipeline(
        gh_malicious_urls=gh_malicious,
        gh_benign_urls=gh_benign,
        local_malicious_dir=local_malicious,
        local_benign_dir=local_benign,
        web_urls=web_urls,
        model_id=model_id
    )

    logger.info("Pipeline run completed.")


def main():
    """Advanced training pipeline with config.yaml."""
    # Load configuration
    try:
        config = load_config("config.yaml")
        logger.info("Configuration loaded successfully")
    except FileNotFoundError:
        logger.error("config.yaml not found. Please create it from config.yaml.example")
        return
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Get model ID from config
    model_id = config.get("training", {}).get("model_id", "bigcode/starcoder2-3b")

    logger.info("Starting ScriptGuard Advanced Training Pipeline...")
    logger.info(f"Model: {model_id}")

    # Run advanced pipeline
    try:
        run = advanced_training_pipeline(
            config=config,
            model_id=model_id
        )
        logger.info("Pipeline run completed successfully!")
        logger.info(f"Results: {run}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
