import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

env_dev = find_dotenv(".env.dev", usecwd=True)
if env_dev:
    load_dotenv(env_dev, override=True)

os.environ.setdefault("ZENML_ACTIVE_PROJECT_NAME", "default")
os.environ.setdefault("ZENML_ACTIVE_WORKSPACE_NAME", "default")

from scriptguard.utils.logger import logger
from scriptguard.config_loader import load_raw_config
from scriptguard.schemas.config_schema import validate_config
from scriptguard.pipelines.training_pipeline import codebert_training_pipeline


def main() -> None:
    config_path = os.getenv("CONFIG_PATH", "config.yaml")

    try:
        raw_config = load_raw_config(config_path)
        logger.info(f"Configuration loaded from: {config_path}")
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}. Set CONFIG_PATH env var.")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Failed to load config: {exc}")
        sys.exit(1)

    try:
        validate_config(raw_config)
        logger.info("Configuration validated successfully")
    except Exception as exc:
        logger.error(f"Configuration validation failed: {exc}")
        sys.exit(1)

    zenml_config_path = os.getenv("SCRIPTGUARD_ZENML_CONFIG", "zenml_config.yaml")

    logger.info("Starting CodeBERT training pipeline...")
    logger.info(f"Model: {raw_config.get('codebert', {}).get('model_name', 'microsoft/codebert-base')}")

    try:
        if os.path.exists(zenml_config_path):
            codebert_training_pipeline.with_options(config_path=zenml_config_path)(config=raw_config)
        else:
            codebert_training_pipeline(config=raw_config)
        logger.info("Pipeline completed successfully")
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
