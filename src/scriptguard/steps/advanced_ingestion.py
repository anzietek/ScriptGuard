"""
Advanced Data Ingestion Step
Integrates multiple data sources with deduplication and database storage.
"""

from scriptguard.utils.logger import logger
from typing import Dict, List
from zenml import step

from ..data_sources import (
    GitHubDataSource,
    MalwareBazaarDataSource,
    HuggingFaceDataSource,
    CVEFeedSource
)
from ..data_sources.additional_hf_datasets import AdditionalHFDatasets
from ..database import DatasetManager, deduplicate_samples
from ..monitoring import DatasetStatistics

@step(enable_cache=False)
def advanced_data_ingestion(config: dict) -> List[Dict]:
    """
    Ingest data from multiple sources with deduplication.

    Args:
        config: Configuration dictionary with data source settings

    Returns:
        List of unique code samples
    """
    logger.info("Starting advanced data ingestion...")

    all_samples = []

    # GitHub Data Source
    if config.get("data_sources", {}).get("github", {}).get("enabled", False):
        logger.info("Fetching data from GitHub...")
        github_config = config["data_sources"]["github"]

        try:
            github_source = GitHubDataSource(
                api_token=config.get("api_keys", {}).get("github_token")
            )

            # Fetch malicious samples
            if github_config.get("fetch_malicious", True):
                malicious_samples = github_source.fetch_malicious_samples(
                    keywords=github_config.get("malicious_keywords"),
                    max_per_keyword=github_config.get("max_samples_per_keyword", 20)
                )
                all_samples.extend(malicious_samples)
                logger.info(f"Fetched {len(malicious_samples)} malicious samples from GitHub")

            # Fetch benign samples
            if github_config.get("fetch_benign", True):
                benign_samples = github_source.fetch_benign_samples(
                    popular_repos=github_config.get("benign_repos"),
                    max_files_per_repo=github_config.get("max_files_per_repo", 50)
                )
                all_samples.extend(benign_samples)
                logger.info(f"Fetched {len(benign_samples)} benign samples from GitHub")

        except Exception as e:
            logger.error(f"GitHub data source failed: {e}")

    # MalwareBazaar Data Source
    if config.get("data_sources", {}).get("malwarebazaar", {}).get("enabled", False):
        logger.info("Fetching data from MalwareBazaar...")
        mb_config = config["data_sources"]["malwarebazaar"]

        try:
            # Get API key from config or environment
            mb_api_key = config.get("api_keys", {}).get("malwarebazaar_api_key")
            mb_source = MalwareBazaarDataSource(api_key=mb_api_key)

            malicious_samples = mb_source.fetch_malicious_samples(
                tags=mb_config.get("tags"),
                max_samples=mb_config.get("max_samples", 100)
            )
            all_samples.extend(malicious_samples)
            logger.info(f"Fetched {len(malicious_samples)} samples from MalwareBazaar")

        except Exception as e:
            logger.error(f"MalwareBazaar data source failed: {e}")

    # Hugging Face Data Source
    if config.get("data_sources", {}).get("huggingface", {}).get("enabled", False):
        logger.info("Fetching data from Hugging Face...")
        hf_config = config["data_sources"]["huggingface"]

        try:
            # Get HuggingFace token from config
            hf_token = config.get("api_keys", {}).get("huggingface_token")
            hf_source = HuggingFaceDataSource(token=hf_token)

            benign_samples = hf_source.fetch_benign_samples(
                max_samples=hf_config.get("max_samples", 10000),
                datasets=hf_config.get("datasets")
            )
            all_samples.extend(benign_samples)
            logger.info(f"Fetched {len(benign_samples)} samples from Hugging Face")

        except Exception as e:
            logger.error(f"Hugging Face data source failed: {e}")

    # CVE Feeds Data Source
    if config.get("data_sources", {}).get("cve_feeds", {}).get("enabled", False):
        logger.info("Fetching data from CVE feeds...")
        cve_config = config["data_sources"]["cve_feeds"]

        try:
            cve_source = CVEFeedSource(
                api_key=config.get("api_keys", {}).get("nvd_api_key")
            )

            # Get synthetic exploit patterns
            exploit_samples = cve_source.get_exploit_pattern_samples()
            all_samples.extend(exploit_samples)
            logger.info(f"Generated {len(exploit_samples)} samples from CVE patterns")

        except Exception as e:
            logger.error(f"CVE feeds data source failed: {e}")

    # Additional HuggingFace Datasets (InQuest, dhuynh, cybersixgill)
    if config.get("data_sources", {}).get("additional_hf", {}).get("enabled", False):
        logger.info("Fetching data from additional HuggingFace datasets...")
        additional_config = config["data_sources"]["addiKtional_hf"]

        try:
            # Pass HuggingFace token
            hf_token = config.get("api_keys", {}).get("huggingface_token")
            additional_source = AdditionalHFDatasets(token=hf_token)

            max_per_dataset = additional_config.get("max_samples_per_dataset", 50)

            # Fetch from all additional datasets
            additional_samples = additional_source.fetch_all_datasets(
                max_per_dataset=max_per_dataset
            )

            all_samples.extend(additional_samples)
            logger.info(f"Fetched {len(additional_samples)} samples from additional HF datasets")

        except Exception as e:
            logger.error(f"Additional HF datasets failed: {e}")

    logger.info(f"Total samples collected: {len(all_samples)}")

    # Deduplicate samples
    logger.info("Deduplicating samples...")
    unique_samples = deduplicate_samples(all_samples)

    # Store in database if configured
    db_path = config.get("database", {}).get("path")
    if db_path:
        logger.info(f"Storing samples in database: {db_path}")
        db_manager = DatasetManager(db_path)

        # Get existing hashes to avoid duplicates
        existing_hashes = db_manager.get_existing_hashes()

        # Filter out samples already in database
        new_samples = [
            s for s in unique_samples
            if s.get("content_hash") not in existing_hashes
        ]

        logger.info(f"New samples to add: {len(new_samples)}")

        # Add to database
        if new_samples:
            result = db_manager.add_samples_batch(new_samples)
            logger.info(f"Database insert result: {result}")

        # Print statistics
        stats = db_manager.get_dataset_stats()
        logger.info(f"Database statistics: {stats}")

        db_manager.close()

    # Print statistics
    logger.info("Generating statistics report...")
    stats = DatasetStatistics(unique_samples)
    stats.print_report()

    logger.info(f"Data ingestion completed: {len(unique_samples)} unique samples")

    return unique_samples
