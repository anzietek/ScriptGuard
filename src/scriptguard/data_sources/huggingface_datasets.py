"""
Hugging Face Datasets Data Source
Loads benign code samples from Hugging Face datasets.
"""

from scriptguard.utils.logger import logger
from scriptguard.utils.retry_utils import retry_with_backoff, RetryStats
from typing import List, Dict, Optional
from datetime import datetime
import os

class HuggingFaceDataSource:
    """Hugging Face datasets integration for benign code samples."""

    def __init__(self, token: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize Hugging Face data source.

        Args:
            token: HuggingFace token for accessing gated datasets
            config: Configuration dictionary with retry/timeout settings
        """
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        self.config = config or {}

        # Read configuration with fallback defaults
        source_config = self.config.get("data_sources", {}).get("huggingface", {})
        self.timeout = source_config.get("timeout", 120)
        self.max_retries = source_config.get("max_retries", 3)
        self.retry_backoff_factor = source_config.get("retry_backoff_factor", 2.0)

        # Initialize retry statistics tracking
        self.retry_stats = RetryStats()

        try:
            from datasets import load_dataset
            self.load_dataset = load_dataset
            self.available = True

            if self.token:
                logger.info("HuggingFace token configured for gated datasets")
            else:
                logger.warning("No HuggingFace token - gated datasets will fail")

        except ImportError:
            logger.warning("datasets library not installed. Hugging Face source unavailable.")
            self.available = False

    def load_benign_code_dataset(
        self,
        dataset_name: str,
        language: str = "Python",
        max_samples: int = 1000,
        split: str = "train"
    ) -> List[Dict]:
        """
        Load benign code from a Hugging Face dataset.

        Args:
            dataset_name: Dataset identifier (e.g., "codeparrot/github-code")
            language: Programming language filter
            max_samples: Maximum number of samples
            split: Dataset split (train/validation/test)

        Returns:
            List of code samples with metadata
        """
        if not self.available:
            logger.error("Datasets library not available")
            return []

        logger.info(f"Loading dataset: {dataset_name} (language: {language})")

        # Security: trust_remote_code disabled by default (supply-chain risk)
        trust_remote = os.getenv("SCRIPTGUARD_TRUST_DATASET_CODE", "false").lower() == "true"

        if trust_remote:
            logger.warning(
                f"⚠️  SECURITY WARNING: trust_remote_code=True for dataset '{dataset_name}'. "
                "This allows arbitrary code execution from the dataset repository. "
                "Only enable for trusted datasets."
            )

        @retry_with_backoff(
            max_retries=self.max_retries,
            backoff_factor=self.retry_backoff_factor,
            initial_delay=2.0,  # Longer delay for large datasets
            exceptions=(Exception,),  # Catch all HF exceptions
            on_retry=lambda e, attempt: setattr(self, '_last_retry_count', attempt)
        )
        def _load_dataset():
            # Load dataset with streaming for large datasets
            return self.load_dataset(
                dataset_name,
                split=split,
                streaming=True,
                trust_remote_code=trust_remote,
                token=self.token  # Add token for gated datasets
            )

        try:
            self._last_retry_count = 0
            dataset = _load_dataset()
            self.retry_stats.record_attempt("load_dataset", True, self._last_retry_count)

            samples = []
            count = 0

            for item in dataset:
                # Different datasets have different structures
                content = None
                repo_name = None
                path = None

                # Try common field names
                if "code" in item:
                    content = item["code"]
                elif "content" in item:
                    content = item["content"]
                elif "text" in item:
                    content = item["text"]

                # Get language if available
                item_language = item.get("language", item.get("lang", ""))

                # Filter by language
                if language and item_language.lower() != language.lower():
                    continue

                # Get repository and path info if available
                if "repo_name" in item:
                    repo_name = item["repo_name"]
                if "path" in item:
                    path = item["path"]

                if content and len(content) > 100:  # Skip very small files
                    samples.append({
                        "content": content,
                        "label": "benign",
                        "source": f"huggingface:{dataset_name}",
                        "url": f"https://huggingface.co/datasets/{dataset_name}",
                        "metadata": {
                            "dataset": dataset_name,
                            "language": item_language,
                            "repo_name": repo_name,
                            "path": path,
                            "fetched_at": datetime.now().isoformat()
                        }
                    })

                    count += 1
                    if count >= max_samples:
                        break

            logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
            return samples

        except Exception as e:
            retry_count = getattr(self, '_last_retry_count', self.max_retries)
            self.retry_stats.record_attempt("load_dataset", False, retry_count)
            logger.error(f"Failed to load dataset {dataset_name} after retries: {e}")
            return []

    def load_the_stack_dataset(
        self,
        language: str = "Python",
        max_samples: int = 5000
    ) -> List[Dict]:
        """
        Load samples from The Stack dataset (deduplicated version).

        Args:
            language: Programming language
            max_samples: Maximum number of samples

        Returns:
            List of code samples with metadata
        """
        logger.info(f"Loading The Stack dataset (language: {language})")

        try:
            # The Stack has language-specific subsets
            dataset_name = "bigcode/the-stack-dedup"
            subset = f"data/{language.lower()}"

            dataset = self.load_dataset(
                dataset_name,
                data_dir=subset,
                split="train",
                streaming=True,
                token=self.token  # Add token for gated dataset
            )

            samples = []
            count = 0

            for item in dataset:
                content = item.get("content", item.get("text", ""))

                if content and len(content) > 100:
                    # Validate Python syntax
                    from scriptguard.utils.file_validator import validate_python_content
                    is_valid, metadata = validate_python_content(content, source_label="the-stack")

                    if not is_valid:
                        continue

                    samples.append({
                        "content": content,
                        "label": "benign",
                        "source": f"huggingface:the-stack-dedup",
                        "url": f"https://huggingface.co/datasets/{dataset_name}",
                        "metadata": {
                            "dataset": "the-stack-dedup",
                            "language": language,
                            "size": item.get("size"),
                            "license": item.get("license"),
                            "fetched_at": datetime.now().isoformat(),
                            "validation": metadata
                        }
                    })

                    count += 1
                    if count >= max_samples:
                        break

            logger.info(f"Loaded {len(samples)} samples from The Stack")
            return samples

        except Exception as e:
            logger.error(f"Failed to load The Stack dataset: {e}")
            return []

    def load_codeparrot_dataset(
        self,
        max_samples: int = 5000
    ) -> List[Dict]:
        """
        Load samples from CodeParrot GitHub code dataset.

        Args:
            max_samples: Maximum number of samples

        Returns:
            List of code samples with metadata
        """
        return self.load_benign_code_dataset(
            dataset_name="codeparrot/github-code",
            language="Python",
            max_samples=max_samples
        )

    def fetch_benign_samples(
        self,
        max_samples: int = 10000,
        datasets: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fetch benign code samples from multiple Hugging Face datasets.

        Args:
            max_samples: Maximum total samples
            datasets: List of dataset names to use

        Returns:
            List of samples with metadata
        """
        if not self.available:
            logger.error("Datasets library not available")
            return []

        if datasets is None:
            datasets = [
                "codeparrot/github-code",
                "bigcode/the-stack-dedup"
            ]

        all_samples = []
        samples_per_dataset = max(1, max_samples // len(datasets))

        for dataset_name in datasets:
            logger.info(f"Fetching samples from: {dataset_name}")

            if "the-stack" in dataset_name.lower():
                samples = self.load_the_stack_dataset(
                    language="Python",
                    max_samples=samples_per_dataset
                )
            else:
                samples = self.load_benign_code_dataset(
                    dataset_name=dataset_name,
                    language="Python",
                    max_samples=samples_per_dataset
                )

            all_samples.extend(samples)

            if len(all_samples) >= max_samples:
                break

        # Trim to max_samples
        all_samples = all_samples[:max_samples]

        logger.info(f"Fetched {len(all_samples)} benign samples from Hugging Face")
        return all_samples
