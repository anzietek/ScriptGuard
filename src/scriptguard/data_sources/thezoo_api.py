"""
TheZoo Data Source
Fetches malware samples from TheZoo GitHub repository.
"""

from scriptguard.utils.logger import logger
from scriptguard.utils.archive_extractor import extract_scripts_from_archive
from scriptguard.utils.retry_utils import retry_with_backoff, RetryStats
import requests
from typing import List, Dict, Optional
from datetime import datetime
import time

class TheZooDataSource:
    """TheZoo GitHub integration for malware script samples."""

    GITHUB_API = "https://api.github.com"
    THEZOO_REPO = "ytisf/theZoo"

    def __init__(self, github_token: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize TheZoo data source.

        Args:
            github_token: GitHub token for higher rate limits
            config: Configuration dictionary with retry/timeout settings
        """
        self.github_token = github_token
        self.config = config or {}
        self.headers = {"Accept": "application/vnd.github+json"}
        if github_token:
            self.headers["Authorization"] = f"Bearer {github_token}"
            logger.info("TheZoo: GitHub token configured")

        # Read configuration with fallback defaults
        source_config = self.config.get("data_sources", {}).get("thezoo", {})
        self.timeout = source_config.get("timeout", 30)
        self.max_retries = source_config.get("max_retries", 3)
        self.retry_backoff_factor = source_config.get("retry_backoff_factor", 2.0)

        # Initialize retry statistics tracking
        self.retry_stats = RetryStats()

    def _make_request(self, url: str) -> Optional[Dict]:
        """Make GET request to GitHub API with retry logic."""
        @retry_with_backoff(
            max_retries=self.max_retries,
            backoff_factor=self.retry_backoff_factor,
            initial_delay=1.0,
            exceptions=(requests.exceptions.Timeout, requests.exceptions.RequestException),
            on_retry=lambda e, attempt: setattr(self, '_last_retry_count', attempt)
        )
        def _do_request():
            response = requests.get(url, headers=self.headers, timeout=self.timeout)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                logger.error("GitHub rate limit exceeded (TheZoo uses GitHub)")
                raise Exception("Rate limit exceeded")
            elif response.status_code == 404:
                return None  # Not found, don't retry
            else:
                raise Exception(f"HTTP {response.status_code}")

        try:
            self._last_retry_count = 0
            result = _do_request()
            self.retry_stats.record_attempt("api_request", True, self._last_retry_count)
            return result
        except Exception as e:
            retry_count = getattr(self, '_last_retry_count', self.max_retries)
            self.retry_stats.record_attempt("api_request", False, retry_count)
            logger.error(f"TheZoo request failed after retries: {e}")
            return None

    def _search_scripts_in_path(self, path: str, extensions: List[str], max_samples: int, depth: int = 0) -> List[Dict]:
        """Search for script files in repository path."""
        if depth > 2 or max_samples <= 0:
            return []

        samples = []
        url = f"{self.GITHUB_API}/repos/{self.THEZOO_REPO}/contents/{path}"

        data = self._make_request(url)
        if not data:
            return samples

        if isinstance(data, dict):
            data = [data]

        for item in data:
            if len(samples) >= max_samples:
                break

            if item.get("type") == "file":
                name = item["name"].lower()
                if any(name.endswith(ext) for ext in extensions) or name.endswith('.zip') or name.endswith('.tar.gz'):
                    if not name.startswith('__'):
                        scripts = self._download_file(item["download_url"], item["name"])
                        for script_name, script_content in scripts:
                            if len(script_content) > 100:
                                samples.append({
                                    "content": script_content,
                                    "label": "malicious",
                                    "source": "thezoo",
                                    "url": item["html_url"],
                                    "metadata": {
                                        "file_name": script_name,
                                        "original_file": item["name"],
                                        "path": item["path"],
                                        "size": len(script_content),
                                        "fetched_at": datetime.now().isoformat()
                                    }
                                })
                                if len(samples) >= max_samples:
                                    break
                        time.sleep(0.2)  # Brief rate limiting

            elif item.get("type") == "dir" and len(samples) < max_samples:
                samples.extend(self._search_scripts_in_path(item["path"], extensions, max_samples - len(samples), depth + 1))

        return samples

    def _download_file(self, url: str, filename: str) -> List[tuple]:
        """Download file content from GitHub and extract if archive."""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                content_bytes = response.content

                if len(content_bytes) < 50:
                    return []

                scripts = extract_scripts_from_archive(content_bytes, filename)
                if scripts:
                    filtered = []
                    for name, content in scripts:
                        if '\x00' not in content and len(content) > 50:
                            filtered.append((name, content))
                    return filtered

                if filename.lower().endswith(('.zip', '.tar', '.gz', '.7z', '.rar')):
                    return []

                try:
                    text = content_bytes.decode('utf-8', errors='ignore')
                    if len(text) > 50 and '\x00' not in text:
                        return [(filename, text)]
                except:
                    pass
        except Exception as e:
            logger.error(f"Download failed: {e}")
        return []

    def fetch_malicious_samples(
        self,
        script_types: Optional[List[str]] = None,
        max_samples: int = 50
    ) -> List[Dict]:
        """
        Fetch malicious script samples from TheZoo.

        Args:
            script_types: List of script extensions to fetch
            max_samples: Maximum samples to fetch

        Returns:
            List of samples with content and metadata
        """
        if script_types is None:
            script_types = [".py", ".ps1", ".js", ".vbs", ".sh", ".bat"]

        logger.info(f"Fetching up to {max_samples} samples from TheZoo")

        all_samples = []

        root_url = f"{self.GITHUB_API}/repos/{self.THEZOO_REPO}/contents"
        root_data = self._make_request(root_url)

        if not root_data:
            logger.error("Could not access TheZoo root")
            return []

        malware_dir = None
        for item in root_data:
            if item.get("type") == "dir" and item.get("name") == "malware":
                malware_dir = item
                break

        if malware_dir:
            logger.info(f"Exploring malware directory: {malware_dir['name']}")
            samples = self._search_scripts_in_path(malware_dir["path"], script_types, max_samples, depth=0)
            all_samples.extend(samples)

        if len(all_samples) < max_samples:
            for item in root_data[:20]:
                if len(all_samples) >= max_samples:
                    break
                if item.get("type") == "dir" and item.get("name") != "malware":
                    logger.info(f"Exploring directory: {item['name']}")
                    samples = self._search_scripts_in_path(item["path"], script_types, max_samples - len(all_samples), depth=0)
                    all_samples.extend(samples)
                    time.sleep(0.5)  # Rate limiting between directories

        logger.info(f"Fetched {len(all_samples)} samples from TheZoo")
        return all_samples[:max_samples]
