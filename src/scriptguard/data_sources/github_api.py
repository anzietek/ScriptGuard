"""
GitHub API Data Source
Fetches malicious and benign code samples from GitHub repositories.
"""

import time
from scriptguard.utils.logger import logger
from typing import List, Dict, Optional
import requests
from datetime import datetime

class GitHubDataSource:
    """GitHub API integration with rate limiting and intelligent searching."""

    BASE_URL = "https://api.github.com"

    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize GitHub data source.

        Args:
            api_token: GitHub Personal Access Token (optional but recommended)
        """
        self.api_token = api_token
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"

        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    def _check_rate_limit(self) -> bool:
        """
        Check if we have remaining API calls.

        Returns:
            bool: True if we can make requests, False if rate limited
        """
        if self.rate_limit_remaining is not None and self.rate_limit_remaining <= 1:
            if self.rate_limit_reset:
                wait_time = self.rate_limit_reset - time.time()
                if wait_time > 0:
                    logger.warning(f"Rate limit reached. Waiting {wait_time:.0f} seconds...")
                    time.sleep(wait_time + 1)
                    return True
            return False
        return True

    def _update_rate_limit(self, response: requests.Response):
        """Update rate limit info from response headers."""
        if "X-RateLimit-Remaining" in response.headers:
            self.rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in response.headers:
            self.rate_limit_reset = int(response.headers["X-RateLimit-Reset"])

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make authenticated request with rate limiting.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            JSON response or None on error
        """
        if not self._check_rate_limit():
            logger.error("Rate limit exceeded and cannot wait")
            return None

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            self._update_rate_limit(response)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                logger.error("API rate limit exceeded or access forbidden")
                return None
            else:
                logger.error(f"GitHub API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def search_repositories(
        self,
        query: str,
        language: str = "Python",
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search for repositories matching query.

        Args:
            query: Search query (e.g., "keylogger", "ransomware")
            language: Programming language filter
            max_results: Maximum number of results

        Returns:
            List of repository data dictionaries
        """
        url = f"{self.BASE_URL}/search/repositories"
        params = {
            "q": f"{query} language:{language}",
            "sort": "stars",
            "order": "desc",
            "per_page": min(max_results, 100)
        }

        logger.info(f"Searching repositories: {query} (language: {language})")
        data = self._make_request(url, params)

        if not data or "items" not in data:
            return []

        return [
            {
                "name": repo["full_name"],
                "url": repo["html_url"],
                "api_url": repo["url"],
                "description": repo.get("description", ""),
                "stars": repo["stargazers_count"],
                "language": repo.get("language", "")
            }
            for repo in data["items"][:max_results]
        ]

    def search_code_snippets(
        self,
        query: str,
        language: str = "Python",
        max_results: int = 30
    ) -> List[Dict]:
        """
        Search for code snippets matching query.

        Args:
            query: Search query (e.g., "os.system exec")
            language: Programming language filter
            max_results: Maximum number of results

        Returns:
            List of code snippet data dictionaries
        """
        url = f"{self.BASE_URL}/search/code"
        params = {
            "q": f"{query} language:{language}",
            "per_page": min(max_results, 100)
        }

        logger.info(f"Searching code: {query} (language: {language})")
        data = self._make_request(url, params)

        if not data or "items" not in data:
            return []

        results = []
        for item in data["items"][:max_results]:
            content = self.fetch_file_content(item["url"])
            if content:
                results.append({
                    "name": item["name"],
                    "path": item["path"],
                    "url": item["html_url"],
                    "repository": item["repository"]["full_name"],
                    "content": content
                })

        return results

    def fetch_file_content(self, file_api_url: str) -> Optional[str]:
        """
        Fetch raw file content from GitHub.

        Args:
            file_api_url: GitHub API URL for file

        Returns:
            File content as string or None
        """
        data = self._make_request(file_api_url)
        if not data or "content" not in data:
            return None

        import base64
        try:
            content = base64.b64decode(data["content"]).decode("utf-8")
            return content
        except Exception as e:
            logger.error(f"Failed to decode file content: {e}")
            return None

    def fetch_malicious_samples(
        self,
        keywords: Optional[List[str]] = None,
        max_per_keyword: int = 20
    ) -> List[Dict]:
        """
        Fetch potentially malicious code samples.

        Args:
            keywords: List of malicious keywords to search
            max_per_keyword: Maximum samples per keyword

        Returns:
            List of samples with metadata
        """
        if keywords is None:
            keywords = [
                "reverse-shell python",
                "keylogger python",
                "ransomware python",
                "backdoor python",
                "credential stealer python",
                "port scanner python",
                "exploit python",
                "payload python"
            ]

        all_samples = []

        for keyword in keywords:
            logger.info(f"Fetching malicious samples for: {keyword}")
            snippets = self.search_code_snippets(keyword, max_results=max_per_keyword)

            for snippet in snippets:
                all_samples.append({
                    "content": snippet["content"],
                    "label": "malicious",
                    "source": "github",
                    "url": snippet["url"],
                    "metadata": {
                        "keyword": keyword,
                        "repository": snippet["repository"],
                        "path": snippet["path"],
                        "fetched_at": datetime.now().isoformat()
                    }
                })

            # Rate limiting: wait between keywords
            time.sleep(2)

        logger.info(f"Fetched {len(all_samples)} malicious samples from GitHub")
        return all_samples

    def fetch_benign_samples(
        self,
        popular_repos: Optional[List[str]] = None,
        max_files_per_repo: int = 50
    ) -> List[Dict]:
        """
        Fetch benign code samples from popular repositories.

        Args:
            popular_repos: List of "owner/repo" strings
            max_files_per_repo: Maximum Python files per repository

        Returns:
            List of samples with metadata
        """
        if popular_repos is None:
            popular_repos = [
                "django/django",
                "pallets/flask",
                "psf/requests",
                "python/cpython",
                "scikit-learn/scikit-learn",
                "pandas-dev/pandas",
                "pytorch/pytorch",
                "tensorflow/tensorflow"
            ]

        all_samples = []

        for repo in popular_repos:
            logger.info(f"Fetching benign samples from: {repo}")

            # Search for Python files in repo
            url = f"{self.BASE_URL}/search/code"
            params = {
                "q": f"repo:{repo} extension:py",
                "per_page": max_files_per_repo
            }

            data = self._make_request(url, params)
            if not data or "items" not in data:
                continue

            for item in data["items"][:max_files_per_repo]:
                content = self.fetch_file_content(item["url"])
                if content and len(content) > 100:  # Skip very small files
                    all_samples.append({
                        "content": content,
                        "label": "benign",
                        "source": "github",
                        "url": item["html_url"],
                        "metadata": {
                            "repository": repo,
                            "path": item["path"],
                            "fetched_at": datetime.now().isoformat()
                        }
                    })

            time.sleep(2)  # Rate limiting

        logger.info(f"Fetched {len(all_samples)} benign samples from GitHub")
        return all_samples
