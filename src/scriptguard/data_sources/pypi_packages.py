"""
PyPI Packages Data Source
Fetches benign code samples from top PyPI packages.
"""

import io
import tarfile
import zipfile
import requests
from typing import List, Dict, Optional
from scriptguard.utils.logger import logger


class PyPIDataSource:
    """
    PyPI package integration for fetching benign code samples.

    Note: This is a simplified implementation that fetches from popular packages.
    For production use, consider rate limiting and caching.
    """

    PYPI_API_BASE = "https://pypi.org/pypi"
    TOP_PACKAGES_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize PyPI data source.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ScriptGuard-Training-Pipeline/1.0"
        })

    def fetch_top_packages(self, top_n: int = 1000) -> List[str]:
        """
        Fetch list of top PyPI packages by download count.

        Args:
            top_n: Number of top packages to fetch

        Returns:
            List of package names
        """
        try:
            logger.info(f"Fetching top {top_n} PyPI packages...")
            response = self.session.get(self.TOP_PACKAGES_URL, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if not data or not isinstance(data, dict):
                logger.error("Invalid response format from top packages API")
                return []

            rows = data.get("rows", [])
            if not rows:
                logger.error("No rows in top packages data")
                return []

            packages = [row.get("project") for row in rows if row.get("project")][:top_n]

            logger.info(f"✓ Found {len(packages)} top packages")
            return packages

        except Exception as e:
            logger.error(f"Failed to fetch top packages: {e}")
            return []

    def get_package_info(self, package_name: str) -> Optional[Dict]:
        """
        Get package metadata from PyPI JSON API.

        Args:
            package_name: Name of the package

        Returns:
            Package metadata dict or None
        """
        try:
            url = f"{self.PYPI_API_BASE}/{package_name}/json"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.debug(f"Failed to get info for {package_name}: {e}")
            return None

    def download_source_dist(self, package_name: str, package_info: Dict) -> Optional[bytes]:
        """
        Download source distribution (sdist) for a package.

        Args:
            package_name: Name of the package
            package_info: Package metadata from PyPI API

        Returns:
            Binary content of source distribution or None
        """
        try:
            # Safety check
            if not package_info or not isinstance(package_info, dict):
                logger.debug(f"Invalid package_info for {package_name}")
                return None

            # Get latest version
            latest_version = package_info.get("info", {}).get("version")
            if not latest_version:
                return None

            # Find source distribution URL
            urls = package_info.get("urls", [])
            source_url = None

            for url_info in urls:
                # Prefer .tar.gz source distributions
                if url_info.get("packagetype") == "sdist":
                    source_url = url_info.get("url")
                    break

            if not source_url:
                logger.debug(f"No source dist found for {package_name}")
                return None

            # Download
            logger.debug(f"Downloading source dist for {package_name}...")
            response = self.session.get(source_url, timeout=self.timeout * 2)
            response.raise_for_status()

            return response.content

        except Exception as e:
            logger.debug(f"Failed to download source for {package_name}: {e}")
            return None

    def extract_python_files(
        self,
        archive_content: bytes,
        package_name: str,
        max_files: int = 50
    ) -> List[str]:
        """
        Extract Python files from archive (tar.gz or zip).

        Args:
            archive_content: Binary content of archive
            package_name: Name of the package
            max_files: Maximum number of files to extract

        Returns:
            List of Python file contents
        """
        py_files = []

        try:
            # Try as tar.gz first
            try:
                with tarfile.open(fileobj=io.BytesIO(archive_content), mode='r:*') as tar:
                    members = [m for m in tar.getmembers() if m.name.endswith('.py') and m.isfile()]

                    for member in members[:max_files]:
                        try:
                            file_obj = tar.extractfile(member)
                            if file_obj:
                                content = file_obj.read().decode('utf-8', errors='ignore')
                                if len(content.strip()) > 100:  # Skip tiny files
                                    py_files.append(content)
                        except Exception:
                            continue

            except tarfile.ReadError:
                # Try as zip
                with zipfile.ZipFile(io.BytesIO(archive_content)) as zf:
                    py_members = [m for m in zf.namelist() if m.endswith('.py')]

                    for member_name in py_members[:max_files]:
                        try:
                            content = zf.read(member_name).decode('utf-8', errors='ignore')
                            if len(content.strip()) > 100:
                                py_files.append(content)
                        except Exception:
                            continue

        except Exception as e:
            logger.debug(f"Failed to extract files from {package_name}: {e}")

        return py_files

    def fetch_samples(
        self,
        top_n: int = 1000,
        max_files_per_package: int = 50,
        max_total_samples: int = 5000
    ) -> List[Dict]:
        """
        Fetch benign code samples from top PyPI packages.

        Args:
            top_n: Number of top packages to process
            max_files_per_package: Maximum Python files per package
            max_total_samples: Maximum total samples to return

        Returns:
            List of code sample dicts with 'content', 'label', 'source', 'metadata'
        """
        logger.info(f"Starting PyPI data collection (top {top_n} packages)...")

        packages = self.fetch_top_packages(top_n)
        if not packages:
            logger.warning("No packages fetched from top list")
            return []

        samples = []
        packages_processed = 0
        packages_successful = 0

        for package_name in packages:
            if len(samples) >= max_total_samples:
                logger.info(f"Reached maximum samples limit ({max_total_samples})")
                break

            packages_processed += 1

            # Get package info
            package_info = self.get_package_info(package_name)
            if not package_info:
                continue

            # Download source distribution
            source_content = self.download_source_dist(package_name, package_info)
            if not source_content:
                continue

            # Extract Python files
            py_files = self.extract_python_files(
                source_content,
                package_name,
                max_files=max_files_per_package
            )

            if not py_files:
                continue

            packages_successful += 1

            # Convert to samples
            for i, content in enumerate(py_files):
                # Safe metadata extraction
                info = package_info.get("info", {}) if package_info else {}
                version = info.get("version", "unknown") if info else "unknown"
                summary = info.get("summary", "") if info else ""
                description = summary[:200] if summary else ""

                samples.append({
                    "content": content,
                    "label": "benign",
                    "source": f"pypi_{package_name}",
                    "metadata": {
                        "package_name": package_name,
                        "file_index": i,
                        "package_version": version,
                        "package_description": description
                    }
                })

                if len(samples) >= max_total_samples:
                    break

            # Log progress every 10 packages
            if packages_processed % 10 == 0:
                logger.info(
                    f"Progress: {packages_processed}/{len(packages)} packages processed, "
                    f"{packages_successful} successful, {len(samples)} samples collected"
                )

        logger.info(
            f"✓ PyPI collection complete: {packages_processed} packages processed, "
            f"{packages_successful} successful, {len(samples)} samples collected"
        )

        return samples


def fetch_pypi_packages(
    top_n: int = 1000,
    max_files_per_package: int = 50,
    max_samples: int = 5000,
    timeout: int = 30
) -> List[Dict]:
    """
    Convenience function to fetch PyPI package samples.

    Args:
        top_n: Number of top packages to process
        max_files_per_package: Max Python files per package
        max_samples: Maximum total samples
        timeout: Request timeout in seconds

    Returns:
        List of code sample dicts
    """
    source = PyPIDataSource(timeout=timeout)
    return source.fetch_samples(
        top_n=top_n,
        max_files_per_package=max_files_per_package,
        max_total_samples=max_samples
    )
