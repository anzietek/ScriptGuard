"""
ScriptGuard Data Sources Module
Provides integrations with various data sources for malicious/benign code samples.
"""

from .github_api import GitHubDataSource
from .malwarebazaar_api import MalwareBazaarDataSource
from .huggingface_datasets import HuggingFaceDataSource
from .cve_feeds import CVEFeedSource

__all__ = [
    "GitHubDataSource",
    "MalwareBazaarDataSource",
    "HuggingFaceDataSource",
    "CVEFeedSource",
]
