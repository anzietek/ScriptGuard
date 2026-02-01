"""
ScriptGuard RAG (Retrieval-Augmented Generation) Module
Provides Qdrant vector store for CVE and malware pattern knowledge.
"""

from .qdrant_store import QdrantStore, bootstrap_cve_data

__all__ = [
    "QdrantStore",
    "bootstrap_cve_data",
]
