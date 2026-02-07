"""
Enhanced Qdrant RAG Store
Improved vector store for CVEs, vulnerabilities, and malware patterns.
"""

import os
import logging
from scriptguard.utils.logger import logger
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer
import hashlib

class QdrantStore:
    """Enhanced Qdrant store for RAG (Retrieval-Augmented Generation)."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = "malware_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        use_https: bool = False
    ):
        """
        Initialize Qdrant RAG store.

        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection
            embedding_model: Sentence transformer model name
            api_key: Optional API key for Qdrant Cloud
            use_https: Use HTTPS connection
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")

        logger.info(f"Initializing Qdrant client: {self.host}:{self.port}")
        if self.api_key:
            logger.info("Using API key authentication")

        # Initialize client
        if self.api_key:
            self.client = QdrantClient(
                url=f"{'https' if use_https else 'http'}://{self.host}:{self.port}",
                api_key=self.api_key
            )
        else:
            self.client = QdrantClient(host=self.host, port=self.port)

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.encoder = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure collection exists with proper configuration."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=10000
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=100
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")

            # Create payload indexes for faster filtering
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="cve_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="severity",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="type",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
            except UnexpectedResponse:
                # Indexes might already exist
                pass

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def _generate_id(self, content: str) -> str:
        """Generate deterministic ID from content."""
        return hashlib.md5(content.encode()).hexdigest()

    def upsert_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]):
        """
        Upsert vulnerability data.

        Args:
            vulnerabilities: List of vulnerability dictionaries with:
                - description: str (required)
                - cve_id: str (optional)
                - severity: str (optional)
                - type: str (optional)
                - pattern: str (optional)
                - metadata: dict (optional)
        """
        if not vulnerabilities:
            return

        points = []
        for vuln in vulnerabilities:
            description = vuln.get("description", "")
            if not description:
                logger.warning("Skipping vulnerability without description")
                continue

            # Generate embedding
            vector = self.encoder.encode(description).tolist()

            # Generate ID
            vuln_id = vuln.get("id") or self._generate_id(description)

            # Prepare payload
            payload = {
                "description": description,
                "cve_id": vuln.get("cve_id", ""),
                "severity": vuln.get("severity", "UNKNOWN"),
                "type": vuln.get("type", "vulnerability"),
                "pattern": vuln.get("pattern", ""),
                "metadata": vuln.get("metadata", {})
            }

            points.append(
                models.PointStruct(
                    id=vuln_id,
                    vector=vector,
                    payload=payload
                )
            )

        # Batch upsert
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} vulnerability records")
        except Exception as e:
            logger.error(f"Failed to upsert vulnerabilities: {e}")
            raise

    def upsert_malware_patterns(self, patterns: List[Dict[str, Any]]):
        """
        Upsert malware pattern data.

        Args:
            patterns: List of pattern dictionaries
        """
        for pattern in patterns:
            pattern["type"] = "malware_pattern"

        self.upsert_vulnerabilities(patterns)

    def search(
        self,
        query: str,
        limit: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant vulnerabilities/patterns.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_conditions: Optional Qdrant filters
            score_threshold: Minimum similarity score

        Returns:
            List of matching records with scores
        """
        # Generate query embedding
        query_vector = self.encoder.encode(query).tolist()

        # Build filter if provided
        search_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            search_filter = models.Filter(must=conditions)

        # Search using search method (compatible API)
        try:
            from qdrant_client.models import SearchRequest

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
                score_threshold=score_threshold,
                with_payload=True
            )

            results = []
            for hit in search_result:
                results.append({
                    "score": hit.score,
                    "payload": hit.payload,
                    "id": hit.id
                })

            return results

        except AttributeError:
            # Try alternative method for newer Qdrant versions
            try:
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit,
                    query_filter=search_filter,
                    score_threshold=score_threshold,
                    with_payload=True
                )

                results = []
                for hit in search_result.points:
                    results.append({
                        "score": hit.score,
                        "payload": hit.payload,
                        "id": hit.id
                    })

                return results
            except Exception as e:
                logger.error(f"Query points failed: {e}")
                return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by_cve(self, cve_id: str) -> List[Dict[str, Any]]:
        """Search for specific CVE."""
        return self.search(
            query=cve_id,
            filter_conditions={"cve_id": cve_id},
            limit=10
        )

    def search_by_severity(self, query: str, severity: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search filtered by severity."""
        return self.search(
            query=query,
            filter_conditions={"severity": severity},
            limit=limit
        )

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else 384,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def delete_by_id(self, point_ids: List[str]):
        """Delete points by IDs."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            logger.info(f"Deleted {len(point_ids)} points")
        except Exception as e:
            logger.error(f"Failed to delete points: {e}")

    def clear_collection(self):
        """Clear all data from collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info("Collection cleared and recreated")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")

def bootstrap_cve_data(store: QdrantStore):
    """Bootstrap store with common CVE patterns."""
    logger.info("Bootstrapping CVE knowledge base...")

    cve_data = [
        {
            "cve_id": "CVE-2021-44228",
            "description": "Log4Shell Remote Code Execution via JNDI LDAP injection in Log4j",
            "severity": "CRITICAL",
            "pattern": "${jndi:ldap://",
            "type": "vulnerability"
        },
        {
            "cve_id": "CVE-2014-6271",
            "description": "Shellshock Bash command injection vulnerability",
            "severity": "HIGH",
            "pattern": "() { :; };",
            "type": "vulnerability"
        },
        {
            "description": "Python code execution via eval() with user input",
            "severity": "HIGH",
            "pattern": "eval(input(",
            "type": "malware_pattern"
        },
        {
            "description": "Python reverse shell using socket module",
            "severity": "HIGH",
            "pattern": "socket.socket()...connect(",
            "type": "malware_pattern"
        },
        {
            "description": "Command injection via os.system with user input",
            "severity": "HIGH",
            "pattern": "os.system(input(",
            "type": "malware_pattern"
        },
        {
            "description": "Unsafe pickle deserialization leading to RCE",
            "severity": "HIGH",
            "pattern": "pickle.loads(",
            "type": "malware_pattern"
        },
        {
            "description": "Base64 encoded malicious payload",
            "severity": "MEDIUM",
            "pattern": "base64.b64decode(",
            "type": "malware_pattern"
        }
    ]

    store.upsert_vulnerabilities(cve_data)
    logger.info(f"Bootstrapped {len(cve_data)} CVE/pattern records")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    store = QdrantStore()
    bootstrap_cve_data(store)

    # Test search
    results = store.search("remote code execution python", limit=3)
    print("\nSearch results:")
    for r in results:
        print(f"Score: {r['score']:.2f} - {r['payload']['description']}")

    # Get info
    info = store.get_collection_info()
    print(f"\nCollection info: {info}")
