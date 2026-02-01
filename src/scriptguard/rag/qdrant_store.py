import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantStore:
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 6333, 
        collection_name: str = "vulnerabilities",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(model_name)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.encoder.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )

    def upsert_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]):
        """
        vulnerabilities: List of dicts with 'id', 'description', and metadata.
        """
        points = []
        for vuln in vulnerabilities:
            vector = self.encoder.encode(vuln["description"]).tolist()
            points.append(
                models.PointStruct(
                    id=vuln["id"],
                    vector=vector,
                    payload=vuln
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} points to Qdrant.")

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        query_vector = self.encoder.encode(query).tolist()
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return [hit.payload for hit in search_result]

if __name__ == "__main__":
    # Example usage / bootstrap
    store = QdrantStore()
    sample_data = [
        {
            "id": 1,
            "description": "SQL Injection in login form via 'admin' parameter.",
            "cve": "CVE-2023-XXXX",
            "severity": "High"
        },
        {
            "id": 2,
            "description": "Remote Code Execution (RCE) via unsafe deserialization.",
            "cve": "CVE-2024-YYYY",
            "severity": "Critical"
        }
    ]
    store.upsert_vulnerabilities(sample_data)
    print(store.search("deserialization exploit"))
