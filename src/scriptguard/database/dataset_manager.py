"""
PostgreSQL Dataset Manager
CRUD operations, versioning, and statistics for code sample database.
"""

import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from psycopg2.extras import execute_values, Json

from .db_schema_postgres import get_connection, return_connection, refresh_statistics
from .deduplication import compute_hash

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages code sample database with PostgreSQL and versioning support."""

    def __init__(self):
        """Initialize dataset manager with PostgreSQL connection pool."""
        logger.info("Dataset manager initialized with PostgreSQL")

    def add_sample(
        self,
        content: str,
        label: str,
        source: str,
        url: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a single sample to database.

        Args:
            content: Code content
            label: "malicious" or "benign"
            source: Source identifier
            url: Optional source URL
            metadata: Optional metadata dictionary

        Returns:
            True if added successfully, False if duplicate
        """
        content_hash = compute_hash(content)
        conn = get_connection()

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO samples (content_hash, content, label, source, url, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
                """,
                (
                    content_hash,
                    content,
                    label,
                    source,
                    url,
                    Json(metadata) if metadata else Json({})
                )
            )
            result = cursor.fetchone()
            conn.commit()

            if result:
                logger.debug(f"Added sample: {content_hash[:16]}...")
                return True
            else:
                logger.debug(f"Duplicate sample skipped: {content_hash[:16]}...")
                return False

        except Exception as e:
            logger.error(f"Failed to add sample: {e}")
            conn.rollback()
            return False
        finally:
            cursor.close()
            return_connection(conn)

    def add_samples_batch(self, samples: List[Dict]) -> Dict[str, int]:
        """
        Add multiple samples in batch (optimized for PostgreSQL).

        Args:
            samples: List of sample dictionaries

        Returns:
            Statistics dictionary with counts
        """
        if not samples:
            return {"added": 0, "duplicates": 0, "total": 0}

        conn = get_connection()
        added = 0
        duplicates = 0

        try:
            cursor = conn.cursor()

            # Prepare batch data
            batch_data = []
            for sample in samples:
                content_hash = compute_hash(sample["content"])
                batch_data.append((
                    content_hash,
                    sample["content"],
                    sample["label"],
                    sample["source"],
                    sample.get("url"),
                    Json(sample.get("metadata", {}))
                ))

            # Use execute_values for efficient batch insert
            execute_values(
                cursor,
                """
                INSERT INTO samples (content_hash, content, label, source, url, metadata)
                VALUES %s
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
                """,
                batch_data,
                page_size=1000
            )

            added = cursor.rowcount
            duplicates = len(samples) - added
            conn.commit()

            logger.info(f"Batch insert: {added} added, {duplicates} duplicates skipped")

        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            conn.rollback()
        finally:
            cursor.close()
            return_connection(conn)

        return {
            "added": added,
            "duplicates": duplicates,
            "total": len(samples)
        }

    def get_all_samples(
        self,
        label: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict]:
        """
        Retrieve samples from database with pagination.

        Args:
            label: Optional filter by label
            limit: Maximum number of samples to return
            offset: Number of samples to skip

        Returns:
            List of sample dictionaries
        """
        conn = get_connection()
        samples = []

        try:
            cursor = conn.cursor()

            if label:
                query = """
                    SELECT id, content, label, source, url, metadata, created_at
                    FROM samples
                    WHERE label = %s
                    ORDER BY created_at DESC
                """
                params = [label]
            else:
                query = """
                    SELECT id, content, label, source, url, metadata, created_at
                    FROM samples
                    ORDER BY created_at DESC
                """
                params = []

            if limit:
                query += " LIMIT %s OFFSET %s"
                params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                samples.append({
                    "id": row["id"],
                    "content": row["content"],
                    "label": row["label"],
                    "source": row["source"],
                    "url": row["url"],
                    "metadata": row["metadata"] or {},
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None
                })

        except Exception as e:
            logger.error(f"Failed to retrieve samples: {e}")
        finally:
            cursor.close()
            return_connection(conn)

        return samples

    def get_sample_by_id(self, sample_id: int) -> Optional[Dict]:
        """Get sample by ID."""
        conn = get_connection()

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, content, label, source, url, metadata, created_at
                FROM samples
                WHERE id = %s
                """,
                (sample_id,)
            )
            row = cursor.fetchone()

            if row:
                return {
                    "id": row["id"],
                    "content": row["content"],
                    "label": row["label"],
                    "source": row["source"],
                    "url": row["url"],
                    "metadata": row["metadata"] or {},
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None
                }

        except Exception as e:
            logger.error(f"Failed to retrieve sample {sample_id}: {e}")
        finally:
            cursor.close()
            return_connection(conn)

        return None

    def get_existing_hashes(self) -> set:
        """Get all existing content hashes."""
        conn = get_connection()
        hashes = set()

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT content_hash FROM samples")
            rows = cursor.fetchall()
            hashes = {row["content_hash"] for row in rows}

        except Exception as e:
            logger.error(f"Failed to retrieve hashes: {e}")
        finally:
            cursor.close()
            return_connection(conn)

        return hashes

    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics (using materialized view if available)."""
        conn = get_connection()
        stats = {}

        try:
            cursor = conn.cursor()

            # Try materialized view first
            try:
                cursor.execute("SELECT * FROM sample_statistics")
                row = cursor.fetchone()
                if row:
                    stats = {
                        "total": row["total_count"],
                        "malicious": row["malicious_count"],
                        "benign": row["benign_count"],
                        "balance_ratio": row["malicious_count"] / row["benign_count"] if row["benign_count"] > 0 else 0
                    }
            except:
                # Fallback to direct query
                cursor.execute("SELECT COUNT(*) FROM samples")
                total = cursor.fetchone()["count"]

                cursor.execute("SELECT COUNT(*) FROM samples WHERE label = 'malicious'")
                malicious = cursor.fetchone()["count"]

                cursor.execute("SELECT COUNT(*) FROM samples WHERE label = 'benign'")
                benign = cursor.fetchone()["count"]

                stats = {
                    "total": total,
                    "malicious": malicious,
                    "benign": benign,
                    "balance_ratio": malicious / benign if benign > 0 else 0
                }

            # Get sources breakdown
            cursor.execute("""
                SELECT source, COUNT(*) as count
                FROM samples
                GROUP BY source
            """)
            sources = {row["source"]: row["count"] for row in cursor.fetchall()}
            stats["sources"] = sources

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
        finally:
            cursor.close()
            return_connection(conn)

        return stats

    def create_version_snapshot(self, version: str, metadata: Optional[Dict] = None) -> bool:
        """Create a versioned snapshot of current dataset."""
        stats = self.get_dataset_stats()
        conn = get_connection()

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO dataset_versions
                (version, total_samples, malicious_count, benign_count, sources, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (version) DO NOTHING
                RETURNING id
                """,
                (
                    version,
                    stats["total"],
                    stats["malicious"],
                    stats["benign"],
                    Json(stats["sources"]),
                    Json(metadata) if metadata else Json({})
                )
            )
            result = cursor.fetchone()
            conn.commit()

            if result:
                logger.info(f"Created dataset version: {version}")
                return True
            else:
                logger.warning(f"Version {version} already exists")
                return False

        except Exception as e:
            logger.error(f"Failed to create version {version}: {e}")
            conn.rollback()
            return False
        finally:
            cursor.close()
            return_connection(conn)

    def get_versions(self) -> List[Dict]:
        """Get all dataset versions."""
        conn = get_connection()
        versions = []

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version, total_samples, malicious_count, benign_count,
                       sources, metadata, created_at
                FROM dataset_versions
                ORDER BY created_at DESC
            """)

            for row in cursor.fetchall():
                versions.append({
                    "version": row["version"],
                    "total_samples": row["total_samples"],
                    "malicious_count": row["malicious_count"],
                    "benign_count": row["benign_count"],
                    "sources": row["sources"] or {},
                    "metadata": row["metadata"] or {},
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None
                })

        except Exception as e:
            logger.error(f"Failed to retrieve versions: {e}")
        finally:
            cursor.close()
            return_connection(conn)

        return versions

    def export_to_jsonl(self, output_path: str, label: Optional[str] = None):
        """Export samples to JSONL format."""
        samples = self.get_all_samples(label=label)

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps({
                    "content": sample["content"],
                    "label": sample["label"],
                    "source": sample["source"]
                }) + "\n")

        logger.info(f"Exported {len(samples)} samples to {output_path}")

    def close(self):
        """Close database connections (for compatibility)."""
        logger.info("Close called (connection pool managed automatically)")
