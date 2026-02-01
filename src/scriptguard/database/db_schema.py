"""
PostgreSQL Database Schema
Defines PostgreSQL schema for storing code samples and dataset versions.
"""

import logging
from typing import Optional
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor, Json
import os

logger = logging.getLogger(__name__)


class DatabasePool:
    """PostgreSQL connection pool manager."""

    _pool: Optional[pool.ThreadedConnectionPool] = None

    @classmethod
    def initialize(
        cls,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        minconn: int = 1,
        maxconn: int = 10
    ):
        """
        Initialize connection pool.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            minconn: Minimum connections in pool
            maxconn: Maximum connections in pool
        """
        if cls._pool is not None:
            logger.warning("Connection pool already initialized")
            return

        # Get from environment if not provided
        host = host or os.getenv("POSTGRES_HOST", "localhost")
        port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        database = database or os.getenv("POSTGRES_DB", "scriptguard")
        user = user or os.getenv("POSTGRES_USER", "scriptguard")
        password = password or os.getenv("POSTGRES_PASSWORD", "scriptguard")

        logger.info(f"Initializing PostgreSQL pool: {user}@{host}:{port}/{database}")

        try:
            cls._pool = pool.ThreadedConnectionPool(
                minconn,
                maxconn,
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                cursor_factory=RealDictCursor
            )
            logger.info("Connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    @classmethod
    def get_connection(cls):
        """Get connection from pool."""
        if cls._pool is None:
            cls.initialize()
        return cls._pool.getconn()

    @classmethod
    def return_connection(cls, conn):
        """Return connection to pool."""
        if cls._pool:
            cls._pool.putconn(conn)

    @classmethod
    def close_all(cls):
        """Close all connections in pool."""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            logger.info("Connection pool closed")


def create_database_schema(
    host: str = None,
    port: int = None,
    database: str = None,
    user: str = None,
    password: str = None
):
    """
    Create database schema with required tables.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: Database name
        user: Database user
        password: Database password
    """
    # Get from environment if not provided
    host = host or os.getenv("POSTGRES_HOST", "localhost")
    port = port or int(os.getenv("POSTGRES_PORT", "5432"))
    database = database or os.getenv("POSTGRES_DB", "scriptguard")
    user = user or os.getenv("POSTGRES_USER", "scriptguard")
    password = password or os.getenv("POSTGRES_PASSWORD", "scriptguard")

    logger.info(f"Creating database schema at {host}:{port}/{database}")

    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    conn.autocommit = True
    cursor = conn.cursor()

    try:
        # Samples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id SERIAL PRIMARY KEY,
                content_hash VARCHAR(64) UNIQUE NOT NULL,
                content TEXT NOT NULL,
                label VARCHAR(20) NOT NULL CHECK (label IN ('malicious', 'benign')),
                source VARCHAR(100) NOT NULL,
                url TEXT,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created 'samples' table")

        # Create indexes for samples
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_content_hash
            ON samples(content_hash)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_label
            ON samples(label)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_source
            ON samples(source)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_created_at
            ON samples(created_at DESC)
        """)

        # GIN index for JSONB metadata
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_metadata_gin
            ON samples USING GIN(metadata)
        """)

        logger.info("Created indexes for 'samples' table")

        # Dataset versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_versions (
                id SERIAL PRIMARY KEY,
                version VARCHAR(50) UNIQUE NOT NULL,
                total_samples INTEGER NOT NULL,
                malicious_count INTEGER NOT NULL,
                benign_count INTEGER NOT NULL,
                sources JSONB DEFAULT '{}'::jsonb,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created 'dataset_versions' table")

        # Create index for versions
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_versions_version
            ON dataset_versions(version)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_versions_created_at
            ON dataset_versions(created_at DESC)
        """)

        logger.info("Created indexes for 'dataset_versions' table")

        # Create trigger for updated_at
        cursor.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql'
        """)

        cursor.execute("""
            DROP TRIGGER IF EXISTS update_samples_updated_at ON samples
        """)

        cursor.execute("""
            CREATE TRIGGER update_samples_updated_at
            BEFORE UPDATE ON samples
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column()
        """)

        logger.info("Created triggers")

        # Create materialized view for statistics (optional, for performance)
        cursor.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS sample_statistics AS
            SELECT
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE label = 'malicious') as malicious_count,
                COUNT(*) FILTER (WHERE label = 'benign') as benign_count,
                COUNT(DISTINCT source) as source_count,
                AVG(LENGTH(content)) as avg_content_length,
                MIN(created_at) as first_sample_date,
                MAX(created_at) as last_sample_date
            FROM samples
            WITH DATA
        """)

        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sample_statistics
            ON sample_statistics ((1))
        """)

        logger.info("Created materialized view 'sample_statistics'")

        conn.commit()
        logger.info("Database schema created successfully")

    except Exception as e:
        logger.error(f"Failed to create schema: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def get_connection():
    """
    Get database connection from pool.

    Returns:
        PostgreSQL connection
    """
    return DatabasePool.get_connection()


def return_connection(conn):
    """
    Return connection to pool.

    Args:
        conn: PostgreSQL connection
    """
    DatabasePool.return_connection(conn)


def refresh_statistics():
    """Refresh materialized view for statistics."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY sample_statistics")
        conn.commit()
        logger.info("Refreshed sample_statistics materialized view")
    except Exception as e:
        logger.error(f"Failed to refresh statistics: {e}")
        conn.rollback()
    finally:
        cursor.close()
        return_connection(conn)
