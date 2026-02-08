"""
Application State Management.
Handles lifecycle of global resources like models and database connections.
"""

import os
import torch
from typing import Optional, Dict, Any, Protocol, runtime_checkable, Any as TypingAny
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from scriptguard.rag.qdrant_store import QdrantStore, bootstrap_cve_data
from scriptguard.utils.logger import logger
from scriptguard.config_loader import load_config
from scriptguard.schemas.config_schema import ScriptGuardConfig

# Optional asyncpg import for graceful degradation
try:
    import asyncpg as _asyncpg
    HAS_ASYNCPG = True
except ImportError:
    _asyncpg = None
    HAS_ASYNCPG = False
    logger.warning("asyncpg not found. Database logging will be disabled.")


@runtime_checkable
class _DbPool(Protocol):
    """Protocol for the subset of asyncpg.Pool API used by this service."""

    async def close(self) -> None:
        """Close the pool."""

    def acquire(self) -> TypingAny:
        """Return an async context manager yielding a connection."""


class AppState:
    """
    Singleton-like class to hold application state.
    """

    def __init__(self):
        self.config: Optional[ScriptGuardConfig] = None
        self.model = None
        self.tokenizer = None
        self.rag_store: Optional[QdrantStore] = None
        self.db_pool: Optional[_DbPool] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration.

        Args:
            config_path: Optional explicit path to config. If not provided, uses
                CONFIG_PATH env var, falling back to "config.yaml".
        """
        resolved_path = config_path or os.getenv("CONFIG_PATH", "config.yaml")
        try:
            self.config = load_config(resolved_path)
            logger.info(f"Configuration loaded from {resolved_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {resolved_path}: {e}")
            raise

    async def load_resources(self) -> None:
        """Load model, tokenizer, RAG store, and DB connection."""
        if not self.config:
            self.load_config()

        self._load_model()
        self._load_rag()
        await self._init_db()

    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Database connection pool closed")

    def _load_model(self):
        """Load the model and tokenizer."""
        model_id = self.config.training.model_id if self.config else "bigcode/starcoder2-3b"
        
        logger.info(f"Loading model: {model_id} on {self.device}")
        
        try:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
            
            adapter_path = os.getenv("ADAPTER_PATH", "./model_checkpoints/final_adapter")
            if os.path.exists(adapter_path):
                self.model = PeftModel.from_pretrained(base_model, adapter_path)
                logger.info(f"✅ Loaded adapter from {adapter_path}")
            else:
                self.model = base_model
                logger.warning(f"⚠️  Adapter not found at {adapter_path}, using base model.")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_rag(self):
        """Initialize Qdrant RAG store."""
        if not self.config:
            return

        qdrant_cfg = self.config.qdrant
        
        try:
            self.rag_store = QdrantStore(
                host=qdrant_cfg.host,
                port=qdrant_cfg.port,
                collection_name=qdrant_cfg.collection_name,
                embedding_model=qdrant_cfg.embedding_model,
                api_key=qdrant_cfg.api_key,
                use_https=qdrant_cfg.use_https
            )

            # Check if bootstrapping is allowed via env var (default: False for API)
            bootstrap_enabled = os.getenv("BOOTSTRAP_QDRANT", "false").lower() == "true"
            
            # Also check config override
            if self.config.qdrant.bootstrap_on_startup:
                bootstrap_enabled = True

            if bootstrap_enabled:
                info = self.rag_store.get_collection_info()
                points_count = info.get('points_count', 0)

                if points_count == 0:
                    logger.info("Qdrant collection is empty. Bootstrapping...")
                    bootstrap_cve_data(self.rag_store)
                    logger.info("✅ Qdrant initialized with CVE patterns")
                else:
                    logger.info(f"✅ Qdrant ready ({points_count} vectors)")
            else:
                # Just check connection without writing
                info = self.rag_store.get_collection_info()
                points_count = info.get('points_count', 0)
                logger.info(f"✅ Qdrant connected ({points_count} vectors). Bootstrapping disabled.")

        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed: {e}")
            logger.warning("API will run without RAG support")
            self.rag_store = None

    async def _init_db(self) -> None:
        """Initialize PostgreSQL connection pool and schema."""
        if not HAS_ASYNCPG or not self.config:
            return

        db_cfg = self.config.database.postgresql
        dsn = (
            f"postgresql://{db_cfg.user}:{db_cfg.password}"
            f"@{db_cfg.host}:{db_cfg.port}/{db_cfg.database}"
        )

        try:
            # asyncpg.create_pool timeout is a connection-establishment timeout.
            # command_timeout should be applied per-connection, not here.
            self.db_pool = await _asyncpg.create_pool(
                dsn=dsn,
                min_size=db_cfg.min_connections,
                max_size=db_cfg.max_connections,
                timeout=db_cfg.connection_timeout,
                command_timeout=db_cfg.command_timeout,
            )

            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scan_history (
                        id SERIAL PRIMARY KEY,
                        request_id UUID NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        script_hash VARCHAR(64) NOT NULL,
                        is_malicious BOOLEAN NOT NULL,
                        confidence FLOAT,
                        model_version VARCHAR(50),
                        api_key_prefix VARCHAR(10),
                        processing_time_ms FLOAT
                    );
                    CREATE INDEX IF NOT EXISTS idx_scan_history_hash ON scan_history(script_hash);
                    CREATE INDEX IF NOT EXISTS idx_scan_history_timestamp ON scan_history(timestamp);
                    """
                )

            logger.info("✅ PostgreSQL connected and schema initialized")

        except Exception as e:
            logger.error(f"❌ PostgreSQL initialization failed: {e}")
            logger.warning("API will run without DB logging")
            self.db_pool = None

    async def log_scan_result(self, data: Dict[str, Any]):
        """
        Log scan result to database asynchronously.
        
        Args:
            data: Dictionary containing scan metadata
        """
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO scan_history 
                    (request_id, script_hash, is_malicious, confidence, model_version, api_key_prefix, processing_time_ms)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                data['request_id'],
                data['script_hash'],
                data['is_malicious'],
                data['confidence'],
                data['model_version'],
                data['api_key_prefix'],
                data['processing_time_ms']
                )
        except Exception as e:
            logger.error(f"Failed to log scan result: {e}")

# Global instance
app_state = AppState()
