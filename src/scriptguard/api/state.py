"""
Application State Management.
Handles lifecycle of global resources like models and database connections.
"""

import os
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
from scriptguard.rag.qdrant_store import QdrantStore, bootstrap_cve_data
from scriptguard.utils.logger import logger
from scriptguard.config_loader import load_config
from scriptguard.schemas.config_schema import ScriptGuardConfig

class AppState:
    """
    Singleton-like class to hold application state.
    """
    def __init__(self):
        self.config: Optional[ScriptGuardConfig] = None
        self.model = None
        self.tokenizer = None
        self.rag_store: Optional[QdrantStore] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_config(self, config_path: str = "config.yaml"):
        """Load configuration."""
        try:
            self.config = load_config(config_path)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Fallback or re-raise depending on strictness requirements
            raise

    def load_resources(self):
        """Load model, tokenizer, and RAG store."""
        if not self.config:
            self.load_config()

        self._load_model()
        self._load_rag()

    def _load_model(self):
        """Load the model and tokenizer."""
        # Use config values if available, else defaults
        model_id = self.config.training.model_id if self.config else "bigcode/starcoder2-3b"
        # For inference, we might want a different model or adapter path
        # Assuming adapter path is relative to where we run or defined in env/config
        # The original code used env vars, let's respect config but allow env overrides if needed
        
        # Note: The original code loaded a CausalLM. 
        # The review suggested SequenceClassification for deterministic output.
        # For now, we will stick to CausalLM as per current implementation but structure it better.
        # Switching to SequenceClassification requires a model trained for that head.
        
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
            
            # Check for adapter
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

            # Bootstrap if empty
            info = self.rag_store.get_collection_info()
            points_count = info.get('points_count', 0)

            if points_count == 0:
                logger.info("Qdrant collection is empty. Bootstrapping...")
                bootstrap_cve_data(self.rag_store)
                logger.info("✅ Qdrant initialized with CVE patterns")
            else:
                logger.info(f"✅ Qdrant ready ({points_count} vectors)")

        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed: {e}")
            logger.warning("API will run without RAG support")
            self.rag_store = None

# Global instance
app_state = AppState()
