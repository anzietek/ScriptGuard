"""
Embedding Service - Unified Strategy for Code Embeddings
Supports multiple pooling strategies, L2 normalization, and configurable models.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Literal
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from scriptguard.utils.logger import logger


PoolingStrategy = Literal["cls", "mean_pooling", "pooler_output", "sentence_transformer"]


class EmbeddingService:
    """
    Unified embedding service with configurable pooling strategies.

    Supports:
    - cls: Use [CLS] token from last_hidden_state
    - mean_pooling: Mean pooling with attention mask
    - pooler_output: Use model's pooler_output (if available)
    - sentence_transformer: Use SentenceTransformer.encode() directly
    """

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        pooling_strategy: PoolingStrategy = "mean_pooling",
        normalize: bool = True,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize embedding service.

        Args:
            model_name: HuggingFace model name or SentenceTransformer name
            pooling_strategy: Pooling strategy to use
            normalize: Apply L2 normalization to embeddings
            max_length: Maximum sequence length
            device: Device to use (auto-detected if None)
        """
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing EmbeddingService:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Pooling: {pooling_strategy}")
        logger.info(f"  Normalize: {normalize}")
        logger.info(f"  Max Length: {max_length}")
        logger.info(f"  Device: {self.device}")

        # Initialize model based on strategy
        if pooling_strategy == "sentence_transformer":
            self._init_sentence_transformer()
        else:
            self._init_transformers()

        logger.info(f"✓ Embedding service ready (dim={self.embedding_dim})")

    def _init_sentence_transformer(self):
        """Initialize SentenceTransformer model."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.tokenizer = None
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _init_transformers(self):
        """Initialize standard Transformers model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()

            # Determine embedding dimension
            with torch.no_grad():
                test_input = self.tokenizer(
                    "test",
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                )
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
                test_output = self.model(**test_input)

                # Get dimension based on pooling strategy
                if self.pooling_strategy == "cls":
                    self.embedding_dim = test_output.last_hidden_state[:, 0, :].shape[-1]
                elif self.pooling_strategy == "mean_pooling":
                    self.embedding_dim = test_output.last_hidden_state.shape[-1]
                elif self.pooling_strategy == "pooler_output":
                    if hasattr(test_output, "pooler_output") and test_output.pooler_output is not None:
                        self.embedding_dim = test_output.pooler_output.shape[-1]
                    else:
                        logger.warning("pooler_output not available, falling back to cls")
                        self.pooling_strategy = "cls"
                        self.embedding_dim = test_output.last_hidden_state[:, 0, :].shape[-1]

        except Exception as e:
            logger.error(f"Failed to load Transformers model: {e}")
            raise

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling with attention mask.

        Args:
            token_embeddings: Token embeddings from model [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled embeddings [batch, hidden_dim]
        """
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings and divide by actual token count
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to embeddings.

        Args:
            embeddings: Embeddings array [batch, dim] or [dim]

        Returns:
            Normalized embeddings with L2 norm ≈ 1
        """
        if embeddings.ndim == 1:
            # Single embedding
            norm = np.linalg.norm(embeddings)
            if norm > 0:
                return embeddings / norm
            return embeddings
        else:
            # Batch of embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
            return embeddings / norms

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings [len(texts), embedding_dim]
        """
        if not texts:
            return np.array([])

        # Use SentenceTransformer if available
        if self.pooling_strategy == "sentence_transformer":
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=False  # We'll normalize manually if needed
            )

            if self.normalize:
                embeddings = self._normalize_embeddings(embeddings)

            return embeddings

        # Manual encoding with custom pooling
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Apply pooling strategy
                if self.pooling_strategy == "cls":
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                elif self.pooling_strategy == "mean_pooling":
                    batch_embeddings = self._mean_pooling(
                        outputs.last_hidden_state,
                        inputs["attention_mask"]
                    ).cpu().numpy()

                elif self.pooling_strategy == "pooler_output":
                    batch_embeddings = outputs.pooler_output.cpu().numpy()

                all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)

        # Apply L2 normalization if enabled
        if self.normalize:
            embeddings = self._normalize_embeddings(embeddings)

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text to embedding.

        Args:
            text: Text string to encode

        Returns:
            Numpy array embedding [embedding_dim]
        """
        embeddings = self.encode([text], batch_size=1, show_progress=False)
        return embeddings[0]

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def verify_normalization(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Verify L2 normalization of embeddings.

        Args:
            embeddings: Embeddings to verify [batch, dim] or [dim]

        Returns:
            Statistics dict with mean, min, max norms
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        norms = np.linalg.norm(embeddings, axis=1)

        return {
            "mean_norm": float(np.mean(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "std_norm": float(np.std(norms))
        }


def load_embedding_service_from_config(config: Dict[str, Any]) -> EmbeddingService:
    """
    Load embedding service from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized EmbeddingService
    """
    embedding_config = config.get("code_embedding", {})

    model_name = embedding_config.get("model", "microsoft/unixcoder-base")
    pooling_strategy = embedding_config.get("pooling_strategy", "mean_pooling")
    normalize = embedding_config.get("normalize", True)
    max_length = embedding_config.get("max_code_length", 512)

    return EmbeddingService(
        model_name=model_name,
        pooling_strategy=pooling_strategy,
        normalize=normalize,
        max_length=max_length
    )


if __name__ == "__main__":
    # Test embedding service
    service = EmbeddingService(
        model_name="microsoft/unixcoder-base",
        pooling_strategy="mean_pooling",
        normalize=True
    )

    test_texts = [
        "import os\nos.system('rm -rf /')",
        "import pandas as pd\ndf = pd.read_csv('data.csv')"
    ]

    embeddings = service.encode(test_texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Verify normalization
    stats = service.verify_normalization(embeddings)
    print(f"Normalization stats: {stats}")
