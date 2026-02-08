"""
Inference Engine Interface.
Provides a deterministic prediction interface for the API.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch

class InferenceEngine(ABC):
    """Abstract base class for inference engines."""
    
    @abstractmethod
    def predict(self, input_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Run prediction on input text.
        
        Args:
            input_text: The script or text to analyze.
            context: Optional context (e.g., from RAG).
            
        Returns:
            Dictionary containing prediction results (is_malicious, confidence, reasoning).
        """
        pass

class CausalLMInferenceEngine(InferenceEngine):
    """Inference engine using a Causal LM (e.g., StarCoder2)."""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def predict(self, input_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        # Placeholder for the logic currently in main.py
        # This class should encapsulate the generation and parsing logic
        # to make it testable and reusable.
        pass
