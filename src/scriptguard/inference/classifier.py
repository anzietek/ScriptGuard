import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer

from scriptguard.exceptions import InferenceError
from scriptguard.features.extractor import FeatureExtractor
from scriptguard.models.fused_classifier import load_fused_model
from scriptguard.utils.tokenization_utils import sliding_window_chunks
from scriptguard.utils.logger import logger


class ScriptGuardClassifier:
    LABEL_MAP: dict[int, str] = {0: "benign", 1: "malicious"}

    def __init__(self, model_path: str, scaler_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise InferenceError(f"Model path does not exist: {model_path}")
        if not Path(scaler_path).exists():
            raise InferenceError(f"Scaler path does not exist: {scaler_path}")

        config_file = path / "inference_config.json"
        if config_file.exists():
            with open(config_file) as f:
                inference_cfg = json.load(f)
            self._max_tokens: int = inference_cfg.get("max_tokens", 512)
            self._chunk_overlap: int = inference_cfg.get("chunk_overlap", 50)
            self._decision_threshold: float = inference_cfg.get("decision_threshold", 0.5)
        else:
            logger.warning("inference_config.json not found; using defaults max_tokens=512, chunk_overlap=50")
            self._max_tokens = 512
            self._chunk_overlap = 50
            self._decision_threshold = 0.5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_fused(path, scaler_path)

    def _init_fused(self, path: Path, scaler_path: str) -> None:
        """Load the gated-fusion model and feature scaler."""
        logger.info(f"Loading fused model from {path}")
        self.model, self.tokenizer = load_fused_model(str(path), self.device)
        self.model.eval()

        logger.info(f"Loading feature scaler from {scaler_path}")
        self._scaler = joblib.load(scaler_path)
        self._extractor = FeatureExtractor()
        logger.info("Fused model and scaler loaded successfully")

    def _chunk_script(self, script: str) -> list[dict]:
        return sliding_window_chunks(
            tokenizer=self.tokenizer,
            text=script,
            max_length=self._max_tokens,
            overlap=self._chunk_overlap,
            script_id=0,
            label=0,
        )

    def classify(self, script: str) -> tuple[str, float]:
        if not script or not script.strip():
            raise InferenceError("Cannot classify empty script")

        # 1. Feature extraction
        # Uwaga: to zwraca 27 elementów
        features_27d = self._extractor.extract(script)

        # 2. HEURISTIC SHORT-CIRCUIT
        # Zamiast ciąć tablicę, wywołujemy bezpośrednio detektor gadżetów
        critical_gadgets = self._extractor._gadget_features(script)
        gadget_count = sum(critical_gadgets)

        if gadget_count >= 1.0:
            logger.info(f"Heuristic Match: {gadget_count} gadgets. Immediate block.")
            return "malicious", 1.0

        # 3. Standard AI Path (przywrócona pętla!)
        scaled_features = self._scaler.transform(np.array([features_27d], dtype=np.float32))
        feature_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)

        chunks = self._chunk_script(script)
        best_malicious_prob: float = 0.0

        with torch.no_grad():
            for chunk in chunks:
                input_ids = torch.tensor([chunk["input_ids"]], dtype=torch.long).to(self.device)
                attention_mask = torch.tensor([chunk["attention_mask"]], dtype=torch.long).to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    feature_vector=feature_tensor,
                )

                probs = torch.softmax(outputs.logits, dim=-1)
                malicious_prob = probs[0][1].item()
                if malicious_prob > best_malicious_prob:
                    best_malicious_prob = malicious_prob

        # 4. Strict Threshold for Benign Safety
        if best_malicious_prob >= self._decision_threshold:
            return "malicious", best_malicious_prob

        return "benign", 1.0 - best_malicious_prob