import json
import os
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scriptguard.exceptions import InferenceError
from scriptguard.utils.tokenization_utils import sliding_window_chunks
from scriptguard.utils.logger import logger


class ScriptGuardClassifier:
    LABEL_MAP: dict[int, str] = {0: "benign", 1: "malicious"}

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise InferenceError(f"Model path does not exist: {model_path}")

        config_file = path / "inference_config.json"
        if config_file.exists():
            with open(config_file) as f:
                inference_cfg = json.load(f)
            self._max_tokens: int = inference_cfg.get("max_tokens", 512)
            self._chunk_overlap: int = inference_cfg.get("chunk_overlap", 50)
            self._decision_threshold: float = inference_cfg.get("decision_threshold", 0.5)
            self._is_fused: bool = inference_cfg.get("model_type") == "fused"
        else:
            logger.warning("inference_config.json not found; using defaults max_tokens=512, chunk_overlap=50")
            self._max_tokens = 512
            self._chunk_overlap = 50
            self._decision_threshold = 0.5
            self._is_fused = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self._is_fused:
            self._init_fused(path)
        else:
            self._init_legacy(path)

    def _init_fused(self, path: Path) -> None:
        """Initialize the fused model path."""
        import joblib
        from scriptguard.features.extractor import FeatureExtractor
        from scriptguard.models.fused_classifier import load_fused_model

        logger.info(f"Loading fused model from {path}")
        self.model, self.tokenizer = load_fused_model(str(path), self.device)

        scaler_file = path / "feature_scaler.joblib"
        if not scaler_file.exists():
            raise InferenceError(f"feature_scaler.joblib not found in {path}")
        self._scaler = joblib.load(str(scaler_file))
        self._extractor = FeatureExtractor()
        logger.info("Fused model loaded successfully")

    def _init_legacy(self, path: Path) -> None:
        """Initialize the legacy (non-fused) AutoModelForSequenceClassification path."""
        logger.info(f"Loading tokenizer from {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        logger.info(f"Loading legacy model from {path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path), use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()
        self._scaler = None
        self._extractor = None

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

        chunks = self._chunk_script(script)
        best_malicious_prob: float = 0.0

        if self._is_fused:
            import numpy as np
            # Extract features once for the full script
            raw = self._extractor.extract(script)
            scaled = self._scaler.transform(np.array([raw], dtype=np.float32))
            feature_tensor = torch.tensor(scaled, dtype=torch.float32).to(self.device)

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
        else:
            with torch.no_grad():
                for chunk in chunks:
                    input_ids = torch.tensor([chunk["input_ids"]], dtype=torch.long).to(self.device)
                    attention_mask = torch.tensor([chunk["attention_mask"]], dtype=torch.long).to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    malicious_prob = probs[0][1].item()
                    if malicious_prob > best_malicious_prob:
                        best_malicious_prob = malicious_prob

        if best_malicious_prob >= self._decision_threshold:
            return "malicious", best_malicious_prob
        return "benign", 1.0 - best_malicious_prob
