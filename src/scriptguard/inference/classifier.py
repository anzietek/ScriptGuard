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
        else:
            logger.warning("inference_config.json not found; using defaults max_tokens=512, chunk_overlap=50")
            self._max_tokens = 512
            self._chunk_overlap = 50
            self._decision_threshold = 0.5

        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        logger.info(f"Loading model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path), use_safetensors=True)
        self.model.to(self.device)
        self.model.eval()

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
