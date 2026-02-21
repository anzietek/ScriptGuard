import json
import pytest
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path
from scriptguard.exceptions import InferenceError


@pytest.fixture
def mock_model_dir(tmp_path: Path) -> Path:
    (tmp_path / "inference_config.json").write_text(
        json.dumps({"max_tokens": 512, "chunk_overlap": 50})
    )
    (tmp_path / "config.json").write_text(json.dumps({"num_labels": 2}))
    return tmp_path


def _make_mock_model(predicted_class: int = 0, confidence: float = 0.95) -> MagicMock:
    logits = torch.zeros(1, 2)
    logits[0, predicted_class] = 5.0
    output = MagicMock()
    output.logits = logits
    model = MagicMock()
    model.return_value = output
    model.eval.return_value = None
    return model


def _make_mock_tokenizer(n_tokens: int = 100) -> MagicMock:
    tok = MagicMock()
    tok.cls_token_id = 0
    tok.sep_token_id = 2
    tok.encode.return_value = list(range(n_tokens))
    return tok


@pytest.fixture
def classifier(mock_model_dir):
    mock_tokenizer = _make_mock_tokenizer()
    mock_model = _make_mock_model(predicted_class=0)

    with patch("scriptguard.inference.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("scriptguard.inference.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
        from scriptguard.inference.classifier import ScriptGuardClassifier
        clf = ScriptGuardClassifier(str(mock_model_dir))
    return clf


class TestScriptGuardClassifier:
    def test_returns_two_tuple(self, classifier):
        result = classifier.classify("import os")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_label_is_valid_string(self, classifier):
        label, _ = classifier.classify("import os")
        assert label in ("malicious", "benign")

    def test_confidence_in_valid_range(self, classifier):
        _, conf = classifier.classify("print('hello')")
        assert 0.0 <= conf <= 1.0

    def test_empty_string_raises_inference_error(self, classifier):
        with pytest.raises(InferenceError):
            classifier.classify("")

    def test_whitespace_only_raises_inference_error(self, classifier):
        with pytest.raises(InferenceError):
            classifier.classify("   \n  ")

    def test_malicious_prediction(self, mock_model_dir):
        mock_tokenizer = _make_mock_tokenizer()
        mock_model = _make_mock_model(predicted_class=1)

        with patch("scriptguard.inference.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("scriptguard.inference.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
            from importlib import reload
            import scriptguard.inference.classifier as clf_module
            reload(clf_module)
            clf = clf_module.ScriptGuardClassifier(str(mock_model_dir))

        label, conf = clf.classify("import subprocess; subprocess.run(['rm','-rf','/'])")
        assert label == "malicious"
        assert conf > 0.5

    def test_nonexistent_model_path_raises(self):
        from scriptguard.inference.classifier import ScriptGuardClassifier
        with pytest.raises(InferenceError):
            ScriptGuardClassifier("/nonexistent/path/that/does/not/exist")
