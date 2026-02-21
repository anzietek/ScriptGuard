import json
import os
import pytest
from scriptguard.exceptions import ModelRegistrationError


def _run_registration(metrics: dict, model_path: str, config: dict) -> bool:
    from scriptguard.steps.model_registration import register_model as _step
    return _step.entrypoint(metrics=metrics, model_path=model_path, config=config)


def _config(threshold: float = 0.92) -> dict:
    return {"codebert": {"eval_threshold": threshold}}


def _make_model_dir(tmp_path, name: str = "model"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "pytorch_model.bin").write_bytes(b"fake weights")
    return model_dir


class TestRegisterModel:
    def test_high_recall_registers(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        metrics = {"malicious_recall": 0.93, "accuracy": 0.88}

        result = _run_registration(metrics, str(model_dir), _config(0.92))

        assert result is True
        latest_path = tmp_path / "latest"
        assert latest_path.exists()
        meta = json.loads((latest_path / "registration_metadata.json").read_text())
        assert meta["metrics"]["malicious_recall"] == 0.93

    def test_low_recall_does_not_register(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        metrics = {"malicious_recall": 0.91}

        result = _run_registration(metrics, str(model_dir), _config(0.92))

        assert result is False
        assert not (tmp_path / "latest").exists()

    def test_exact_threshold_registers(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        metrics = {"malicious_recall": 0.92}

        result = _run_registration(metrics, str(model_dir), _config(0.92))

        assert result is True

    def test_missing_recall_key_raises(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        metrics = {"accuracy": 0.88, "f1": 0.90}

        with pytest.raises(ModelRegistrationError):
            _run_registration(metrics, str(model_dir), _config(0.92))

    def test_empty_metrics_raises(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)

        with pytest.raises(ModelRegistrationError):
            _run_registration({}, str(model_dir), _config(0.92))
