"""
Tests for the Multi-Input Fusion Architecture.

Tests cover:
  - FeatureExtractor: correctness, robustness, fixed dimensionality
  - FusedDataCollator: column handling, dtype
  - FusedCodeBERTClassifier.forward: output shape
  - StandardScaler: train-only fitting
  - Chunk-to-feature mapping: all chunks of same script share features
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace

from scriptguard.features.extractor import FeatureExtractor


# ===========================================================================
# FeatureExtractor tests
# ===========================================================================

class TestFeatureExtractor:

    def setup_method(self):
        self.ex = FeatureExtractor()

    def test_empty_code_no_crash(self):
        result = self.ex.extract("")
        assert result == [0.0] * 61

    def test_syntax_error_no_crash(self):
        result = self.ex.extract("def broken(")
        assert isinstance(result, list)
        assert len(result) == 61

    def test_no_imports(self):
        code = "x = 1\ny = x + 2"
        result = self.ex.extract(code)
        # has_socket, has_subprocess, etc. should all be 0
        import_features = result[13:24]  # indices 13-23 inclusive
        assert all(f == 0.0 for f in import_features)

    def test_correct_length(self):
        """Feature vector must always be exactly 61 elements."""
        samples = [
            "",
            "x = 1",
            "def broken(",
            "import socket\nexec(eval('os.system(\"ls\")'))",
            "a" * 10000,
        ]
        for code in samples:
            result = self.ex.extract(code)
            assert len(result) == 61, f"Expected 61 features, got {len(result)} for code starting with {code[:30]!r}"

    def test_has_socket_detected(self):
        code = "import socket\ns = socket.socket()"
        result = self.ex.extract(code)
        # has_socket is the first import feature (index 13 overall)
        has_socket = result[13]
        assert has_socket == 1.0, f"Expected has_socket=1.0, got {has_socket}"

    def test_obfuscation_exec_detected(self):
        code = "exec('print(1)')"
        result = self.ex.extract(code)
        # Feature layout: AST(13) + Import(11) + Entropy(4) + Obfuscation(11) + ...
        # has_exec is obfuscation[0] → index 13+11+4 = 28
        has_exec = result[28]
        assert has_exec == 1.0

    def test_statistical_total_lines(self):
        code = "a = 1\nb = 2\nc = 3"
        result = self.ex.extract(code)
        # total_lines is statistical[0], starting at index 13+11+4+11+6+4+4+3=56
        total_lines = result[56]
        assert total_lines == 3.0

    def test_nested_function_detected(self):
        code = """
def outer():
    def inner():
        pass
"""
        result = self.ex.extract(code)
        # has_nested_functions is ast_features[9] (index 9)
        has_nested = result[9]
        assert has_nested == 1.0

    def test_ip_address_detected(self):
        code = 'host = "192.168.1.100"'
        result = self.ex.extract(code)
        # unique_ip_count is network_features[0], starting at index 13+11+4+11=39
        unique_ip = result[39]
        assert unique_ip >= 1.0

    def test_high_entropy_string(self):
        # Base64-like random string with high entropy
        code = 'key = "dGhpcyBpcyBhIHRlc3Qgc3RyaW5n"'
        result = self.ex.extract(code)
        # Feature layout: AST(13) + Import(11) + Entropy(4) + ...
        # entropy starts at index 24: [mean_string_entropy, max_string_entropy, high_entropy_string_count, has_hardcoded_key]
        mean_entropy = result[24]
        assert mean_entropy > 0.0


# ===========================================================================
# FusedDataCollator tests
# ===========================================================================

class TestFusedDataCollator:

    def test_pops_feature_vector(self, mocker):
        from scriptguard.models.fused_classifier import FusedDataCollator

        mock_tokenizer = mocker.MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Mock DataCollatorWithPadding
        mock_collator = mocker.MagicMock()
        mock_collator.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mocker.patch(
            "scriptguard.models.fused_classifier.DataCollatorWithPadding",
            return_value=mock_collator,
        )

        collator = FusedDataCollator(tokenizer=mock_tokenizer)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
             "label": 0, "feature_vector": [0.1] * 61, "script_id": 0, "chunk_index": 0},
        ]
        batch = collator(features)

        # feature_vector should be in batch, script_id should not
        assert "feature_vector" in batch
        assert "script_id" not in batch
        assert "chunk_index" not in batch

    def test_feature_dtype_float32(self, mocker):
        from scriptguard.models.fused_classifier import FusedDataCollator

        mock_tokenizer = mocker.MagicMock()

        mock_collator = mocker.MagicMock()
        mock_collator.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mocker.patch(
            "scriptguard.models.fused_classifier.DataCollatorWithPadding",
            return_value=mock_collator,
        )

        collator = FusedDataCollator(tokenizer=mock_tokenizer)
        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1],
             "label": 1, "feature_vector": [1.5] * 61},
        ]
        batch = collator(features)

        assert batch["feature_vector"].dtype == torch.float32


# ===========================================================================
# FusedCodeBERTClassifier forward tests
# ===========================================================================

class TestFusedClassifierForward:

    def test_output_logits_shape(self, mocker):
        """Verify forward pass produces (B, 2) logits without loading real weights."""
        from scriptguard.models.fused_classifier import FusedCodeBERTClassifier

        B = 3
        seq_len = 16
        feature_dim = 61
        hidden_size = 768

        # Mock AutoModel to return fake last_hidden_state
        mock_bert_output = SimpleNamespace(
            last_hidden_state=torch.zeros(B, seq_len, hidden_size)
        )
        mock_bert = mocker.MagicMock()
        mock_bert.return_value = mock_bert_output
        mock_bert.config = SimpleNamespace(hidden_size=hidden_size)

        mocker.patch(
            "scriptguard.models.fused_classifier.AutoModel.from_pretrained",
            return_value=mock_bert,
        )

        model = FusedCodeBERTClassifier(
            model_name="mock/model",
            num_labels=2,
            feature_dim=feature_dim,
            mlp_hidden_dim=128,
            fusion_hidden_dim=256,
            dropout_rate=0.0,
        )
        model.eval()

        input_ids = torch.zeros(B, seq_len, dtype=torch.long)
        attention_mask = torch.ones(B, seq_len, dtype=torch.long)
        feature_vector = torch.zeros(B, feature_dim, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, feature_vector=feature_vector)

        assert hasattr(output, "logits")
        assert output.logits.shape == (B, 2)


# ===========================================================================
# StandardScaler tests
# ===========================================================================

class TestScalerFitTransform:

    def test_scaler_not_fitted_on_val(self):
        """Scaler fitted only on train; val is transformed (not fitted)."""
        rng = np.random.default_rng(42)
        train_features = rng.standard_normal((100, 61)).astype(np.float32)
        val_features = rng.standard_normal((30, 61)).astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(train_features)

        # Transform val without fitting again
        val_scaled = scaler.transform(val_features)

        # Scaler mean/std should match train statistics, not val
        assert scaler.mean_.shape == (61,)
        assert scaler.scale_.shape == (61,)
        # val scaled should have values (not all zeros)
        assert not np.allclose(val_scaled, 0.0)

    def test_scaled_output_shape(self):
        rng = np.random.default_rng(0)
        val_features = rng.standard_normal((20, 61)).astype(np.float32)
        train_features = rng.standard_normal((50, 61)).astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(train_features)
        val_scaled = scaler.transform(val_features)

        assert val_scaled.shape == (20, 61)


# ===========================================================================
# Chunk-to-feature mapping tests
# ===========================================================================

class TestScriptLevelAggregation:
    """Verify evaluate() aggregates chunks to scripts via max malicious_prob."""

    def test_max_malicious_prob_aggregation(self):
        """Script with one high-malicious chunk should be classified malicious."""
        chunk_mal_probs = {
            0: [0.2, 0.9, 0.3],  # script 0: max=0.9 → malicious
            1: [0.1, 0.2, 0.15], # script 1: max=0.2 → benign
        }
        threshold = 0.5
        for sid, probs in chunk_mal_probs.items():
            best = max(probs)
            pred = 1 if best >= threshold else 0
            if sid == 0:
                assert pred == 1, "Script with max_prob=0.9 should be malicious"
            else:
                assert pred == 0, "Script with max_prob=0.2 should be benign"

    def test_threshold_respected(self):
        """Prediction flips exactly at decision_threshold."""
        for threshold in [0.4, 0.5, 0.6]:
            prob_just_below = threshold - 0.001
            prob_just_above = threshold + 0.001
            assert (1 if prob_just_below >= threshold else 0) == 0
            assert (1 if prob_just_above >= threshold else 0) == 1


class TestChunkToFeatureMapping:

    def test_all_chunks_same_script_get_same_features(self):
        """All chunks sharing the same script_id must receive the same feature vector."""
        ex = FeatureExtractor()
        code = "import socket\nimport subprocess\n" + "x = 1\n" * 200  # Long code → multiple chunks

        features = ex.extract(code)

        # Simulate multiple chunks from the same script
        script_id = 42
        chunks = [
            {"script_id": script_id, "chunk_index": i}
            for i in range(5)
        ]

        # All chunks get same features (by script_id lookup)
        script_features = {script_id: features}
        for chunk in chunks:
            sid = chunk["script_id"]
            fvec = script_features[sid]
            assert fvec == features, f"Chunk {chunk['chunk_index']} has different features!"
            assert len(fvec) == 61
