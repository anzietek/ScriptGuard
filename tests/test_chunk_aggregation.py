import pytest
from scriptguard.steps.evaluation import aggregate_chunk_predictions


class TestAggregateChunkPredictions:
    def test_single_chunk_returns_its_prediction(self):
        chunks = [{"predicted_label": 1, "confidence": 0.85}]
        label, conf = aggregate_chunk_predictions(chunks)
        assert label == 1
        assert conf == pytest.approx(0.85)

    def test_two_chunks_same_label_returns_highest_confidence(self):
        chunks = [
            {"predicted_label": 1, "confidence": 0.70},
            {"predicted_label": 1, "confidence": 0.90},
        ]
        label, conf = aggregate_chunk_predictions(chunks)
        assert label == 1
        assert conf == pytest.approx(0.90)

    def test_conflicting_labels_returns_highest_confidence_label(self):
        chunks = [
            {"predicted_label": 1, "confidence": 0.80},
            {"predicted_label": 0, "confidence": 0.95},
        ]
        label, conf = aggregate_chunk_predictions(chunks)
        assert label == 0
        assert conf == pytest.approx(0.95)

    def test_all_uncertain_returns_highest_confidence_chunk(self):
        chunks = [
            {"predicted_label": 1, "confidence": 0.52},
            {"predicted_label": 0, "confidence": 0.55},
            {"predicted_label": 1, "confidence": 0.51},
        ]
        label, conf = aggregate_chunk_predictions(chunks)
        assert label == 0
        assert conf == pytest.approx(0.55)

    def test_three_chunks_picks_max_across_all(self):
        chunks = [
            {"predicted_label": 0, "confidence": 0.60},
            {"predicted_label": 1, "confidence": 0.99},
            {"predicted_label": 0, "confidence": 0.75},
        ]
        label, conf = aggregate_chunk_predictions(chunks)
        assert label == 1
        assert conf == pytest.approx(0.99)

    def test_first_chunk_wins_when_equal_confidence(self):
        chunks = [
            {"predicted_label": 1, "confidence": 0.80},
            {"predicted_label": 0, "confidence": 0.80},
        ]
        label, conf = aggregate_chunk_predictions(chunks)
        assert label == 1
        assert conf == pytest.approx(0.80)
