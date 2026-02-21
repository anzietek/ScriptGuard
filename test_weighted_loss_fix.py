"""
Test script to verify WeightedLossTrainer implementation.
Confirms that sample-level weighting is actually applied.
"""

import torch
from transformers import AutoTokenizer, TrainingArguments
from unittest.mock import Mock, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scriptguard.models.qlora_finetuner import WeightedLossTrainer


def test_sample_weighting():
    """Test that samples are weighted correctly based on their class."""

    print("=" * 80)
    print("Testing WeightedLossTrainer Sample-Level Weighting")
    print("=" * 80)

    # Setup: Create mock components
    print("\n1. Setting up mock tokenizer and model...")

    # Mock tokenizer
    tokenizer = Mock()

    # Create realistic text samples
    malicious_text = (
        '"""\nSecurity Analysis Report\n------------------------\n'
        'Target Script:\nimport os; os.system("rm -rf /")\n"""\n'
        '# Analysis: The script above is classified as: MALICIOUS'
    )

    benign_text = (
        '"""\nSecurity Analysis Report\n------------------------\n'
        'Target Script:\nprint("Hello, world!")\n"""\n'
        '# Analysis: The script above is classified as: BENIGN'
    )

    # Mock tokenizer decode to return our texts
    def mock_decode(ids, skip_special_tokens=False):
        # Return malicious for odd indices, benign for even
        idx = ids.tolist()[0] if hasattr(ids, 'tolist') else ids
        return malicious_text if idx % 2 == 1 else benign_text

    tokenizer.decode = mock_decode

    # Mock model
    mock_model = Mock()

    # Mock training args
    training_args = Mock(spec=TrainingArguments)
    training_args.label_smoothing_factor = 0.0  # Disable for testing

    # Setup class weights
    class_weights = {
        'malicious': 2.0,  # High weight for minority class
        'benign': 0.5      # Low weight for majority class
    }

    print(f"   Class weights: {class_weights}")

    # Create WeightedLossTrainer instance
    print("\n2. Creating WeightedLossTrainer...")
    trainer = WeightedLossTrainer(
        model=mock_model,
        args=training_args,
        tokenizer=tokenizer,
        class_weights=class_weights
    )

    # Test 1: Verify weight assignment for malicious samples
    print("\n3. Testing malicious sample weighting...")

    # Create batch with all malicious samples (odd token IDs)
    malicious_batch = {
        'input_ids': torch.tensor([[1], [3], [5], [7]]),  # Odd = malicious
        'labels': torch.tensor([[1], [3], [5], [7]])
    }

    # Mock the parent compute_loss to return a known value
    def mock_parent_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        base_loss = torch.tensor(1.0, requires_grad=True)
        mock_outputs = Mock()
        return (base_loss, mock_outputs) if return_outputs else base_loss

    # Patch parent's compute_loss
    original_compute_loss = WeightedLossTrainer.__bases__[0].compute_loss
    WeightedLossTrainer.__bases__[0].compute_loss = mock_parent_loss

    try:
        # Compute weighted loss
        weighted_loss = trainer.compute_loss(mock_model, malicious_batch, return_outputs=False)

        # Expected: all samples are malicious (weight=2.0)
        # Average weight = 2.0
        # Weighted loss = base_loss (1.0) * avg_weight (2.0) = 2.0
        expected_loss = 2.0

        print(f"   Base loss: 1.0")
        print(f"   Sample weights: [2.0, 2.0, 2.0, 2.0] (all malicious)")
        print(f"   Average weight: 2.0")
        print(f"   Expected weighted loss: {expected_loss}")
        print(f"   Actual weighted loss: {weighted_loss.item():.3f}")

        assert abs(weighted_loss.item() - expected_loss) < 0.01, \
            f"Malicious batch: Expected {expected_loss}, got {weighted_loss.item()}"

        print("   ✓ PASS: Malicious samples weighted correctly (2.0×)")

        # Test 2: Verify weight assignment for benign samples
        print("\n4. Testing benign sample weighting...")

        benign_batch = {
            'input_ids': torch.tensor([[0], [2], [4], [6]]),  # Even = benign
            'labels': torch.tensor([[0], [2], [4], [6]])
        }

        weighted_loss = trainer.compute_loss(mock_model, benign_batch, return_outputs=False)

        # Expected: all samples are benign (weight=0.5)
        # Average weight = 0.5
        # Weighted loss = base_loss (1.0) * avg_weight (0.5) = 0.5
        expected_loss = 0.5

        print(f"   Base loss: 1.0")
        print(f"   Sample weights: [0.5, 0.5, 0.5, 0.5] (all benign)")
        print(f"   Average weight: 0.5")
        print(f"   Expected weighted loss: {expected_loss}")
        print(f"   Actual weighted loss: {weighted_loss.item():.3f}")

        assert abs(weighted_loss.item() - expected_loss) < 0.01, \
            f"Benign batch: Expected {expected_loss}, got {weighted_loss.item()}"

        print("   ✓ PASS: Benign samples weighted correctly (0.5×)")

        # Test 3: Verify mixed batch weighting
        print("\n5. Testing mixed batch weighting...")

        mixed_batch = {
            'input_ids': torch.tensor([[1], [2], [3], [4]]),  # 2 malicious, 2 benign
            'labels': torch.tensor([[1], [2], [3], [4]])
        }

        weighted_loss = trainer.compute_loss(mock_model, mixed_batch, return_outputs=False)

        # Expected: 2 malicious (2.0) + 2 benign (0.5)
        # Average weight = (2.0 + 0.5 + 2.0 + 0.5) / 4 = 1.25
        # Weighted loss = base_loss (1.0) * avg_weight (1.25) = 1.25
        expected_loss = 1.25

        print(f"   Base loss: 1.0")
        print(f"   Sample weights: [2.0, 0.5, 2.0, 0.5] (2 malicious, 2 benign)")
        print(f"   Average weight: 1.25")
        print(f"   Expected weighted loss: {expected_loss}")
        print(f"   Actual weighted loss: {weighted_loss.item():.3f}")

        assert abs(weighted_loss.item() - expected_loss) < 0.01, \
            f"Mixed batch: Expected {expected_loss}, got {weighted_loss.item()}"

        print("   ✓ PASS: Mixed batch weighted correctly (1.25×)")

        # Test 4: Verify fallback when no weights configured
        print("\n6. Testing fallback with no class weights...")

        trainer_no_weights = WeightedLossTrainer(
            model=mock_model,
            args=training_args,
            tokenizer=tokenizer,
            class_weights=None  # No weights
        )

        weighted_loss = trainer_no_weights.compute_loss(mock_model, mixed_batch, return_outputs=False)

        # Expected: no weighting applied, should return base loss (1.0)
        expected_loss = 1.0

        print(f"   Base loss: 1.0")
        print(f"   Class weights: None")
        print(f"   Expected weighted loss: {expected_loss}")
        print(f"   Actual weighted loss: {weighted_loss.item():.3f}")

        assert abs(weighted_loss.item() - expected_loss) < 0.01, \
            f"No weights: Expected {expected_loss}, got {weighted_loss.item()}"

        print("   ✓ PASS: Fallback to standard loss when no weights configured")

        # Test 5: Verify return_outputs behavior
        print("\n7. Testing return_outputs=True...")

        loss, outputs = trainer.compute_loss(mock_model, malicious_batch, return_outputs=True)

        assert outputs is not None, "Should return outputs when return_outputs=True"
        assert abs(loss.item() - 2.0) < 0.01, "Loss value should match with return_outputs=True"

        print(f"   Loss: {loss.item():.3f}")
        print(f"   Outputs: {type(outputs)}")
        print("   ✓ PASS: return_outputs=True works correctly")

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nConclusion:")
        print("  - Sample-level weighting is correctly applied")
        print("  - Malicious samples get 2.0× weight")
        print("  - Benign samples get 0.5× weight")
        print("  - Mixed batches compute average weight correctly")
        print("  - Fallback works when no weights configured")
        print("  - return_outputs parameter handled correctly")
        print("\nWeightedLossTrainer implementation is WORKING! ✓")
        print("=" * 80)

    finally:
        # Restore original compute_loss
        WeightedLossTrainer.__bases__[0].compute_loss = original_compute_loss


if __name__ == "__main__":
    test_sample_weighting()
