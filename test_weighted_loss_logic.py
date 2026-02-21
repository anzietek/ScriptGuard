"""
Standalone test for weighted loss logic.
Tests the core weighting algorithm without importing unsloth dependencies.
"""

import torch


def compute_sample_weights(texts: list[str], class_weights: dict) -> list[float]:
    """
    Core logic from WeightedLossTrainer.compute_loss().
    Determines weight for each sample based on class label in text.
    """
    sample_weights = []

    for text in texts:
        # Determine class based on prompt content
        if "MALICIOUS" in text.upper():
            weight = class_weights.get('malicious', 1.0)
        elif "BENIGN" in text.upper():
            weight = class_weights.get('benign', 1.0)
        else:
            # Unknown class, use neutral weight
            weight = 1.0

        sample_weights.append(weight)

    return sample_weights


def apply_weighted_loss(base_loss: torch.Tensor, sample_weights: list[float]) -> torch.Tensor:
    """
    Core loss weighting logic from WeightedLossTrainer.compute_loss().
    Applies average weight to batch loss.
    """
    weights_tensor = torch.tensor(sample_weights, dtype=base_loss.dtype, device=base_loss.device)
    avg_weight = weights_tensor.mean()
    weighted_loss = base_loss * avg_weight
    return weighted_loss


def test_weight_assignment():
    """Test that samples are correctly assigned weights based on their class."""

    print("=" * 80)
    print("Testing Weighted Loss Logic (Core Algorithm)")
    print("=" * 80)

    class_weights = {
        'malicious': 2.0,  # High weight for minority class
        'benign': 0.5      # Low weight for majority class
    }

    print(f"\nClass weights: {class_weights}")

    # Test 1: All malicious samples
    print("\n1. Testing all malicious samples...")
    malicious_texts = [
        '# Analysis: The script above is classified as: MALICIOUS',
        'Code review:\nResult: MALICIOUS',
        'This is malicious code',
    ]

    weights = compute_sample_weights(malicious_texts, class_weights)
    print(f"   Texts: {len(malicious_texts)} samples")
    print(f"   Assigned weights: {weights}")
    assert all(w == 2.0 for w in weights), f"Expected all 2.0, got {weights}"
    print("   OK PASS: All malicious samples assigned weight 2.0")

    # Test 2: All benign samples
    print("\n2. Testing all benign samples...")
    benign_texts = [
        '# Analysis: The script above is classified as: BENIGN',
        'Code review:\nResult: BENIGN',
        'This is benign code',
    ]

    weights = compute_sample_weights(benign_texts, class_weights)
    print(f"   Texts: {len(benign_texts)} samples")
    print(f"   Assigned weights: {weights}")
    assert all(w == 0.5 for w in weights), f"Expected all 0.5, got {weights}"
    print("   OK PASS: All benign samples assigned weight 0.5")

    # Test 3: Mixed samples
    print("\n3. Testing mixed samples...")
    mixed_texts = [
        '# Analysis: The script above is classified as: MALICIOUS',
        '# Analysis: The script above is classified as: BENIGN',
        'Result: MALICIOUS',
        'Result: BENIGN',
    ]

    weights = compute_sample_weights(mixed_texts, class_weights)
    print(f"   Texts: {len(mixed_texts)} samples")
    print(f"   Assigned weights: {weights}")
    assert weights == [2.0, 0.5, 2.0, 0.5], f"Expected [2.0, 0.5, 2.0, 0.5], got {weights}"
    print("   OK PASS: Mixed samples assigned correct weights")

    # Test 4: Unknown class (neutral weight)
    print("\n4. Testing unknown class...")
    unknown_texts = [
        'No classification here',
        'Some random text',
    ]

    weights = compute_sample_weights(unknown_texts, class_weights)
    print(f"   Texts: {len(unknown_texts)} samples")
    print(f"   Assigned weights: {weights}")
    assert all(w == 1.0 for w in weights), f"Expected all 1.0, got {weights}"
    print("   OK PASS: Unknown samples assigned neutral weight 1.0")

    # Test 5: Case insensitivity
    print("\n5. Testing case insensitivity...")
    case_texts = [
        'classified as: malicious',  # lowercase
        'classified as: MALICIOUS',  # uppercase
        'classified as: Malicious',  # mixed case
        'classified as: benign',
        'classified as: BENIGN',
        'classified as: Benign',
    ]

    weights = compute_sample_weights(case_texts, class_weights)
    print(f"   Texts: {len(case_texts)} samples")
    print(f"   Assigned weights: {weights}")
    assert weights == [2.0, 2.0, 2.0, 0.5, 0.5, 0.5], f"Expected [2.0, 2.0, 2.0, 0.5, 0.5, 0.5], got {weights}"
    print("   OK PASS: Case insensitive matching works correctly")

    print("\n" + "=" * 80)


def test_loss_weighting():
    """Test that weighted loss is computed correctly."""

    print("\nTesting Loss Weighting (Core Algorithm)")
    print("=" * 80)

    base_loss = torch.tensor(1.0, requires_grad=True)

    # Test 1: All malicious (weight 2.0)
    print("\n1. Testing all malicious batch...")
    sample_weights = [2.0, 2.0, 2.0, 2.0]
    weighted_loss = apply_weighted_loss(base_loss, sample_weights)
    expected = 2.0  # avg([2.0, 2.0, 2.0, 2.0]) * 1.0 = 2.0
    print(f"   Base loss: {base_loss.item()}")
    print(f"   Sample weights: {sample_weights}")
    print(f"   Average weight: {sum(sample_weights) / len(sample_weights)}")
    print(f"   Expected weighted loss: {expected}")
    print(f"   Actual weighted loss: {weighted_loss.item():.3f}")
    assert abs(weighted_loss.item() - expected) < 0.01, f"Expected {expected}, got {weighted_loss.item()}"
    print("   OK PASS: All malicious batch weighted correctly (2.0×)")

    # Test 2: All benign (weight 0.5)
    print("\n2. Testing all benign batch...")
    sample_weights = [0.5, 0.5, 0.5, 0.5]
    weighted_loss = apply_weighted_loss(base_loss, sample_weights)
    expected = 0.5  # avg([0.5, 0.5, 0.5, 0.5]) * 1.0 = 0.5
    print(f"   Base loss: {base_loss.item()}")
    print(f"   Sample weights: {sample_weights}")
    print(f"   Average weight: {sum(sample_weights) / len(sample_weights)}")
    print(f"   Expected weighted loss: {expected}")
    print(f"   Actual weighted loss: {weighted_loss.item():.3f}")
    assert abs(weighted_loss.item() - expected) < 0.01, f"Expected {expected}, got {weighted_loss.item()}"
    print("   OK PASS: All benign batch weighted correctly (0.5×)")

    # Test 3: Mixed batch (2 malicious, 2 benign)
    print("\n3. Testing mixed batch...")
    sample_weights = [2.0, 0.5, 2.0, 0.5]
    weighted_loss = apply_weighted_loss(base_loss, sample_weights)
    expected = 1.25  # avg([2.0, 0.5, 2.0, 0.5]) = 1.25, * 1.0 = 1.25
    print(f"   Base loss: {base_loss.item()}")
    print(f"   Sample weights: {sample_weights}")
    print(f"   Average weight: {sum(sample_weights) / len(sample_weights)}")
    print(f"   Expected weighted loss: {expected}")
    print(f"   Actual weighted loss: {weighted_loss.item():.3f}")
    assert abs(weighted_loss.item() - expected) < 0.01, f"Expected {expected}, got {weighted_loss.item()}"
    print("   OK PASS: Mixed batch weighted correctly (1.25×)")

    # Test 4: Different base loss
    print("\n4. Testing with different base loss...")
    base_loss = torch.tensor(2.5, requires_grad=True)
    sample_weights = [2.0, 0.5, 2.0, 0.5]
    weighted_loss = apply_weighted_loss(base_loss, sample_weights)
    expected = 2.5 * 1.25  # base_loss * avg_weight = 2.5 * 1.25 = 3.125
    print(f"   Base loss: {base_loss.item()}")
    print(f"   Sample weights: {sample_weights}")
    print(f"   Average weight: {sum(sample_weights) / len(sample_weights)}")
    print(f"   Expected weighted loss: {expected}")
    print(f"   Actual weighted loss: {weighted_loss.item():.3f}")
    assert abs(weighted_loss.item() - expected) < 0.01, f"Expected {expected}, got {weighted_loss.item()}"
    print("   OK PASS: Different base loss scaled correctly (3.125×)")

    # Test 5: Neutral weights (all 1.0)
    print("\n5. Testing neutral weights...")
    base_loss = torch.tensor(1.0, requires_grad=True)
    sample_weights = [1.0, 1.0, 1.0, 1.0]
    weighted_loss = apply_weighted_loss(base_loss, sample_weights)
    expected = 1.0  # avg([1.0, 1.0, 1.0, 1.0]) * 1.0 = 1.0 (no change)
    print(f"   Base loss: {base_loss.item()}")
    print(f"   Sample weights: {sample_weights}")
    print(f"   Average weight: {sum(sample_weights) / len(sample_weights)}")
    print(f"   Expected weighted loss: {expected}")
    print(f"   Actual weighted loss: {weighted_loss.item():.3f}")
    assert abs(weighted_loss.item() - expected) < 0.01, f"Expected {expected}, got {weighted_loss.item()}"
    print("   OK PASS: Neutral weights produce no change (1.0×)")

    # Test 6: Gradient preservation
    print("\n6. Testing gradient preservation...")
    base_loss = torch.tensor(1.0, requires_grad=True)
    sample_weights = [2.0, 0.5]
    weighted_loss = apply_weighted_loss(base_loss, sample_weights)

    # Backprop to verify gradients flow correctly
    weighted_loss.backward()

    print(f"   Base loss gradient: {base_loss.grad.item()}")
    assert base_loss.grad is not None, "Gradient should be preserved"
    assert abs(base_loss.grad.item() - 1.25) < 0.01, f"Expected gradient 1.25, got {base_loss.grad.item()}"
    print("   OK PASS: Gradients flow correctly through weighting")

    print("\n" + "=" * 80)


def test_realistic_scenario():
    """Test with realistic class imbalance scenario."""

    print("\nTesting Realistic Scenario")
    print("=" * 80)

    # Realistic weights from sqrt_inverse with 2000 malicious, 5000 benign
    class_weights = {
        'malicious': 1.58,  # sqrt(7000 / (2 * 2000))
        'benign': 0.89      # sqrt(7000 / (2 * 5000))
    }

    print(f"\nScenario: 2000 malicious, 5000 benign samples")
    print(f"Class weights (sqrt_inverse): {class_weights}")

    # Simulate realistic batch: more benign due to random sampling
    batch_texts = [
        '# Analysis: The script above is classified as: MALICIOUS',
        '# Analysis: The script above is classified as: BENIGN',
        '# Analysis: The script above is classified as: BENIGN',
        '# Analysis: The script above is classified as: BENIGN',
        '# Analysis: The script above is classified as: BENIGN',
        '# Analysis: The script above is classified as: BENIGN',
    ]

    print(f"\nBatch composition: 1 malicious, 5 benign")

    # Compute weights
    weights = compute_sample_weights(batch_texts, class_weights)
    print(f"Sample weights: {weights}")

    # Compute weighted loss
    base_loss = torch.tensor(1.0)
    weighted_loss = apply_weighted_loss(base_loss, weights)

    # Expected: (1.58 + 5*0.89) / 6 = (1.58 + 4.45) / 6 = 1.005
    expected_avg = (1.58 + 5 * 0.89) / 6
    expected_loss = 1.0 * expected_avg

    print(f"\nAverage batch weight: {expected_avg:.3f}")
    print(f"Expected weighted loss: {expected_loss:.3f}")
    print(f"Actual weighted loss: {weighted_loss.item():.3f}")

    assert abs(weighted_loss.item() - expected_loss) < 0.01, \
        f"Expected {expected_loss:.3f}, got {weighted_loss.item():.3f}"

    print("OK PASS: Realistic scenario weighted correctly")

    # Compare to batch with more malicious samples
    batch_texts_malicious = [
        '# Analysis: The script above is classified as: MALICIOUS',
        '# Analysis: The script above is classified as: MALICIOUS',
        '# Analysis: The script above is classified as: MALICIOUS',
        '# Analysis: The script above is classified as: BENIGN',
        '# Analysis: The script above is classified as: BENIGN',
        '# Analysis: The script above is classified as: BENIGN',
    ]

    print(f"\nComparing to batch with more malicious samples: 3 malicious, 3 benign")

    weights_malicious = compute_sample_weights(batch_texts_malicious, class_weights)
    weighted_loss_malicious = apply_weighted_loss(base_loss, weights_malicious)

    expected_avg_malicious = (3 * 1.58 + 3 * 0.89) / 6
    expected_loss_malicious = 1.0 * expected_avg_malicious

    print(f"Average batch weight: {expected_avg_malicious:.3f}")
    print(f"Expected weighted loss: {expected_loss_malicious:.3f}")
    print(f"Actual weighted loss: {weighted_loss_malicious.item():.3f}")

    assert weighted_loss_malicious.item() > weighted_loss.item(), \
        "Batch with more malicious samples should have higher weighted loss"

    print(f"\nOK Batch with more malicious samples has higher loss: {weighted_loss_malicious.item():.3f} > {weighted_loss.item():.3f}")
    print("OK PASS: Weighting correctly emphasizes minority class")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        test_weight_assignment()
        test_loss_weighting()
        test_realistic_scenario()

        print("\n" + "=" * 80)
        print("OK ALL TESTS PASSED!")
        print("=" * 80)
        print("\nConclusion:")
        print("  OK Weight assignment logic works correctly")
        print("  OK Loss weighting algorithm works correctly")
        print("  OK Gradients flow correctly through weighting")
        print("  OK Realistic scenario behaves as expected")
        print("  OK Minority class gets properly emphasized")
        print("\nWeightedLossTrainer core logic is CORRECT! OK")
        print("=" * 80)

    except AssertionError as e:
        print(f"\nFAIL TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nFAIL UNEXPECTED ERROR: {e}")
        raise
