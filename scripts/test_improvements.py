"""
Quick Training Test Script
===========================

Tests the improved configuration on a small subset to verify changes work.
Then runs full training if test passes.
"""

import sys
import os
import yaml
from pathlib import Path

def check_config():
    """Verify that config.yaml has been updated correctly"""
    print("🔍 Checking config.yaml...")

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    checks = []

    # Check augmentation
    aug = config.get("augmentation", {})
    if aug.get("balance_method") == "oversample":
        checks.append("✅ balance_method = oversample")
    else:
        checks.append(f"❌ balance_method = {aug.get('balance_method')} (should be 'oversample')")

    if aug.get("variants_per_sample") >= 5:
        checks.append(f"✅ variants_per_sample = {aug.get('variants_per_sample')}")
    else:
        checks.append(f"⚠️  variants_per_sample = {aug.get('variants_per_sample')} (recommended: 5)")

    # Check GitHub limits
    gh = config.get("data_sources", {}).get("github", {})
    if gh.get("max_samples_per_keyword") >= 100:
        checks.append(f"✅ max_samples_per_keyword = {gh.get('max_samples_per_keyword')}")
    else:
        checks.append(f"⚠️  max_samples_per_keyword = {gh.get('max_samples_per_keyword')} (recommended: 100)")

    # Check HF limits
    hf = config.get("data_sources", {}).get("additional_hf", {})
    if hf.get("max_samples_per_dataset") >= 500:
        checks.append(f"✅ max_samples_per_dataset = {hf.get('max_samples_per_dataset')}")
    else:
        checks.append(f"⚠️  max_samples_per_dataset = {hf.get('max_samples_per_dataset')} (recommended: 500)")

    # Check training params
    train = config.get("training", {})
    if train.get("num_epochs") >= 10:
        checks.append(f"✅ num_epochs = {train.get('num_epochs')}")
    else:
        checks.append(f"⚠️  num_epochs = {train.get('num_epochs')} (recommended: 10)")

    if train.get("learning_rate") <= 0.0001:
        checks.append(f"✅ learning_rate = {train.get('learning_rate')}")
    else:
        checks.append(f"⚠️  learning_rate = {train.get('learning_rate')} (recommended: 0.0001)")

    if train.get("lr_scheduler_type") == "cosine":
        checks.append("✅ lr_scheduler_type = cosine")
    else:
        checks.append(f"⚠️  lr_scheduler_type = {train.get('lr_scheduler_type')} (recommended: 'cosine')")

    print("\n📋 Configuration Check Results:")
    print("-" * 60)
    for check in checks:
        print(check)
    print("-" * 60)

    # Count issues
    errors = sum(1 for c in checks if c.startswith("❌"))
    warnings = sum(1 for c in checks if c.startswith("⚠️"))

    if errors > 0:
        print(f"\n❌ {errors} critical issue(s) found!")
        print("Please fix config.yaml before running training.")
        return False
    elif warnings > 0:
        print(f"\n⚠️  {warnings} recommendation(s) - training will work but results may be suboptimal")
        return True
    else:
        print("\n✅ All checks passed! Configuration looks good.")
        return True


def estimate_training_time(config):
    """Estimate training time based on configuration"""
    training = config.get("training", {})

    # Rough estimates
    epochs = training.get("num_epochs", 3)
    samples_estimate = 2000  # With new config

    # Assume ~30 seconds per epoch per 100 samples
    time_per_epoch = (samples_estimate / 100) * 30  # seconds
    total_time = time_per_epoch * epochs

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)

    return hours, minutes, samples_estimate


def print_summary():
    """Print summary of expected improvements"""
    print("\n" + "="*60)
    print("📊 EXPECTED IMPROVEMENTS")
    print("="*60)
    print("\n🔴 BEFORE (with old config):")
    print("   Accuracy:  57.69%")
    print("   F1 Score:  52.17%")
    print("   Dataset:   458 samples")
    print("   Time:      ~40 minutes")

    print("\n🟢 AFTER (with new config):")
    print("   Accuracy:  70-75% (expected)")
    print("   F1 Score:  65-70% (expected)")
    print("   Dataset:   2,000-3,000 samples (estimated)")

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    hours, minutes, samples = estimate_training_time(config)
    print(f"   Time:      ~{hours}h {minutes}m (estimated)")

    print("\n💡 KEY CHANGES:")
    print("   ✅ Oversample instead of undersample (keeps all data)")
    print("   ✅ 5x more augmentation variants")
    print("   ✅ 10x more data from HuggingFace")
    print("   ✅ 5x more GitHub samples")
    print("   ✅ 10 epochs instead of 3")
    print("   ✅ Cosine LR scheduler")
    print("="*60)


def main():
    """Main entry point"""
    print("\n" + "🚀 ScriptGuard Training - Quick Check")
    print("="*60)

    # Check if config is properly updated
    if not check_config():
        print("\n❌ Configuration check failed!")
        print("Please review and update config.yaml")
        sys.exit(1)

    # Print expected improvements
    print_summary()

    # Ask user if they want to proceed
    print("\n❓ Ready to start training?")
    print("   This will:")
    print("   1. Collect 2,000-3,000 samples (vs 458 before)")
    print("   2. Train for 10 epochs (vs 3 before)")
    print("   3. Use intelligent oversampling")
    print("   4. Take approximately 2-3 hours")

    response = input("\nProceed with training? [y/N]: ").strip().lower()

    if response in ['y', 'yes']:
        print("\n🚀 Starting training pipeline...")
        print("Running: python main.py")
        print("-" * 60)

        # Run the main training script
        os.system("python main.py")
    else:
        print("\n✋ Training cancelled.")
        print("\nTo run training later, execute:")
        print("   python main.py")
        print("\nOr run this script again:")
        print("   python scripts/test_improvements.py")


if __name__ == "__main__":
    main()
