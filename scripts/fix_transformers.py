#!/usr/bin/env python3
"""
Fix transformers dependency issues.

Reinstalls transformers and sentence-transformers with correct versions.
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run command and show output."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print('='*70)
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    else:
        print("✅ Success")
        return True

def main():
    """Fix transformers."""
    print("="*70)
    print("FIXING TRANSFORMERS DEPENDENCY")
    print("="*70)

    # Step 1: Uninstall problematic packages
    print("\n1. Uninstalling transformers and sentence-transformers...")
    run_command(
        [sys.executable, "-m", "pip", "uninstall", "-y", "transformers", "sentence-transformers"],
        "Uninstall old versions"
    )

    # Step 2: Clear pip cache
    print("\n2. Clearing pip cache...")
    run_command(
        [sys.executable, "-m", "pip", "cache", "purge"],
        "Clear pip cache"
    )

    # Step 3: Reinstall with specific versions
    print("\n3. Reinstalling transformers...")
    run_command(
        [sys.executable, "-m", "pip", "install", "transformers>=4.51.3"],
        "Install transformers"
    )

    print("\n4. Reinstalling sentence-transformers...")
    run_command(
        [sys.executable, "-m", "pip", "install", "sentence-transformers>=3.3.0"],
        "Install sentence-transformers"
    )

    # Step 4: Verify installation
    print("\n5. Verifying installation...")
    try:
        import transformers
        from transformers import PreTrainedModel
        print(f"✅ transformers {transformers.__version__} installed correctly")

        import sentence_transformers
        print(f"✅ sentence-transformers {sentence_transformers.__version__} installed correctly")

        print("\n" + "="*70)
        print("✅ ALL DEPENDENCIES FIXED!")
        print("="*70)
        print("\nYou can now run:")
        print("  python scripts/test_hybrid_balanced.py")

        return 0

    except ImportError as e:
        print(f"\n❌ Import still failing: {e}")
        print("\nTry manual fix:")
        print("  uv pip install --reinstall transformers sentence-transformers")
        return 1

if __name__ == "__main__":
    sys.exit(main())
