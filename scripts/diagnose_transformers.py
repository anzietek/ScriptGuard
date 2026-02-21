#!/usr/bin/env python3
"""Diagnose transformers installation issue."""

import sys
import os

print("="*70)
print("TRANSFORMERS DIAGNOSTIC")
print("="*70)

# Check if transformers is installed
try:
    import transformers
    print(f"\n✅ transformers package found")
    print(f"   Version: {transformers.__version__}")
    print(f"   Location: {transformers.__file__}")
except ImportError as e:
    print(f"\n❌ transformers not found: {e}")
    sys.exit(1)

# Check what's in __init__.py
print(f"\n2. Checking transformers.__init__.py contents...")
init_file = transformers.__file__
print(f"   File: {init_file}")

try:
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if PreTrainedModel is mentioned
    if 'PreTrainedModel' in content:
        print(f"   ✅ PreTrainedModel is in __init__.py")

        # Find the import line
        for line in content.split('\n'):
            if 'PreTrainedModel' in line and not line.strip().startswith('#'):
                print(f"   Import line: {line.strip()}")
    else:
        print(f"   ❌ PreTrainedModel NOT in __init__.py!")
        print(f"   This means the package is corrupted")
except Exception as e:
    print(f"   ❌ Failed to read __init__.py: {e}")

# Try to import PreTrainedModel
print(f"\n3. Trying to import PreTrainedModel...")
try:
    from transformers import PreTrainedModel
    print(f"   ✅ Import successful!")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")

    # Try direct import
    print(f"\n4. Trying direct import from modeling_utils...")
    try:
        from transformers.modeling_utils import PreTrainedModel
        print(f"   ✅ Direct import works!")
        print(f"   Issue: __init__.py is not exporting PreTrainedModel correctly")
    except ImportError as e2:
        print(f"   ❌ Direct import also failed: {e2}")
        print(f"   Issue: Package is severely corrupted")

# Check torch version compatibility
print(f"\n5. Checking PyTorch compatibility...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")

    # Check if torch is compatible with transformers
    torch_version = torch.__version__.split('+')[0]
    print(f"   Base version: {torch_version}")

    if torch_version.startswith('2.'):
        print(f"   ✅ PyTorch 2.x compatible with transformers")
    else:
        print(f"   ⚠️  Old PyTorch version may cause issues")
except ImportError:
    print(f"   ❌ PyTorch not installed!")

# Check sentence-transformers
print(f"\n6. Checking sentence-transformers...")
try:
    import sentence_transformers
    print(f"   ✅ sentence-transformers {sentence_transformers.__version__}")
except ImportError as e:
    print(f"   ❌ sentence-transformers not found: {e}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)

print("\nRecommended fix:")
print("  1. Delete .venv/Lib/site-packages/transformers directory")
print("  2. Reinstall with: uv pip install --force-reinstall transformers")
print("  3. Or use conda: conda install -c huggingface transformers")
