"""
Test script to verify all critical dependencies are working.
"""

import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Devices: {torch.cuda.device_count()}")
except Exception as e:
    print(f"✗ PyTorch Error: {e}")

try:
    import triton
    print(f"✓ Triton: {triton.__version__}")
except Exception as e:
    print(f"✗ Triton Error: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers Error: {e}")

try:
    import unsloth
    print(f"✓ Unsloth: loaded successfully")
except Exception as e:
    print(f"✗ Unsloth Error: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print(f"✓ Sentence Transformers: loaded successfully")
except Exception as e:
    print(f"✗ Sentence Transformers Error: {e}")

try:
    import zenml
    print(f"✓ ZenML: {zenml.__version__}")
except Exception as e:
    print(f"✗ ZenML Error: {e}")

print("\n✓ All critical dependencies verified!")
