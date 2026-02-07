"""
Test sprawdzający czy model evaluation może załadować model bez błędów
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model_loading():
    """Test czy model ładuje się bez błędu Params4bit"""

    base_model_id = "bigcode/starcoder2-3b"

    print("=" * 60)
    print("TEST: Model Loading for Evaluation")
    print("=" * 60)

    # Test 1: Sprawdź czy GPU loading z float16 działa
    print("\n✓ Test 1: GPU with float16 and memory management...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={0: "3.5GB"}  # Reserve some VRAM for operations
        )
        print("  ✅ GPU float16 with memory management - OK")
        print(f"  Device map: {model.hf_device_map}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return True
    except Exception as e:
        print(f"  ⚠️ GPU with memory limit failed: {type(e).__name__}: {str(e)[:100]}")

    # Test 2: Sprawdź czy loading bez quantization działa
    print("\n✓ Test 2: GPU without quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("  ✅ GPU fp16 - OK")
        print(f"  Device: {model.device}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return True
    except Exception as e:
        print(f"  ⚠️ GPU fp16 failed: {type(e).__name__}: {str(e)[:100]}")

    # Test 3: CPU fallback
    print("\n✓ Test 3: CPU fallback...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        print("  ✅ CPU fp32 - OK")
        print(f"  Device: {model.device}")
        del model
        return True
    except Exception as e:
        print(f"  ❌ CPU failed: {type(e).__name__}: {str(e)[:100]}")
        return False

if __name__ == "__main__":
    print("🧪 Testing model loading strategies...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    success = test_model_loading()

    print("\n" + "=" * 60)
    if success:
        print("✅ MODEL LOADING TEST PASSED!")
        print("Model evaluation powinien teraz działać bez błędów.")
    else:
        print("❌ MODEL LOADING TEST FAILED!")
        print("Model może nie ładować się - sprawdź instalacje transformers/bitsandbytes")
    print("=" * 60)
