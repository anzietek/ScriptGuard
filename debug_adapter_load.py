"""Debug script to test adapter loading."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import traceback

model_id = "bigcode/starcoder2-3b"
adapter_path = "./model_checkpoints/final_adapter"

print(f"Loading base model: {model_id}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    print("✅ Base model loaded")

    print(f"\nLoading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("✅ Adapter loaded successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

