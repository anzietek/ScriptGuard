"""Debug tokenizer encoding for BENIGN/MALICIOUS."""
from transformers import AutoTokenizer

model_id = "bigcode/starcoder2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Testing tokenization:")
print("=" * 60)

# Test various forms
test_words = [
    "BENIGN",
    " BENIGN",
    "\nBENIGN",
    "Benign",
    " Benign",
    "MALICIOUS",
    " MALICIOUS",
    "\nMALICIOUS",
    "Malicious",
    " Malicious",
]

for word in test_words:
    tokens = tokenizer.encode(word, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"Word: {repr(word):20} -> Token IDs: {tokens} -> Decoded: {decoded}")

print("\n" + "=" * 60)
print("Testing prompt completion context:")
print("=" * 60)

# Simulate the prompt ending
prompt_ending = '# Analysis: The script above is classified as:'
prompt_tokens = tokenizer.encode(prompt_ending, add_special_tokens=False)
print(f"Prompt ending tokens: {prompt_tokens}")
print(f"Decoded: {[tokenizer.decode([t]) for t in prompt_tokens]}")

# What comes next?
full_benign = prompt_ending + " BENIGN"
full_malicious = prompt_ending + " MALICIOUS"

full_benign_tokens = tokenizer.encode(full_benign, add_special_tokens=False)
full_malicious_tokens = tokenizer.encode(full_malicious, add_special_tokens=False)

# Get only the new tokens after the prompt
benign_new_tokens = full_benign_tokens[len(prompt_tokens):]
malicious_new_tokens = full_malicious_tokens[len(prompt_tokens):]

print(f"\nNew tokens for BENIGN: {benign_new_tokens}")
print(f"Decoded: {[tokenizer.decode([t]) for t in benign_new_tokens]}")

print(f"\nNew tokens for MALICIOUS: {malicious_new_tokens}")
print(f"Decoded: {[tokenizer.decode([t]) for t in malicious_new_tokens]}")

print("\n" + "=" * 60)
print("Vocabulary lookup:")
print("=" * 60)

# Direct vocabulary lookup
vocab = tokenizer.get_vocab()
for key in sorted(vocab.keys()):
    if 'benign' in key.lower() or 'malicious' in key.lower():
        print(f"Token: {repr(key):30} -> ID: {vocab[key]}")
