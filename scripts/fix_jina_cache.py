#!/usr/bin/env python3
"""Fix HuggingFace cache path issue for Jina-v3."""

import os
import shutil
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "jinaai"

print("Fixing Jina-v3 cache paths...")

if cache_dir.exists():
    # Find the problematic directory
    for subdir in cache_dir.iterdir():
        if "xlm_hyphen_roberta" in subdir.name:
            print(f"\nFound problematic dir: {subdir.name}")

            # Create proper symlink or copy with correct name
            correct_name = subdir.name.replace("_hyphen_", "-")
            correct_path = cache_dir / correct_name

            if not correct_path.exists():
                print(f"Creating: {correct_name}")
                shutil.copytree(subdir, correct_path)
                print("✓ Fixed!")
            else:
                print("Already exists")
else:
    print("Cache dir doesn't exist yet")

print("\nNow try: python scripts/test_jina_v3_comparison.py")
