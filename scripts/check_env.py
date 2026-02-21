#!/usr/bin/env python3
"""Check if environment variables are properly set."""

import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def check_env():
    """Check required environment variables."""
    print("=" * 60)
    print("ENVIRONMENT VARIABLES CHECK")
    print("=" * 60)

    # Check .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ .env file found at: {env_file.absolute()}")
    else:
        print(f"❌ .env file NOT found!")
        print(f"   Expected location: {env_file.absolute()}")
        print(f"   Create it with: cp .env.example .env")
        return 1

    # Required variables
    required_vars = {
        "QDRANT_API_KEY": "Qdrant authentication",
        "SCRIPTGUARD_API_KEY": "ScriptGuard API authentication",
        "POSTGRES_PASSWORD": "PostgreSQL password",
    }

    # Optional but useful
    optional_vars = {
        "GITHUB_API_TOKEN": "GitHub data source",
        "NVD_API_KEY": "NVD CVE data",
        "HUGGINGFACE_TOKEN": "HuggingFace model downloads",
    }

    print("\n" + "=" * 60)
    print("REQUIRED VARIABLES")
    print("=" * 60)

    all_ok = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask the value for security
            masked = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
            print(f"✅ {var}: {masked}")
            print(f"   ({description})")
        else:
            print(f"❌ {var}: NOT SET")
            print(f"   ({description})")
            all_ok = False

    print("\n" + "=" * 60)
    print("OPTIONAL VARIABLES")
    print("=" * 60)

    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            masked = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
            print(f"✅ {var}: {masked}")
        else:
            print(f"⚠️  {var}: Not set (optional)")
        print(f"   ({description})")

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ALL REQUIRED VARIABLES ARE SET")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME REQUIRED VARIABLES ARE MISSING")
        print("=" * 60)
        print("\nTo fix:")
        print("1. Edit .env file")
        print("2. Add missing variables")
        print("3. Run this script again")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(check_env())
