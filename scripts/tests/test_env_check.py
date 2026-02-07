#!/usr/bin/env python
"""
Minimalny test - sprawdź czy env variables są załadowane
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("ENV VARIABLES CHECK")
print("=" * 60)

env_vars = [
    "GITHUB_API_TOKEN",
    "NVD_API_KEY",
    "MALWAREBAZAAR_API_KEY",
    "HUGGINGFACE_TOKEN",
    "POSTGRES_HOST",
    "POSTGRES_PORT"
]

for var in env_vars:
    value = os.getenv(var)
    if value:
        # Show first 10 chars only for security
        masked = value[:10] + "..." if len(value) > 10 else value
        print(f"✅ {var}: {masked}")
    else:
        print(f"❌ {var}: NOT SET")

print("=" * 60)

# Now test if config.yaml can load
print("\nTesting config.yaml loading...")
import yaml

try:
    with open("../../config.yaml", "r") as f:
        config_str = f.read()

    # Replace env vars manually
    import re
    def replace_env_var(match):
        var_name = match.group(1).split(":-")[0]  # Handle ${VAR:-default}
        return os.getenv(var_name, "")

    config_str = re.sub(r'\$\{([^}]+)\}', replace_env_var, config_str)
    config = yaml.safe_load(config_str)

    print("✅ Config loaded successfully")
    print(f"GitHub enabled: {config.get('data_sources', {}).get('github', {}).get('enabled')}")
    print(f"CVE feeds enabled: {config.get('data_sources', {}).get('cve_feeds', {}).get('enabled')}")

except Exception as e:
    print(f"❌ Config loading failed: {e}")

print("\nDone!")
