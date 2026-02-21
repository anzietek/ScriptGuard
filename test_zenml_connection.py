"""Test ZenML server connection."""
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

print(f"ZENML_SERVER_URL from env: {os.getenv('ZENML_SERVER_URL')}")

# Now import ZenML
from zenml.client import Client

try:
    client = Client()
    print(f"\nZenML Client Info:")
    print(f"  Active store: {client.zen_store.url}")
    print(f"  Store type: {client.zen_store.type}")

    # Try to list pipelines
    pipelines = client.list_pipelines()
    print(f"\n  Total pipelines in server: {len(pipelines)}")

    print("\n✅ ZenML connected successfully!")
except Exception as e:
    print(f"\n❌ Error connecting to ZenML: {e}")
    import traceback
    traceback.print_exc()
