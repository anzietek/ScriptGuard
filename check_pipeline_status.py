"""Check ZenML pipeline status without emojis."""
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

    # List all pipelines
    pipelines = client.list_pipelines()
    print(f"\n  Total pipelines in server: {len(pipelines)}")

    for i, pipeline in enumerate(pipelines, 1):
        print(f"\n  Pipeline {i}:")
        print(f"    Name: {pipeline.name}")
        print(f"    ID: {pipeline.id}")
        print(f"    Created: {pipeline.created}")
        print(f"    Updated: {pipeline.updated}")

        # Get runs for this pipeline
        try:
            runs = client.list_pipeline_runs(pipeline_id=pipeline.id, size=5)
            print(f"    Total runs: {len(runs)}")
        except TypeError:
            # Try different API
            runs = client.list_pipeline_runs(size=5)
            runs = [r for r in runs if r.pipeline.id == pipeline.id]
            print(f"    Total runs: {len(runs)}")

        if runs:
            print(f"    Latest run:")
            latest = runs[0]
            print(f"      Status: {latest.status}")
            print(f"      Started: {latest.created}")
            print(f"      Run ID: {latest.id}")

    print("\n[SUCCESS] ZenML connected successfully!")
    print(f"\nDashboard: http://localhost:8237")
except Exception as e:
    print(f"\n[ERROR] Error connecting to ZenML: {e}")
    import traceback
    traceback.print_exc()
