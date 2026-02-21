"""Query ZenML API directly to get workspace info."""
import os
from dotenv import load_dotenv
load_dotenv()

from zenml.client import Client
import json

client = Client()

# Get the REST store and make direct API calls
store = client.zen_store

print("="*70)
print("DIRECT API QUERY")
print("="*70)

try:
    # Get pipeline with full details
    pipelines = client.list_pipelines()

    for p in pipelines:
        print(f"\nPipeline: {p.name}")

        # Get the full body
        if hasattr(p, 'body'):
            body = p.body
            print(f"\nBody attributes:")
            for attr in dir(body):
                if not attr.startswith('_'):
                    try:
                        value = getattr(body, attr)
                        if not callable(value):
                            print(f"  {attr}: {value}")
                    except:
                        pass

        # Try to get the pipeline's workspace using the store
        try:
            # Get pipeline by ID with full details
            pipeline_id = p.id
            print(f"\nFetching pipeline details for ID: {pipeline_id}")

            # Use the zen_store to get more details
            from zenml.enums import StrEnum

            # Try to access the workspace field
            if hasattr(p, 'workspace'):
                ws = p.workspace
                print(f"  Workspace object: {ws}")
                if hasattr(ws, 'name'):
                    print(f"  Workspace name: {ws.name}")
                if hasattr(ws, 'id'):
                    print(f"  Workspace ID: {ws.id}")

        except Exception as e:
            print(f"  Error getting workspace: {e}")

    # Try to list all workspaces
    print("\n" + "="*70)
    print("ATTEMPTING TO LIST WORKSPACES")
    print("="*70)

    try:
        # Different methods to try
        methods = [
            'list_workspaces',
            'get_workspaces',
            'workspaces',
        ]

        for method_name in methods:
            if hasattr(store, method_name):
                print(f"\nTrying: store.{method_name}()")
                method = getattr(store, method_name)
                result = method() if callable(method) else method
                print(f"  Result: {result}")
                break
    except Exception as e:
        print(f"  Error: {e}")

    # Check client for workspace methods
    print("\n" + "="*70)
    print("CLIENT WORKSPACE METHODS")
    print("="*70)

    workspace_methods = [attr for attr in dir(client) if 'workspace' in attr.lower()]
    print(f"Methods with 'workspace': {workspace_methods}")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("NEXT STEP")
print("="*70)
print("\nW dashboard zrob screenshot:")
print("1. Calego ekranu z lista pipelines (pusta)")
print("2. Gornego lewego rogu (workspace selector jesli jest)")
print("3. Menu po lewej stronie")
print("\nTo pomoze mi zrozumiec co dashboard pokazuje")
