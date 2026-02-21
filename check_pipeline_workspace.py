"""Check which workspace the pipeline is in."""
import os
from dotenv import load_dotenv
load_dotenv()

from zenml.client import Client

client = Client()

print("="*70)
print("WORKSPACE DIAGNOSIS")
print("="*70)

# Get current active workspace name
try:
    # Different ways to get workspace depending on ZenML version
    store = client.zen_store

    # Try to get workspace info from store
    print(f"\nStore URL: {store.url}")

    # Get all pipelines with their metadata
    pipelines = client.list_pipelines()
    print(f"\nTotal pipelines found: {len(pipelines)}")

    for p in pipelines:
        print(f"\nPipeline: {p.name}")
        print(f"  ID: {p.id}")

        # Try different ways to get workspace info
        workspace_info = None

        # Method 1: Check body attribute
        if hasattr(p, 'body'):
            body = p.body
            if hasattr(body, 'workspace'):
                workspace_info = body.workspace
                print(f"  Workspace (from body): {workspace_info}")

        # Method 2: Check user attribute
        if hasattr(p, 'user'):
            user = p.user
            print(f"  User: {user.name if hasattr(user, 'name') else user}")

        # Method 3: Get the raw response
        print(f"  Raw attributes: {[attr for attr in dir(p) if not attr.startswith('_')][:10]}")

    # Try to list workspaces
    print("\n" + "="*70)
    print("CHECKING FOR WORKSPACES")
    print("="*70)

    # Try to get workspace from config
    from zenml.config.global_config import GlobalConfiguration
    gc = GlobalConfiguration()

    print(f"\nGlobal Config Store:")
    print(f"  Store URL: {gc.store_configuration.url if hasattr(gc, 'store_configuration') else 'N/A'}")
    print(f"  Store Type: {gc.store_configuration.type if hasattr(gc, 'store_configuration') else 'N/A'}")

    # Check active workspace
    try:
        active_ws_name = gc.get_active_workspace_name() if hasattr(gc, 'get_active_workspace_name') else None
        if active_ws_name:
            print(f"  Active Workspace: {active_ws_name}")
        else:
            print(f"  Active Workspace: Cannot determine")
    except:
        print(f"  Active Workspace: Error getting workspace name")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DASHBOARD INSTRUCTIONS")
print("="*70)
print("\nW dashboard (http://localhost:8237):")
print("1. Sprawdz GORNY LEWY ROG ekranu")
print("2. Powinienes widziec nazwe workspace (np 'default' lub inna)")
print("3. Kliknij na te nazwe - rozwinie sie menu z lista workspace")
print("4. Sprawdz czy sa inne workspace dostepne")
print("5. Sprobuj kliknac kazde workspace i sprawdz czy pipelines sie pojawia")
print("\nJeśli w górnym rogu NIE MA nazwy workspace:")
print("- Kliknij na swoj username (prawy gorny rog)")
print("- Sprawdz czy jest opcja 'Switch Workspace'")
