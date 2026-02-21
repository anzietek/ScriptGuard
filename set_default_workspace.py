"""Set 'default' as active workspace for ZenML."""
import os
from dotenv import load_dotenv
load_dotenv()

from zenml.client import Client

print("="*70)
print("SETTING ACTIVE WORKSPACE TO 'default'")
print("="*70)

client = Client()

# Get current state
print("\nCurrent state:")
try:
    # Try different methods to get active workspace
    from zenml.config.global_config import GlobalConfiguration
    gc = GlobalConfiguration()

    # Check store configuration
    store_config = gc.store_configuration
    print(f"  Store URL: {store_config.url}")
    print(f"  Store Type: {store_config.type}")

except Exception as e:
    print(f"  Error: {e}")

# List available workspaces
print("\nAvailable workspaces:")
try:
    # Try to get workspaces through the store
    response = client.zen_store._get("/api/v1/workspaces")
    workspaces = response.get('items', [])

    for ws in workspaces:
        name = ws.get('name', 'N/A')
        ws_id = ws.get('id', 'N/A')
        print(f"  - {name} (ID: {ws_id})")

except Exception as e:
    print(f"  Method 1 failed: {e}")

    # Alternative method
    try:
        import requests
        response = requests.get(
            "http://localhost:8237/api/v1/workspaces",
            headers={"Authorization": f"Bearer {os.getenv('ZENML_API_KEY', '')}"}
        )
        if response.status_code == 200:
            data = response.json()
            for ws in data.get('items', []):
                print(f"  - {ws['name']} (ID: {ws['id']})")
    except Exception as e2:
        print(f"  Method 2 failed: {e2}")

# Set default workspace as active
print("\nSetting 'default' as active workspace...")

try:
    # Method 1: Use client.set_active_workspace()
    if hasattr(client, 'set_active_workspace'):
        client.set_active_workspace('default')
        print("  ✓ Successfully set via client.set_active_workspace()")
    else:
        print("  ✗ client.set_active_workspace() not available")

        # Method 2: Use GlobalConfiguration
        from zenml.config.global_config import GlobalConfiguration
        gc = GlobalConfiguration()

        if hasattr(gc, 'set_active_workspace'):
            gc.set_active_workspace('default')
            print("  ✓ Successfully set via GlobalConfiguration.set_active_workspace()")
        else:
            print("  ✗ GlobalConfiguration.set_active_workspace() not available")

            # Method 3: Use zenml CLI programmatically
            import subprocess
            result = subprocess.run(
                ['zenml', 'workspace', 'set', 'default'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("  ✓ Successfully set via zenml CLI")
            else:
                print(f"  ✗ zenml CLI failed: {result.stderr}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Verify the change
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

try:
    # Create a new client instance to check
    client2 = Client()

    # Try to get active workspace
    print("\nChecking active workspace...")

    # Check if pipeline is now visible
    pipelines = client2.list_pipelines()
    print(f"  Pipelines visible: {len(pipelines)}")

    if len(pipelines) > 0:
        print("\n  ✓ SUCCESS! Pipeline is now visible!")
        for p in pipelines:
            print(f"    - {p.name}")
    else:
        print("\n  ⚠ Warning: Still no pipelines visible")
        print("  This might mean the workspace switch didn't work")

except Exception as e:
    print(f"  Error during verification: {e}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\n1. Refresh dashboard (Ctrl+F5)")
print("2. Verify you're in 'default' workspace (górny lewy róg)")
print("3. Check if pipeline appears in the list")
print("\nJeśli nadal nie widać:")
print("  - Spróbuj 'zenml workspace set default' w terminalu")
print("  - Lub podaj mi screenshot dashboardu")
