"""Check ZenML authentication and store info."""
import os
from dotenv import load_dotenv
load_dotenv()

from zenml.client import Client

client = Client()

print(f"ZenML Store URL: {client.zen_store.url}")
print(f"Store Type: {client.zen_store.type}")

# Check active user
try:
    active_user = client.active_user
    print(f"\nActive User: {active_user.name}")
    print(f"User ID: {active_user.id}")
    print(f"Is Admin: {active_user.is_admin}")
except Exception as e:
    print(f"\nError getting active user: {e}")

# Check if we're connected to the server
try:
    store_info = client.zen_store.get_store_info()
    print(f"\nStore Info:")
    print(f"  Server URL: {store_info.server_url if hasattr(store_info, 'server_url') else 'N/A'}")
    print(f"  Version: {store_info.version}")
except Exception as e:
    print(f"\nError getting store info: {e}")

# List workspaces
try:
    workspaces = client.list_workspaces()
    print(f"\nWorkspaces ({len(workspaces)}):")
    for ws in workspaces:
        print(f"  - {ws.name} (ID: {ws.id})")

    active_workspace = client.active_workspace
    print(f"\nActive Workspace: {active_workspace.name}")
except Exception as e:
    print(f"\nError listing workspaces: {e}")

# Check pipelines in active workspace
try:
    pipelines = client.list_pipelines()
    print(f"\nPipelines in active workspace: {len(pipelines)}")
    for p in pipelines:
        print(f"  - {p.name}")
except Exception as e:
    print(f"\nError listing pipelines: {e}")
