"""Debug dashboard visibility issue."""
import os
from dotenv import load_dotenv
load_dotenv()

from zenml.client import Client

client = Client()

print("="*70)
print("ZENML CONNECTION INFO")
print("="*70)

# User info
user = client.active_user
print(f"\nCurrent User: {user.name}")
print(f"User ID: {user.id}")
print(f"Is Admin: {user.is_admin}")

# Store info
print(f"\nStore URL: {client.zen_store.url}")
print(f"Store Type: {client.zen_store.type}")

# Get store config
try:
    config = client.zen_store.config
    print(f"\nStore Config:")
    print(f"  Type: {config.type}")
    print(f"  URL: {config.url if hasattr(config, 'url') else 'N/A'}")
except Exception as e:
    print(f"\nStore Config Error: {e}")

# Check all users in system
try:
    # This might not work depending on API version
    print(f"\n" + "="*70)
    print("ALL USERS IN SYSTEM")
    print("="*70)
    # Try to access user management
    from zenml.zen_stores.rest_zen_store import RestZenStore
    if isinstance(client.zen_store, RestZenStore):
        print("\nConnected to REST store - checking users...")
        # Can't easily list all users without admin API
        print("(User listing requires admin API access)")
except Exception as e:
    print(f"\nUsers check error: {e}")

# Pipeline info
print(f"\n" + "="*70)
print("PIPELINES")
print("="*70)
pipelines = client.list_pipelines()
print(f"\nTotal pipelines: {len(pipelines)}")

for p in pipelines:
    print(f"\nPipeline: {p.name}")
    print(f"  ID: {p.id}")
    print(f"  Created: {p.created}")
    print(f"  Updated: {p.updated}")

    # Try to get workspace info
    try:
        if hasattr(p, 'body') and p.body:
            print(f"  Body workspace: {p.body.workspace if hasattr(p.body, 'workspace') else 'N/A'}")
        if hasattr(p, 'workspace'):
            print(f"  Workspace: {p.workspace}")
    except Exception as e:
        pass

# Runs info
print(f"\n" + "="*70)
print("PIPELINE RUNS")
print("="*70)

runs = client.list_pipeline_runs(size=10)
print(f"\nTotal runs (last 10): {len(runs)}")

for r in runs:
    print(f"\nRun: {r.name}")
    print(f"  ID: {r.id}")
    print(f"  Pipeline: {r.pipeline.name if r.pipeline else 'N/A'}")
    print(f"  Status: {r.status}")
    print(f"  Created: {r.created}")

    # Try to get workspace
    try:
        if hasattr(r, 'workspace'):
            print(f"  Workspace: {r.workspace}")
        if hasattr(r, 'body') and hasattr(r.body, 'workspace'):
            print(f"  Body workspace: {r.body.workspace}")
    except:
        pass

print(f"\n" + "="*70)
print("INSTRUKCJE")
print("="*70)
print("\n1. W dashboard, sprawdz GORNY LEWY ROG")
print("   Czy widzisz nazwe workspace (np 'default')?")
print("\n2. Kliknij na nazwe workspace - czy sa inne workspace?")
print("\n3. Czy w lewym menu widzisz 'Pipelines'? Kliknij tam")
print("\n4. Czy sa jakies filtry aktywne? (Search bar, date filters)")
print("\n5. Sprobuj odswiezyc strone: Ctrl+Shift+R")
