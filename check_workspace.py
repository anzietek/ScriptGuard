"""Check ZenML workspace and project info."""
import os
from dotenv import load_dotenv
load_dotenv()

from zenml.client import Client

client = Client()

# Get active workspace
try:
    active_workspace = client.active_workspace
    print(f"Active Workspace:")
    print(f"  Name: {active_workspace.name}")
    print(f"  ID: {active_workspace.id}")
except Exception as e:
    print(f"Error getting workspace: {e}")

# List all pipelines with details
print(f"\nPipelines:")
pipelines = client.list_pipelines()
for p in pipelines:
    print(f"\n  Pipeline: {p.name}")
    print(f"    ID: {p.id}")
    print(f"    Workspace: {p.workspace.name if hasattr(p, 'workspace') else 'N/A'}")

    # Get runs
    try:
        runs = client.list_pipeline_runs(size=100)
        pipeline_runs = [r for r in runs if r.pipeline and r.pipeline.id == p.id]
        print(f"    Runs: {len(pipeline_runs)}")
        if pipeline_runs:
            latest = pipeline_runs[0]
            print(f"    Latest run: {latest.status} ({latest.created})")
    except Exception as e:
        print(f"    Error getting runs: {e}")

print("\n" + "="*60)
print("Dashboard URL: http://localhost:8237")
print("="*60)
print("\nInstrukcja:")
print("1. Otwórz http://localhost:8237 w przeglądarce")
print("2. Jeśli poprosi o login, użyj:")
print("   Username: adix79")
print("   Password: [sprawdź co ustawiłeś lub użyj default]")
print("3. Po zalogowaniu sprawdź górny lewy róg - czy jesteś w workspace 'default'?")
print("4. Kliknij 'Pipelines' w menu po lewej")
