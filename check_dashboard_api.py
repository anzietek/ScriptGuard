"""Check what the dashboard API actually returns with user's cookies."""
import requests
import json

# User's auth token
zenml_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5NmIwZTM5MC04ODVjLTQzMzUtYWI0MC1kMDczZWJiZjNkYjQiLCJpc3MiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMiLCJhdWQiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMifQ.H5yiIjwWLdmqbHGql1H2eSEwD2wrxjU-LTGWFdv-98o"

base_url = "http://localhost:8237"
headers = {
    "Cookie": f"zenml-server-ba3171c4-1cb3-410b-87c7-4b29e33e105c={zenml_token}",
    "Content-Type": "application/json"
}

print("="*70)
print("CHECKING WHAT DASHBOARD SEES")
print("="*70)

# 1. Check current user and their active project
print("\n1. Current user info:")
response = requests.get(f"{base_url}/api/v1/current-user", headers=headers)
if response.status_code == 200:
    user = response.json()
    print(f"   User: {user.get('name')}")
    print(f"   User ID: {user.get('id')}")

    # Check if there's a default_project_id
    body = user.get('body', {})
    default_project = body.get('default_project_id')
    print(f"   Default project ID: {default_project}")

# 2. List all projects
print("\n2. Available projects:")
response = requests.get(f"{base_url}/api/v1/workspaces", headers=headers)
if response.status_code == 200:
    data = response.json()
    projects = data.get('items', [])
    for p in projects:
        print(f"   - {p['name']} (ID: {p['id']})")

# 3. Query pipelines WITHOUT workspace filter (what dashboard might do)
print("\n3. Pipelines query (no filter):")
response = requests.get(f"{base_url}/api/v1/pipelines", headers=headers)
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    count = data.get('total', 0)
    print(f"   Total: {count}")
    if count > 0:
        for item in data.get('items', []):
            print(f"   - {item['name']}")
    else:
        print("   [EMPTY - This is what you see in dashboard!]")

# 4. Query pipelines FOR EACH workspace explicitly
print("\n4. Pipelines per workspace:")
workspace_ids = {
    "scriptguard": "6903852e-53fe-4e47-8a95-43b94919b6f1",
    "default": "76e1b71e-18db-42c9-85da-639839a7fcfa"
}

for ws_name, ws_id in workspace_ids.items():
    # Try different query patterns
    patterns = [
        f"/api/v1/pipelines?workspace={ws_id}",
        f"/api/v1/pipelines?workspace_id={ws_id}",
        f"/api/v1/pipelines?project={ws_id}",
        f"/api/v1/pipelines?project_id={ws_id}",
    ]

    found = False
    for pattern in patterns:
        response = requests.get(f"{base_url}{pattern}", headers=headers)
        if response.status_code == 200:
            data = response.json()
            count = data.get('total', 0)
            if count > 0:
                print(f"   {ws_name}: {count} pipelines (via {pattern})")
                for item in data.get('items', []):
                    print(f"     - {item['name']}")
                found = True
                break

    if not found:
        print(f"   {ws_name}: No query pattern worked")

# 5. Check the specific pipeline by ID
print("\n5. Direct pipeline query:")
pipeline_id = "7f56b96e-3ad9-46f1-ad69-e7e9a4832563"
response = requests.get(f"{base_url}/api/v1/pipelines/{pipeline_id}", headers=headers)
if response.status_code == 200:
    data = response.json()
    print(f"   Name: {data['name']}")
    project_id = data['body'].get('project_id')
    print(f"   Project ID: {project_id}")

    # Find which project this is
    for name, pid in workspace_ids.items():
        if pid == project_id:
            print(f"   Project name: {name}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Try to figure out what's wrong
print("\nPossible issues:")
print("1. Dashboard queries /api/v1/pipelines without workspace filter")
print("2. API returns empty when no workspace is specified")
print("3. Need to explicitly filter by workspace/project")

print("\nSOLUTION:")
print("Dashboard needs to have 'default' project selected as active")
print("This should be in browser session/cookies, not just CLI config")
