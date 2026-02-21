"""Find which workspace the pipeline is in."""
import requests
import json

zenml_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5NmIwZTM5MC04ODVjLTQzMzUtYWI0MC1kMDczZWJiZjNkYjQiLCJpc3MiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMiLCJhdWQiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMifQ.H5yiIjwWLdmqbHGql1H2eSEwD2wrxjU-LTGWFdv-98o"

base_url = "http://localhost:8237"
headers = {
    "Cookie": f"zenml-server-ba3171c4-1cb3-410b-87c7-4b29e33e105c={zenml_token}",
    "Content-Type": "application/json"
}

print("="*70)
print("SEARCHING FOR PIPELINE IN EACH WORKSPACE")
print("="*70)

workspaces = [
    {"name": "scriptguard", "id": "6903852e-53fe-4e47-8a95-43b94919b6f1"},
    {"name": "default", "id": "76e1b71e-18db-42c9-85da-639839a7fcfa"}
]

for ws in workspaces:
    print(f"\n{'='*70}")
    print(f"Workspace: {ws['name']}")
    print(f"{'='*70}")

    # Try querying pipelines with workspace filter
    endpoints = [
        f"/api/v1/pipelines?workspace_id={ws['id']}",
        f"/api/v1/pipelines?workspace={ws['name']}",
        f"/api/v1/workspaces/{ws['id']}/pipelines",
    ]

    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                data = response.json()

                if isinstance(data, dict) and 'items' in data:
                    count = len(data['items'])
                    if count > 0:
                        print(f"\n  ✓ FOUND {count} pipeline(s) via: {endpoint}")
                        for item in data['items']:
                            print(f"    - {item.get('name', 'N/A')}")
                            print(f"      ID: {item.get('id', 'N/A')}")
                elif isinstance(data, list):
                    count = len(data)
                    if count > 0:
                        print(f"\n  ✓ FOUND {count} pipeline(s) via: {endpoint}")
            elif response.status_code == 404:
                pass  # Endpoint doesn't exist
            else:
                print(f"  - {endpoint}: Status {response.status_code}")

        except Exception as e:
            pass

# Also try getting the specific pipeline by ID
print(f"\n{'='*70}")
print("DIRECT PIPELINE QUERY BY ID")
print(f"{'='*70}")

pipeline_id = "7f56b96e-3ad9-46f1-ad69-e7e9a4832563"
url = f"{base_url}/api/v1/pipelines/{pipeline_id}"

try:
    response = requests.get(url, headers=headers, timeout=5)
    print(f"\nGET /api/v1/pipelines/{pipeline_id}")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nPipeline found!")
        print(f"  Name: {data.get('name', 'N/A')}")

        # Check workspace field
        if 'workspace' in data:
            ws = data['workspace']
            if isinstance(ws, dict):
                print(f"  Workspace name: {ws.get('name', 'N/A')}")
                print(f"  Workspace ID: {ws.get('id', 'N/A')}")
            else:
                print(f"  Workspace: {ws}")

        # Check body for workspace
        if 'body' in data and isinstance(data['body'], dict):
            body = data['body']
            if 'workspace' in body:
                print(f"  Body workspace: {body['workspace']}")

        print(f"\nFull response (first 1000 chars):")
        print(json.dumps(data, indent=2)[:1000])

except Exception as e:
    print(f"Error: {e}")

print(f"\n{'='*70}")
print("ROZWIAZANIE")
print(f"{'='*70}")
print("\nW dashboard:")
print("1. Znajdz workspace selector (gorny lewy rog)")
print("2. Kliknij i przełącz na workspace gdzie jest pipeline")
print("3. Lista pipelines powinna się pojawić!")
