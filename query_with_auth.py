"""Query API with authentication cookie."""
import requests
import json

# Extract ZenML auth token from cookie
zenml_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5NmIwZTM5MC04ODVjLTQzMzUtYWI0MC1kMDczZWJiZjNkYjQiLCJpc3MiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMiLCJhdWQiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMifQ.H5yiIjwWLdmqbHGql1H2eSEwD2wrxjU-LTGWFdv-98o"

base_url = "http://localhost:8237"

# Set up headers with authentication
headers = {
    "Cookie": f"zenml-server-ba3171c4-1cb3-410b-87c7-4b29e33e105c={zenml_token}",
    "Content-Type": "application/json"
}

print("="*70)
print("AUTHENTICATED API QUERIES")
print("="*70)

# Query various endpoints
endpoints = [
    "/api/v1/current-user",
    "/api/v1/workspaces",
    "/api/v1/workspaces?hydrate=true",
    "/api/v1/pipelines",
    "/api/v1/pipelines?page=1&size=50",
]

for endpoint in endpoints:
    url = f"{base_url}{endpoint}"
    print(f"\n{'='*70}")
    print(f"GET {endpoint}")
    print(f"{'='*70}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if "workspaces" in endpoint:
                print(f"\nWorkspaces response:")
                print(json.dumps(data, indent=2)[:1000])

            elif "pipelines" in endpoint:
                print(f"\nPipelines response:")
                if isinstance(data, dict):
                    if 'items' in data:
                        print(f"  Total items: {data.get('total', 'N/A')}")
                        print(f"  Items count: {len(data['items'])}")
                        for item in data['items']:
                            name = item.get('name', 'N/A')
                            workspace = item.get('workspace', {})
                            ws_name = workspace.get('name', 'N/A') if isinstance(workspace, dict) else 'N/A'
                            print(f"    - {name} (workspace: {ws_name})")
                    else:
                        print(json.dumps(data, indent=2)[:500])
                else:
                    print(f"  Response: {data}")

            elif "current-user" in endpoint:
                print(f"\nCurrent user:")
                user_name = data.get('name', 'N/A')
                user_id = data.get('id', 'N/A')
                print(f"  Name: {user_name}")
                print(f"  ID: {user_id}")

        else:
            print(f"Error: {response.text[:200]}")

    except Exception as e:
        print(f"Exception: {e}")

print(f"\n{'='*70}")
print("ANALIZA")
print(f"{'='*70}")
print("\nSprawdz wyzej:")
print("1. Ile workspaces masz dostepnych?")
print("2. W ktorym workspace jest pipeline?")
print("3. Czy API zwraca pipeline z prawidlowym workspace name?")
