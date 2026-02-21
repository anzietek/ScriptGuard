"""Debug why dashboard shows 'No results' even with correct URL."""
import requests
import json

zenml_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5NmIwZTM5MC04ODVjLTQzMzUtYWI0MC1kMDczZWJiZjNkYjQiLCJpc3MiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMiLCJhdWQiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMifQ.H5yiIjwWLdmqbHGql1H2eSEwD2wrxjU-LTGWFdv-98o"

base_url = "http://localhost:8237"
headers = {"Cookie": f"zenml-server-ba3171c4-1cb3-410b-87c7-4b29e33e105c={zenml_token}"}
default_project_id = "76e1b71e-18db-42c9-85da-639839a7fcfa"

print("="*70)
print("DEBUGGING 'No results' ISSUE")
print("="*70)

# Test different API query combinations that dashboard might use
test_queries = [
    # Basic queries
    "/api/v1/pipelines",
    "/api/v1/pipelines?page=1",
    f"/api/v1/pipelines?project={default_project_id}",
    f"/api/v1/pipelines?project_id={default_project_id}",
    f"/api/v1/pipelines?workspace={default_project_id}",

    # With sorting
    "/api/v1/pipelines?page=1&sort_by=desc:latest_run",
    f"/api/v1/pipelines?page=1&sort_by=desc:latest_run&project={default_project_id}",

    # With hydration
    f"/api/v1/pipelines?project={default_project_id}&hydrate=true",

    # Pipeline snapshots (alternative endpoint)
    "/api/v1/pipeline_snapshots",
    f"/api/v1/pipeline_snapshots?project={default_project_id}",
    "/api/v1/pipeline_snapshots?named_only=true",
    f"/api/v1/pipeline_snapshots?named_only=true&project={default_project_id}",
]

found_pipelines = []

for query in test_queries:
    url = f"{base_url}{query}"
    try:
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()

            # Check for results
            total = 0
            if isinstance(data, dict):
                total = data.get('total', 0)
                if total == 0 and 'items' in data:
                    total = len(data['items'])
            elif isinstance(data, list):
                total = len(data)

            status = f"[{total} results]" if total > 0 else "[EMPTY]"
            print(f"{status:12} {query}")

            if total > 0:
                found_pipelines.append((query, total, data))
        else:
            print(f"[{response.status_code}] {query}")

    except Exception as e:
        print(f"[ERROR] {query}: {str(e)[:50]}")

print("\n" + "="*70)
print("QUERIES THAT RETURNED PIPELINES")
print("="*70)

if found_pipelines:
    for query, count, data in found_pipelines:
        print(f"\n{query}")
        print(f"  Total: {count}")

        # Show pipeline names
        items = data.get('items', []) if isinstance(data, dict) else data
        for item in items[:3]:  # Show first 3
            name = item.get('name', 'N/A') if isinstance(item, dict) else 'N/A'
            print(f"    - {name}")
else:
    print("\nZADNE query nie zwrocilo pipelines!")

print("\n" + "="*70)
print("CHECKING BROWSER NETWORK TAB")
print("="*70)
print("\nW przegladarce na stronie http://localhost:8237/projects/default/pipelines")
print("\nOtworz DevTools (F12) -> Network tab -> odswież stronę")
print("\nSzukaj requestow do /api/v1/pipelines* lub /api/v1/pipeline_snapshots*")
print("\nWyslij mi DOKLADNY URL z requestu ktory dashboard wysyla!")
print("Np: /api/v1/pipelines?page=1&size=20&sort_by=...")
