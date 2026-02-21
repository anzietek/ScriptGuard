"""Set default project for the user via API."""
import requests
import json

zenml_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5NmIwZTM5MC04ODVjLTQzMzUtYWI0MC1kMDczZWJiZjNkYjQiLCJpc3MiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMiLCJhdWQiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMifQ.H5yiIjwWLdmqbHGql1H2eSEwD2wrxjU-LTGWFdv-98o"

base_url = "http://localhost:8237"
headers = {
    "Cookie": f"zenml-server-ba3171c4-1cb3-410b-87c7-4b29e33e105c={zenml_token}",
    "Content-Type": "application/json"
}

user_id = "96b0e390-885c-4335-ab40-d073ebbf3db4"
default_project_id = "76e1b71e-18db-42c9-85da-639839a7fcfa"

print("="*70)
print("SETTING DEFAULT PROJECT FOR USER")
print("="*70)

# Update user's default project
update_data = {
    "default_project_id": default_project_id
}

print(f"\nUpdating user {user_id}...")
print(f"Setting default_project_id to: {default_project_id} (default)")

response = requests.put(
    f"{base_url}/api/v1/users/{user_id}",
    headers=headers,
    json=update_data
)

print(f"\nResponse status: {response.status_code}")

if response.status_code == 200:
    print("[SUCCESS] User default project updated!")

    # Verify
    response = requests.get(f"{base_url}/api/v1/current-user", headers=headers)
    if response.status_code == 200:
        user = response.json()
        default_proj = user.get('body', {}).get('default_project_id')
        print(f"\nVerification:")
        print(f"  User default_project_id: {default_proj}")

        if default_proj == default_project_id:
            print("  [OK] Default project correctly set!")
        else:
            print("  [WARNING] Default project not updated")

    # Now check if pipelines appear
    print("\n" + "="*70)
    print("TESTING PIPELINE QUERY")
    print("="*70)

    response = requests.get(f"{base_url}/api/v1/pipelines", headers=headers)
    if response.status_code == 200:
        data = response.json()
        count = data.get('total', 0)
        print(f"\nPipelines without filter: {count}")

        if count > 0:
            print("[SUCCESS] Pipelines now visible!")
            for item in data.get('items', []):
                print(f"  - {item['name']}")
        else:
            print("[STILL EMPTY] Pipelines query still returns 0")
            print("\nThis might mean dashboard uses a different query")
            print("Try refreshing dashboard and selecting 'default' project manually")

else:
    print(f"[ERROR] Failed to update user")
    print(f"Response: {response.text[:500]}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\n1. Refresh dashboard: Ctrl+Shift+R")
print("2. Click workspace/project selector (top left)")
print("3. Select 'default'")
print("4. Pipeline should appear!")
