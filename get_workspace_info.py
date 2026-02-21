"""Get workspace information for dashboard URL."""
import os
from dotenv import load_dotenv
load_dotenv()

from zenml.client import Client
import requests

client = Client()

# Get workspace info from ZenML store
print("Getting workspace information...")
print("="*70)

# Try to get workspace from API
try:
    response = requests.get("http://localhost:8237/api/v1/workspaces", timeout=5)

    if response.status_code == 401:
        print("API requires authentication")
        print("\nTrying with ZenML client credentials...")
    elif response.status_code == 200:
        data = response.json()
        print(f"Workspaces from API: {data}")
except Exception as e:
    print(f"API call failed: {e}")

# Get info from ZenML client
print("\n" + "="*70)
print("ZenML Client Info:")
print("="*70)

try:
    # Get the zen store
    store = client.zen_store

    # Try to get workspace info directly from store
    print(f"Store URL: {store.url}")
    print(f"Store Type: {store.type}")

    # Check if we can access workspace info
    from zenml.client import Client as ZenMLClient

    # Try to construct proper dashboard URL
    print("\n" + "="*70)
    print("DASHBOARD URLS TO TRY:")
    print("="*70)

    # Different possible URL patterns for ZenML dashboard
    urls = [
        "http://localhost:8237/",
        "http://localhost:8237/#/pipelines",
        "http://localhost:8237/dashboard",
        "http://localhost:8237/dashboard/pipelines",
        "http://localhost:8237/login",
    ]

    for url in urls:
        print(f"  {url}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Otwórz: http://localhost:8237/")
    print("   - Jeśli widzisz login screen - zaloguj się")
    print("   - Jeśli widzisz dashboard - opisz co widzisz")
    print("   - Jeśli 404 - sprawdź console (F12) dla błędów")

    print("\n2. Po zalogowaniu sprawdź URL bar:")
    print("   - Czy URL zmienił się na coś w stylu:")
    print("     http://localhost:8237/workspaces/XXXXX/...")
    print("   - Skopiuj dokładny URL i wyślij mi")

    print("\n3. Sprawdź czy w dashboard jest menu:")
    print("   - Home")
    print("   - Pipelines")
    print("   - Runs")
    print("   - Stacks")
    print("   etc.")

except Exception as e:
    print(f"Error getting workspace info: {e}")
    import traceback
    traceback.print_exc()
