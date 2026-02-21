"""Verify pipeline will connect to Docker ZenML server."""
import os
from zenml.client import Client

def main():
    print("=" * 60)
    print("Pipeline Connection Verification")
    print("=" * 60)

    # Load .env
    from pathlib import Path
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("ZENML_SERVER_URL="):
                    url = line.split("=", 1)[1].strip()
                    os.environ["ZENML_SERVER_URL"] = url
                    print(f"\n[OK] Found ZENML_SERVER_URL in .env: {url}")
                    break
    else:
        print("\n[WARNING] No .env file found")

    # Check environment variable
    zenml_url = os.getenv("ZENML_SERVER_URL")
    if not zenml_url:
        print("\n[ERROR] ZENML_SERVER_URL not set!")
        print("Add to .env: ZENML_SERVER_URL=http://localhost:8237")
        return False

    print(f"\n[Checking Connection]")
    print(f"  Target URL: {zenml_url}")

    try:
        # Test client initialization
        client = Client()
        store_url = str(client.zen_store.url)

        print(f"  Connected to: {store_url}")

        # Verify it's the Docker server
        if "localhost:8237" in store_url or "127.0.0.1:8237" in store_url:
            print(f"\n[SUCCESS] ✓ Pipeline WILL use Docker server")
            print(f"  Dashboard: http://localhost:8237")
            print(f"  Project: {client.active_project.name}")
            print(f"  User: {client.active_user.name}")
            return True
        else:
            print(f"\n[WARNING] Connected to different server: {store_url}")
            print(f"  Expected: http://localhost:8237")
            print(f"  Check your ZENML_SERVER_URL setting")
            return False

    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. docker ps | grep zenml  # Check if server is running")
        print("2. curl http://localhost:8237/health  # Test server")
        print("3. Check .env has: ZENML_SERVER_URL=http://localhost:8237")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 60)
        print("Ready to run pipeline: python src/main.py --config config.yaml")
        print("=" * 60)
    exit(0 if success else 1)
