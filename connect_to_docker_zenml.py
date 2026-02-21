"""Connect local ZenML client to Docker server."""
import os
from zenml.client import Client

def main():
    # Set environment variable for this session
    os.environ["ZENML_SERVER_URL"] = "http://localhost:8237"

    print("Connecting to Docker ZenML server...")
    print("URL: http://localhost:8237")

    try:
        # Initialize client (will use ZENML_SERVER_URL)
        client = Client()

        # Verify connection
        store_url = client.zen_store.url
        print(f"✓ Connected to: {store_url}")

        # Check project
        project = client.active_project
        print(f"✓ Active project: {project.name}")

        # Check user
        user = client.active_user
        print(f"✓ Active user: {user.name}")

        print("\n[SUCCESS] ZenML client connected to Docker server!")
        print("Your pipelines will now appear in dashboard at http://localhost:8237")

    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Docker ZenML server is running: docker ps | grep zenml")
        print("2. Check server accessibility: curl http://localhost:8237/health")
        print("3. Verify port 8237 is not blocked by firewall")
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
