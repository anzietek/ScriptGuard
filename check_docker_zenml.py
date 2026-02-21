"""Check if Docker ZenML server is running and accessible."""
import requests
import subprocess

def main():
    print("=" * 60)
    print("Docker ZenML Server Health Check")
    print("=" * 60)

    # 1. Check if container is running
    print("\n[1. Container Status]")
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "publish=8237", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.stdout.strip():
            print(f"  ✓ Container running: {result.stdout.strip()}")
        else:
            print("  ✗ No container found on port 8237")
            print("\n  Start with: docker-compose -f docker/docker-compose.dev.yml --profile with-zenml up -d")
            return False

    except Exception as e:
        print(f"  ✗ Failed to check Docker: {e}")
        return False

    # 2. Check if port is accessible
    print("\n[2. Port Accessibility]")
    try:
        response = requests.get("http://localhost:8237/health", timeout=5)

        if response.status_code == 200:
            print(f"  ✓ Server responding: {response.json()}")
        else:
            print(f"  ✗ Server returned status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("  ✗ Cannot connect to localhost:8237")
        print("  Check if server is running: docker logs <container-name>")
        return False
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False

    # 3. Check API version
    print("\n[3. API Version]")
    try:
        response = requests.get("http://localhost:8237/api/v1/info", timeout=5)

        if response.status_code == 200:
            info = response.json()
            print(f"  ✓ ZenML Version: {info.get('version', 'unknown')}")
            print(f"  ✓ Server ID: {info.get('server_id', 'unknown')}")
        else:
            print("  ⚠ Could not get version info")

    except Exception as e:
        print(f"  ⚠ Version check failed: {e}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Docker ZenML server is ready!")
    print("Dashboard: http://localhost:8237")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
