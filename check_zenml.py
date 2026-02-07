"""
ZenML Installation Verification and Setup
Run this script to ensure ZenML is properly installed and configured.
"""

import sys


def check_zenml_installation() -> bool:
    """Check if ZenML is installed and working."""
    print("=" * 60)
    print("ZenML Installation Check")
    print("=" * 60)
    print()

    try:
        import zenml
        print(f"✓ ZenML installed: version {zenml.__version__}")
        return True
    except ImportError as e:
        print(f"✗ ZenML not installed: {e}")
        print()
        print("Install ZenML with:")
        print("  uv pip install 'zenml[server]'")
        print("  or")
        print("  pip install 'zenml[server]'")
        return False


def check_zenml_initialization() -> bool:
    """Check if ZenML is initialized."""
    from pathlib import Path

    print()
    print("-" * 60)
    print("ZenML Initialization Check")
    print("-" * 60)

    zen_dir = Path(".zen")
    if zen_dir.exists():
        print(f"✓ ZenML initialized in: {zen_dir.absolute()}")
        return True
    else:
        print("✗ ZenML not initialized in this directory")
        print()
        print("Initialize ZenML with:")
        print("  uv run zenml init")
        print("  or")
        print("  zenml init")
        return False


def get_zenml_status() -> None:
    """Get ZenML status and configuration."""
    import subprocess

    print()
    print("-" * 60)
    print("ZenML Status")
    print("-" * 60)

    try:
        result = subprocess.run(
            ["zenml", "status"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Timeout while getting ZenML status")
    except FileNotFoundError:
        print("✗ zenml command not found in PATH")
        print("Try running with: uv run zenml status")
    except Exception as e:
        print(f"✗ Error getting ZenML status: {e}")


def get_zenml_stacks() -> None:
    """List available ZenML stacks."""
    import subprocess

    print()
    print("-" * 60)
    print("ZenML Stacks")
    print("-" * 60)

    try:
        result = subprocess.run(
            ["zenml", "stack", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(result.stdout)
    except subprocess.TimeoutExpired:
        print("✗ Timeout while listing ZenML stacks")
    except FileNotFoundError:
        print("✗ zenml command not found in PATH")
    except Exception as e:
        print(f"✗ Error listing ZenML stacks: {e}")


def check_zenml_server() -> None:
    """Check if ZenML server is running."""
    import os
    import requests

    print()
    print("-" * 60)
    print("ZenML Server Check")
    print("-" * 60)

    server_url = os.getenv("ZENML_SERVER_URL")

    if not server_url:
        print("ℹ No ZENML_SERVER_URL configured (using local ZenML)")
        print()
        print("To use a remote ZenML server:")
        print("  1. Set ZENML_SERVER_URL in .env")
        print("  2. Connect with: zenml connect --url <SERVER_URL>")
        return

    print(f"Checking server at: {server_url}")

    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ ZenML server is accessible")
        else:
            print(f"✗ ZenML server returned status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to ZenML server: {e}")


def show_setup_instructions() -> None:
    """Show setup instructions."""
    print()
    print("=" * 60)
    print("ZenML Setup Instructions")
    print("=" * 60)
    print()
    print("Option 1: Local ZenML (Recommended for development)")
    print("  1. Install: uv pip install 'zenml[server]'")
    print("  2. Initialize: uv run zenml init")
    print("  3. Start server: uv run zenml up")
    print("  4. Access dashboard: http://localhost:8237")
    print()
    print("Option 2: Remote ZenML Server")
    print("  1. Install: uv pip install 'zenml[server]'")
    print("  2. Initialize: uv run zenml init")
    print("  3. Connect: uv run zenml connect --url <SERVER_URL>")
    print("  4. Set ZENML_SERVER_URL in .env")
    print()
    print("Option 3: Cloud ZenML (ZenML Cloud/Pro)")
    print("  1. Sign up at: https://zenml.io")
    print("  2. Get connection string")
    print("  3. Connect: uv run zenml connect --url <CLOUD_URL>")
    print()


def main() -> int:
    """Main check routine."""
    if not check_zenml_installation():
        show_setup_instructions()
        return 1

    initialized = check_zenml_initialization()

    if initialized:
        get_zenml_status()
        get_zenml_stacks()

    check_zenml_server()

    if not initialized:
        print()
        print("Please initialize ZenML with: uv run zenml init")
        return 1

    print()
    print("=" * 60)
    print("✓ ZenML is ready!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user.")
        sys.exit(1)
