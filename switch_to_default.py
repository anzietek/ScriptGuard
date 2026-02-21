"""Script to switch to default ZenML project."""
from zenml.client import Client

def main():
    client = Client()

    print(f"Current project: {client.active_project.name}")

    # Try to get the default project
    try:
        projects = client.list_projects()
        print(f"\nAll projects ({len(projects.items)}):")
        for p in projects.items:
            print(f"  - {p.name} (ID: {p.id})")

        # Try to find default project
        default_project = None
        for p in projects.items:
            if p.name.lower() == "default":
                default_project = p
                break

        if not default_project:
            print("\nTrying to get 'default' project by name...")
            try:
                default_project = client.get_project("default")
                print(f"Found default project: {default_project.name}")
            except Exception as e:
                print(f"Could not get default project: {e}")
                return False

        # Switch to default project
        print(f"\nSwitching to project: {default_project.name}")
        client.set_active_project(default_project.id)

        # Verify
        current_project = client.active_project
        print(f"[OK] Active project is now: {current_project.name}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
