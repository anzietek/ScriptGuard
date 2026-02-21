"""Script to create and switch to default ZenML project."""
from zenml.client import Client

def main():
    client = Client()

    # Try to create default project
    try:
        new_project = client.create_project(
            name="default",
            description="Default project for ZenML community edition"
        )
        print(f"[OK] Created project: {new_project.name} (ID: {new_project.id})")
    except Exception as e:
        print(f"Could not create project: {e}")
        print("Project might already exist or creation is restricted in community edition")
        return False

    # Try to set it as active
    try:
        client.set_active_project(new_project.id)
        print(f"[OK] Switched to project: {new_project.name}")

        # Verify
        current_project = client.active_project
        print(f"[OK] Active project is now: {current_project.name}")
        return True
    except Exception as e:
        print(f"Could not switch to project: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
