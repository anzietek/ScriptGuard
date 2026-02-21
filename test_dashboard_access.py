"""Test script to verify ZenML dashboard access."""
from zenml.client import Client

def main():
    client = Client()

    print("=" * 60)
    print("ZenML Configuration Summary")
    print("=" * 60)

    print(f"\n[Server Connection]")
    print(f"  URL: {client.zen_store.url}")
    print(f"  User: {client.active_user.name}")

    print(f"\n[Active Project]")
    project = client.active_project
    print(f"  Name: {project.name}")
    print(f"  ID: {project.id}")
    print(f"  Description: {project.description or '(none)'}")

    print(f"\n[Dashboard Access]")
    if project.name.lower() == "default":
        print(f"  Status: [OK] Using default project")
        print(f"  Pipelines: Will be visible in dashboard")
        print(f"  Artifacts: Will be accessible in dashboard")
        print(f"  Dashboard URL: http://localhost:8237")
    else:
        print(f"  Status: [WARNING] Using custom project '{project.name}'")
        print(f"  Pipelines: May not be visible in dashboard (Pro feature)")

    print(f"\n[Available Projects]")
    projects = client.list_projects()
    for p in projects.items:
        status = "[ACTIVE]" if p.id == project.id else "[      ]"
        print(f"  {status} {p.name} ({p.id})")

    print("\n" + "=" * 60)
    print("Configuration is ready!")
    print("=" * 60)

if __name__ == "__main__":
    main()
