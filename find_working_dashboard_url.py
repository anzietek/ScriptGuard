"""Find the working dashboard URL for pipelines."""
import requests

zenml_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5NmIwZTM5MC04ODVjLTQzMzUtYWI0MC1kMDczZWJiZjNkYjQiLCJpc3MiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMiLCJhdWQiOiJiYTMxNzFjNC0xY2IzLTQxMGItODdjNy00YjI5ZTMzZTEwNWMifQ.H5yiIjwWLdmqbHGql1H2eSEwD2wrxjU-LTGWFdv-98o"

base_url = "http://localhost:8237"
headers = {"Cookie": f"zenml-server-ba3171c4-1cb3-410b-87c7-4b29e33e105c={zenml_token}"}

default_project_id = "76e1b71e-18db-42c9-85da-639839a7fcfa"

print("="*70)
print("TESTING DASHBOARD URLS")
print("="*70)

# Test different URL patterns
urls_to_test = [
    f"/projects/{default_project_id}/pipelines",
    f"/workspaces/{default_project_id}/pipelines",
    f"/projects/default/pipelines",
    f"/workspaces/default/pipelines",
    f"/?project={default_project_id}",
    f"/pipelines?project={default_project_id}",
]

working_urls = []

for url_path in urls_to_test:
    url = f"{base_url}{url_path}"
    try:
        response = requests.get(url, headers=headers, allow_redirects=False, timeout=5)

        if response.status_code == 200:
            # Check if it's HTML (dashboard page)
            content_type = response.headers.get('Content-Type', '')
            if 'html' in content_type:
                print(f"[OK] {url}")
                working_urls.append(url)
            else:
                print(f"[SKIP] {url} (not HTML)")
        elif response.status_code == 404:
            print(f"[404] {url}")
        else:
            print(f"[{response.status_code}] {url}")
    except Exception as e:
        print(f"[ERR] {url}: {e}")

print("\n" + "="*70)
print("WORKING URLS")
print("="*70)

if working_urls:
    print("\nOtworz jeden z tych URLi:")
    for url in working_urls:
        print(f"  {url}")
else:
    print("\nZaden URL z projektem nie zadziala bezposrednio")
    print("\nDashboard wymaga manualnego wyboru projektu w UI")
    print("Sprawdzam gdzie jest selector...")

    # Check the main dashboard HTML for project selector info
    response = requests.get(f"{base_url}/", headers=headers)
    html = response.text

    # Look for project/workspace related elements
    if 'workspace' in html.lower():
        print("  [FOUND] 'workspace' w HTML")
    if 'project' in html.lower():
        print("  [FOUND] 'project' w HTML")

    print("\n" + "="*70)
    print("OSTATECZNE ROZWIAZANIE")
    print("="*70)
    print("\nDashboard ZenML wymaga wyboru projektu przez UI.")
    print("NIE MA bezposredniego URLa z projektem w tej wersji.")
    print("\nMusisz:")
    print("1. Otworzyc: http://localhost:8237/")
    print("2. Znalezc i kliknac selector projektu (szukaj w:")
    print("   - Gorny lewy rog")
    print("   - Menu hamburger (3 kreski)")
    print("   - Settings/Preferences")
    print("   - Avatar menu (prawy gorny rog)")
    print("3. Wybrac 'default' z listy")
    print("\nJeśli NIE WIDZISZ selectora - wyslij screenshot!")
