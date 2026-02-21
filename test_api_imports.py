"""Test API endpoints without full server."""
import sys
import os

os.environ["CONFIG_PATH"] = "config.yaml"

print("=" * 60)
print("SCRIPTGUARD API TEST")
print("=" * 60)

try:
    print("\n[1/5] Importing FastAPI...")
    from fastapi import FastAPI
    print("✅ FastAPI OK")

    print("\n[2/5] Importing schemas...")
    from scriptguard.api.schemas import ScriptAnalysisRequest, ScriptAnalysisResponse
    print("✅ Schemas OK")

    print("\n[3/5] Importing state...")
    from scriptguard.api.state import app_state
    print("✅ State OK")

    print("\n[4/5] Importing prompts...")
    from scriptguard.utils.prompts import format_inference_prompt
    print("✅ Prompts OK")

    print("\n[5/5] Importing main app...")
    from scriptguard.api.main import app
    print("✅ App OK")

    print("\n" + "=" * 60)
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("=" * 60)

    print("\n📋 Available endpoints:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = ', '.join(route.methods)
            print(f"  {methods:10} {route.path}")

    print("\n💡 To start server run:")
    print("   .venv\\Scripts\\python.exe -m uvicorn scriptguard.api.main:app --host 127.0.0.1 --port 8000")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

