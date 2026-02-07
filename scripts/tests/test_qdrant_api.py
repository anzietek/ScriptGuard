"""Test Qdrant API to find correct parameter names"""
from qdrant_client import QdrantClient
import inspect

client = QdrantClient(":memory:")

# Check search method signature
if hasattr(client, 'search'):
    sig = inspect.signature(client.search)
    print("client.search parameters:")
    for name, param in sig.parameters.items():
        print(f"  - {name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
elif hasattr(client, 'query_points'):
    sig = inspect.signature(client.query_points)
    print("client.query_points parameters:")
    for name, param in sig.parameters.items():
        print(f"  - {name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
else:
    print("No search method found!")
    print("Available methods:")
    methods = [m for m in dir(client) if not m.startswith('_') and callable(getattr(client, m))]
    for m in sorted(methods)[:20]:
        print(f"  - {m}")
