"""Get direct URL to pipeline in dashboard."""
import os
from dotenv import load_dotenv
load_dotenv()

from zenml.client import Client

client = Client()

pipelines = client.list_pipelines()
for p in pipelines:
    print(f"\nPipeline: {p.name}")
    print(f"Direct URL: http://localhost:8237/pipelines/{p.id}")
    print(f"\nOtworz ten link w przegladarce!")
