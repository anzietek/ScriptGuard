"""Simple test pipeline to verify ZenML dashboard visibility."""
import os
import sys
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Set active project BEFORE importing ZenML
os.environ["ZENML_ACTIVE_PROJECT_NAME"] = "default"
os.environ["ZENML_ACTIVE_WORKSPACE_NAME"] = "default"

from zenml import pipeline, step

@step
def step_1() -> str:
    """First test step."""
    print("Step 1: Hello from ZenML!")
    return "Step 1 completed"

@step
def step_2(input_data: str) -> str:
    """Second test step."""
    print(f"Step 2: Received '{input_data}'")
    return f"Step 2 processed: {input_data}"

@step
def step_3(input_data: str) -> str:
    """Third test step."""
    print(f"Step 3: Final step with '{input_data}'")
    return "Pipeline completed successfully!"

@pipeline
def simple_test_pipeline():
    """
    Simple test pipeline with 3 steps.
    Should appear in ZenML dashboard if workspace is configured correctly.
    """
    result_1 = step_1()
    result_2 = step_2(result_1)
    result_3 = step_3(result_2)
    return result_3

if __name__ == "__main__":
    print("="*70)
    print("RUNNING SIMPLE TEST PIPELINE")
    print("="*70)
    print(f"Active project: {os.getenv('ZENML_ACTIVE_PROJECT_NAME', 'NOT SET')}")
    print(f"Server URL: {os.getenv('ZENML_SERVER_URL', 'NOT SET')}")
    print("="*70)

    try:
        # Run the pipeline
        run = simple_test_pipeline()

        print("\n" + "="*70)
        print("PIPELINE RUN COMPLETED!")
        print("="*70)
        print(f"Run ID: {run.id if hasattr(run, 'id') else 'N/A'}")
        print(f"Status: {run.status if hasattr(run, 'status') else 'N/A'}")
        print("\nCheck dashboard at: http://localhost:8237/pipelines")
        print("Look for pipeline: simple_test_pipeline")
        print("="*70)

    except Exception as e:
        print(f"\nERROR: Pipeline failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
