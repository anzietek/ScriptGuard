"""
Quick script to run ONLY the evaluation step without re-running the entire pipeline.
Uses artifacts from the previous failed run.
"""
import yaml
import os
from pathlib import Path
from scriptguard.steps.model_evaluation import evaluate_model
from scriptguard.utils.logger import logger
from datasets import Dataset

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file and substitute environment variables.

    Supports syntax: ${ENV_VAR:-default_value} or ${ENV_VAR}
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    def substitute_env_vars(obj):
        """Recursively substitute environment variables in config."""
        if isinstance(obj, dict):
            return {k: substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_expr = obj[2:-1]
            if ":-" in env_expr:
                env_var, default = env_expr.split(":-", 1)
                return os.getenv(env_var, default)
            else:
                env_var = env_expr
                return os.getenv(env_var, "")
        else:
            return obj

    return substitute_env_vars(config)

def find_latest_checkpoint():
    """Find the latest checkpoint from training."""
    checkpoints_dir = Path("../../models/scriptguard-model")

    # Try final adapter first
    final_adapter = checkpoints_dir / "final_adapter"
    if final_adapter.exists():
        logger.info(f"Found final adapter: {final_adapter}")
        return str(final_adapter)

    # Otherwise, find latest checkpoint
    checkpoints = list(checkpoints_dir.glob("checkpoint-*"))
    if checkpoints:
        # Sort by checkpoint number
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        logger.info(f"Found latest checkpoint: {latest}")
        return str(latest)

    raise FileNotFoundError("No checkpoint found in models/scriptguard-model/")

def load_test_dataset():
    """Load test dataset from ZenML artifacts or data directory."""
    import json

    # PRIORITY 1: Try to load from ZenML artifacts (most reliable)
    try:
        from zenml.client import Client
        client = Client()

        # Get latest successful pipeline run
        runs = client.list_pipeline_runs(sort_by="desc:created", size=10)

        for run in runs:
            logger.info(f"Checking run: {run.name} (status: {run.status})")

            # Try to find train_model step which outputs test_set
            try:
                # Look for train_model step (it outputs model_path, metrics, test_set)
                if "train_model" not in run.steps:
                    continue

                step_run = run.steps["train_model"]
                logger.info(f"Found train_model step in run {run.name}")
                outputs = step_run.outputs

                logger.debug(f"  Output keys: {list(outputs.keys())}")
                logger.debug(f"  Output types: {[type(v).__name__ for v in outputs.values()]}")

                # test_set is typically the 3rd output
                if "test_set" in outputs:
                    artifact = outputs["test_set"]
                    logger.debug(f"  Loading artifact: {artifact}")
                    test_data = artifact.load()
                    logger.info(f"✅ Loaded test_set from ZenML (run: {run.name}): {len(test_data)} samples")
                    return test_data

                # Fallback: try by index
                output_list = list(outputs.values())
                if len(output_list) >= 3:
                    artifact = output_list[2]
                    logger.debug(f"  Loading artifact by index [2]: {artifact}")
                    test_data = artifact.load()  # 3rd output
                    logger.info(f"✅ Loaded test data from ZenML by index (run: {run.name}): {len(test_data)} samples")
                    return test_data
            except Exception as e:
                logger.warning(f"Could not load test_set from train_model in run {run.name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())

            # Also try split_raw_data step as fallback
            try:
                if "split_raw_data" not in run.steps:
                    continue

                step_run = run.steps["split_raw_data"]
                logger.info(f"Found split_raw_data step in run {run.name}")
                outputs = step_run.outputs

                logger.debug(f"  Type of outputs: {type(outputs)}")
                logger.debug(f"  Output keys: {list(outputs.keys())}")
                logger.debug(f"  Output value types: {[(k, type(v).__name__) for k, v in outputs.items()]}")

                # raw_test_data is the 3rd output - try direct access first
                if "raw_test_data" in outputs:
                    test_data_value = outputs["raw_test_data"]
                    logger.debug(f"  raw_test_data type: {type(test_data_value)}")

                    # If it's an artifact response, load it
                    if hasattr(test_data_value, 'load'):
                        test_data = test_data_value.load()
                    elif hasattr(test_data_value, 'read'):
                        test_data = test_data_value.read()
                    else:
                        # Maybe it's already loaded?
                        test_data = test_data_value

                    logger.info(f"✅ Loaded raw_test_data from ZenML (run: {run.name}): {len(test_data)} samples")
                    return test_data

                # Try all outputs
                for key, value in outputs.items():
                    logger.debug(f"  Trying output '{key}' (type: {type(value).__name__}, len: {len(value) if hasattr(value, '__len__') else 'N/A'})")
                    try:
                        # Value might be a list of ArtifactVersionResponse objects
                        if isinstance(value, list) and len(value) > 0:
                            first_item = value[0]
                            logger.debug(f"    First item type: {type(first_item).__name__}")

                            # If it's an ArtifactVersionResponse, load it
                            if hasattr(first_item, 'load'):
                                logger.debug(f"    Loading artifact...")
                                data = first_item.load()
                                logger.debug(f"    Loaded data type: {type(data).__name__}, len: {len(data) if hasattr(data, '__len__') else 'N/A'}")

                                # Check if it's test data
                                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                                    logger.info(f"✅ Found raw test data in output '{key}' (run: {run.name}): {len(data)} samples")
                                    return Dataset.from_list(data)
                                elif isinstance(data, Dataset):
                                    logger.info(f"✅ Found test Dataset in output '{key}' (run: {run.name}): {len(data)} samples")
                                    return data
                                else:
                                    logger.debug(f"    Not test data: {type(data)}")
                            else:
                                logger.debug(f"    First item has no load() method")
                        else:
                            logger.debug(f"    Not a non-empty list")
                    except Exception as ex:
                        logger.debug(f"    Failed to process output '{key}': {ex}")

            except Exception as e:
                logger.warning(f"Could not load from split_raw_data step in run {run.name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        logger.warning("No test data found in ZenML artifacts")
    except Exception as e:
        logger.warning(f"Could not access ZenML artifacts: {e}")

    # PRIORITY 2: Try to find test data JSON in data/
    data_dir = Path("../../data")
    test_files = list(data_dir.glob("*test*.json"))
    if test_files:
        logger.info(f"Found test file: {test_files[0]}")
        with open(test_files[0], 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        logger.info(f"✅ Loaded test data from {test_files[0]}: {len(test_data)} samples")
        return Dataset.from_list(test_data)

    # PRIORITY 3: Try raw_samples.json and split it ourselves
    raw_samples_file = data_dir / "raw_samples.json"
    if raw_samples_file.exists():
        logger.info(f"Found raw_samples.json, splitting manually...")
        with open(raw_samples_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        # Take last 10% as test set (same as pipeline)
        test_size = int(len(all_data) * 0.1)
        if test_size < 10:
            test_size = min(10, len(all_data))  # At least 10 samples if possible

        test_data = all_data[-test_size:]

        logger.info(f"✅ Loaded and split test data from raw_samples.json: {len(test_data)} samples")
        return Dataset.from_list(test_data)

    # PRIORITY 4: Check for enhanced_samples.json
    enhanced_file = data_dir / "enhanced_samples.json"
    if enhanced_file.exists():
        logger.info(f"Found enhanced_samples.json, splitting manually...")
        with open(enhanced_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        test_size = int(len(all_data) * 0.1)
        if test_size < 10:
            test_size = min(10, len(all_data))

        test_data = all_data[-test_size:]

        logger.info(f"✅ Loaded and split test data from enhanced_samples.json: {len(test_data)} samples")
        return Dataset.from_list(test_data)

    # LAST RESORT: Error - no real test data available
    logger.error("=" * 60)
    logger.error("CRITICAL ERROR: NO TEST DATA FOUND!")
    logger.error("=" * 60)
    logger.error("Cannot run evaluation without test data.")
    logger.error("Please run the full pipeline first to generate test data:")
    logger.error("  python main.py")
    logger.error("")
    logger.error("Or ensure data/ directory contains one of:")
    logger.error("  - *test*.json")
    logger.error("  - raw_samples.json")
    logger.error("  - enhanced_samples.json")
    logger.error("=" * 60)

    raise FileNotFoundError("No test dataset found. Run full pipeline first.")

def main():
    logger.info("=" * 60)
    logger.info("Running EVALUATION ONLY (no training)")
    logger.info("=" * 60)

    # Load config
    config = load_config()

    # Find checkpoint
    try:
        adapter_path = find_latest_checkpoint()
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please ensure you have trained a model first.")
        return

    # Load test dataset
    test_dataset = load_test_dataset()

    # Get model ID from config
    model_id = config.get("training", {}).get("model_id", "bigcode/starcoder2-3b")

    # Run evaluation step
    logger.info(f"Evaluating model: {model_id}")
    logger.info(f"Adapter path: {adapter_path}")
    logger.info(f"Test samples: {len(test_dataset)}")

    try:
        metrics = evaluate_model(
            adapter_path=adapter_path,
            test_dataset=test_dataset,
            base_model_id=model_id,
            config=config,
            use_fewshot_rag=True  # Enable Few-Shot RAG
        )

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Metrics: {metrics}")

        # Save metrics to file
        import json
        metrics_file = Path("evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
