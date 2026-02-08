import os
import sys

# CRITICAL: Monkey-patch torch to add missing low-bit int dtypes BEFORE any imports
# torchao >= 0.7 requires torch.int1/int2/int3/int4/int5/int6/int7 added in PyTorch 2.6+
# We're on PyTorch 2.5.1 for unsloth compatibility, so we fake these dtypes
import torch
if not hasattr(torch, 'int1'):
    # Create fake dtypes as aliases to int8 (they won't actually be used in our pipeline)
    torch.int1 = torch.int8  # type: ignore
    torch.int2 = torch.int8  # type: ignore
    torch.int3 = torch.int8  # type: ignore
    torch.int4 = torch.int8  # type: ignore
    torch.int5 = torch.int8  # type: ignore
    torch.int6 = torch.int8  # type: ignore
    torch.int7 = torch.int8  # type: ignore

# CRITICAL: Disable torch.compile immediately after torch import
# This must happen before any other imports or code execution to prevent unsloth
# from triggering compilation with incompatible options on PyTorch 2.5.1
try:
    torch._dynamo.config.suppress_errors = True  # type: ignore
    torch._dynamo.config.disable = True  # type: ignore
    
    # Monkeypatch torch.compile to be a no-op
    # unsloth uses @torch.compile with options invalid for PyTorch 2.5.1
    def no_op_compile(*args, **kwargs):
        """No-op replacement for torch.compile that returns functions unchanged."""
        # If called with a function as first arg, return it unchanged
        if args and callable(args[0]):
            return args[0]
        # Otherwise return a decorator that returns functions unchanged
        def decorator(func):
            return func
        return decorator
    torch.compile = no_op_compile
    
    print("DEBUG: torch._dynamo disabled and torch.compile monkey-patched", file=sys.stderr)
except (AttributeError, ImportError):
    pass

# CRITICAL: Import Windows Triton fix FIRST - before any other imports
# This monkey-patches torch.compile to prevent Triton CUDA version errors
if sys.platform == "win32":
    from scriptguard.utils.windows_triton_fix import *  # noqa: F401, F403

import signal
import yaml
import shutil
from scriptguard.utils.logger import logger

# Disable torch.compile FIRST - before ANY imports or .env loading
# CRITICAL: PyTorch 2.5.1 + unsloth_zoo tries to use triton options not available in this version
# This prevents RuntimeError: Unexpected optimization option triton.enable_persistent_tma_matmul
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
if sys.platform == "win32":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

1# CUDA Memory Management - prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CRITICAL: Clear unsloth compiled cache - it may contain code using incompatible triton options
unsloth_cache = "/tmp/unsloth_compiled_cache"
if os.path.exists(unsloth_cache):
    try:
        shutil.rmtree(unsloth_cache)
        logger.info(f"Cleared stale unsloth cache: {unsloth_cache}")
    except Exception as e:
        logger.warning(f"Could not clear unsloth cache: {e}")

logger.info("torch.compile disabled (PyTorch 2.5.1 compatibility)")

# Now load .env files
from dotenv import load_dotenv, find_dotenv

# Load .env - find_dotenv() automatically searches parent directories
load_dotenv(find_dotenv(usecwd=True))

# Also load .env.dev if it exists (for development with more keys)
env_dev = find_dotenv(".env.dev", usecwd=True)
if env_dev:
    load_dotenv(env_dev, override=True)
    logger.info(f"Loaded .env.dev from {env_dev}")

# Disable transformers lazy loading (fixes Python 3.13 Ctrl+C issues)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure torch._dynamo AFTER env is loaded but BEFORE importing torch libraries
if sys.platform == "win32":
    import torch
    try:
        torch._dynamo.config.suppress_errors = True  # type: ignore
        torch._dynamo.config.disable = True  # type: ignore
        logger.info("torch._dynamo disabled successfully")
    except (AttributeError, ImportError):
        logger.warning("Could not disable torch._dynamo - may not be available")

# Fix ZenML Windows path handling (monkey-patch)
if sys.platform == "win32":
    import zenml.io.fileio as zenml_fileio
    _original_open = zenml_fileio.open

    def _patched_open(path, mode="r"):
        # Normalize Windows paths to use consistent separators
        normalized_path = path.replace("\\", "/")
        return _original_open(normalized_path, mode=mode)

    zenml_fileio.open = _patched_open

# Graceful shutdown handler for Ctrl+C
def signal_handler(sig, frame):
    logger.info("\n\nShutting down gracefully...")
    logger.info("Cleaning up resources...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# CRITICAL: Patch torch._inductor.runtime.hints to avoid TypeError on Windows
# unsloth imports DeviceProperties from this module, but it fails on load
# because of a dataclasses issue with AttrsDescriptor.
if sys.platform == "win32":
    import types
    # Create mock module
    mock_hints = types.ModuleType("torch._inductor.runtime.hints")
    class DeviceProperties:
        pass
    mock_hints.DeviceProperties = DeviceProperties
    
    # Inject into sys.modules - FORCE overwrite
    sys.modules["torch._inductor.runtime.hints"] = mock_hints
    
    # Ensure hierarchy exists to satisfy imports that traverse the package
    if "torch._inductor" not in sys.modules:
        sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
    
    if "torch._inductor.runtime" not in sys.modules:
        sys.modules["torch._inductor.runtime"] = types.ModuleType("torch._inductor.runtime")
        # Link to parent
        sys.modules["torch._inductor"].runtime = sys.modules["torch._inductor.runtime"]
        
    # Link hints to runtime
    sys.modules["torch._inductor.runtime"].hints = mock_hints
    
    print("DEBUG: Force-mocked torch._inductor.runtime.hints to bypass Windows compatibility issue", file=sys.stderr)

# Import unsloth FIRST - must precede transformers/peft for optimizations
import unsloth  # noqa: F401

# Import heavy dependencies after signal handler setup
from scriptguard.pipelines.training_pipeline import (
    malware_detection_training_pipeline,
    advanced_training_pipeline
)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file and substitute environment variables.

    Supports syntax: ${ENV_VAR:-default_value} or ${ENV_VAR}
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    def convert_type(value: str):
        """Convert string value to appropriate type (int, float, bool, or str)."""
        if not isinstance(value, str):
            return value

        # Try to convert to boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass

        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def substitute_env_vars(obj):
        """Recursively substitute environment variables in config."""
        if isinstance(obj, dict):
            return {k: substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            # Parse ${ENV_VAR:-default}
            env_expr = obj[2:-1]
            if ":-" in env_expr:
                env_var, default = env_expr.split(":-", 1)
                result = os.getenv(env_var, default)
            else:
                env_var = env_expr
                result = os.getenv(env_var, "")
            return convert_type(result)
        else:
            return obj

    config = substitute_env_vars(config)
    return config


def main_legacy():
    """Legacy training pipeline (original implementation)."""
    gh_malicious = [
        "https://github.com/example/malware/blob/main/shell.py"
    ]
    gh_benign = [
        "https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py"
    ]

    local_malicious = "./data/malicious_scripts"
    local_benign = "./data/benign_scripts"

    web_urls = ["https://raw.githubusercontent.com/python/cpython/main/Lib/os.py"]
    model_id = "bigcode/starcoder2-3b"

    logger.info("Starting ScriptGuard training pipeline with explicit label separation...")

    run = malware_detection_training_pipeline(
        gh_malicious_urls=gh_malicious,
        gh_benign_urls=gh_benign,
        local_malicious_dir=local_malicious,
        local_benign_dir=local_benign,
        web_urls=web_urls,
        model_id=model_id
    )

    logger.info("Pipeline run completed.")


def initialize_qdrant(config: dict) -> bool:
    """
    Initialize Qdrant vector store with initial data if needed.

    Returns:
        True if successful or already initialized, False on error
    """
    try:
        from scriptguard.rag.qdrant_store import QdrantStore, bootstrap_cve_data

        qdrant_config = config.get("qdrant", {})

        # Try to connect to Qdrant
        store = QdrantStore(
            host=qdrant_config.get("host", "localhost"),
            port=qdrant_config.get("port", 6333),
            collection_name=qdrant_config.get("collection_name", "malware_knowledge"),
            embedding_model=qdrant_config.get("embedding_model", "all-MiniLM-L6-v2"),
            api_key=qdrant_config.get("api_key"),  # Pass directly, QdrantStore will handle env vars
            use_https=qdrant_config.get("use_https", False)
        )

        # Check if collection has data
        info = store.get_collection_info()
        points_count = info.get('points_count', 0)

        if points_count == 0:
            logger.info("Qdrant collection is empty. Bootstrapping with initial CVE data...")
            bootstrap_cve_data(store)
            logger.info(f"✅ Qdrant initialized with CVE patterns")
        else:
            logger.info(f"✅ Qdrant already initialized ({points_count} vectors)")

        return True

    except Exception as e:
        logger.warning(f"⚠️  Qdrant initialization failed: {e}")
        logger.warning("Training will continue without RAG support")
        logger.warning("To use RAG, ensure Qdrant is running: docker-compose up -d")
        return False


def main():
    """Advanced training pipeline with config.yaml."""
    # Load configuration from environment variable or default
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    zenml_config_path = os.getenv("SCRIPTGUARD_ZENML_CONFIG", "zenml_config.yaml")

    try:
        config = load_config(config_path)
        logger.info(f"Configuration loaded successfully from: {config_path}")
    except FileNotFoundError:
        logger.error(f"{config_path} not found. Please create it or set CONFIG_PATH environment variable")
        return
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Initialize Qdrant (optional, non-blocking)
    initialize_qdrant(config)

    # Get model ID from config
    model_id = config.get("training", {}).get("model_id", "bigcode/starcoder2-3b")

    # Add timestamp to force cache invalidation (optional)
    from datetime import datetime
    config["_run_timestamp"] = datetime.now().isoformat()

    logger.info("Starting ScriptGuard Advanced Training Pipeline...")
    logger.info(f"Model: {model_id}")

    # Run advanced pipeline with ZenML config for caching control
    try:
        run = advanced_training_pipeline.with_options(
            config_path=zenml_config_path  # Load cache settings from env or default
        )(
            config=config,
            model_id=model_id
        )
        logger.info("Pipeline run completed successfully!")
        logger.info(f"Results: {run}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
