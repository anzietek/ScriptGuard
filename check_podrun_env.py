"""
ScriptGuard Podrun Environment Checker
Quick validation script to verify environment readiness.
"""

import sys
import os
from pathlib import Path


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_check(name: str, status: bool, message: str = "") -> None:
    """Print check result."""
    symbol = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {name}: {message if message else ('OK' if status else 'FAILED')}")


def check_python() -> bool:
    """Check Python version."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    is_compatible = version.major == 3 and version.minor in (12, 13)
    print_check("Python Version", is_compatible, version_str)
    return is_compatible


def check_package(package_name: str) -> bool:
    """Check if package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def check_zenml() -> bool:
    """Check ZenML installation and initialization."""
    if not check_package("zenml"):
        print_check("ZenML Package", False, "Not installed")
        return False

    print_check("ZenML Package", True, "Installed")

    # Check if ZenML is initialized
    zen_dir = Path(".zen")
    initialized = zen_dir.exists()
    print_check("ZenML Initialized", initialized)

    return True


def check_pytorch() -> bool:
    """Check PyTorch installation and CUDA support."""
    if not check_package("torch"):
        print_check("PyTorch", False, "Not installed")
        return False

    import torch

    version = torch.__version__
    print_check("PyTorch Version", True, version)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        print_check("CUDA Available", True, f"{device_count} GPU(s) - {device_name}")
    else:
        print_check("CUDA Available", False, "Using CPU")

    return True


def check_qdrant() -> bool:
    """Check Qdrant connection."""
    if not check_package("qdrant_client"):
        print_check("Qdrant Client", False, "Not installed")
        return False

    print_check("Qdrant Client", True, "Installed")

    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    try:
        client = QdrantClient(url=qdrant_url)
        collections = client.get_collections()
        print_check("Qdrant Connection", True, f"{len(collections.collections)} collections")
        return True
    except Exception as e:
        print_check("Qdrant Connection", False, str(e))
        return False


def check_config_files() -> bool:
    """Check required configuration files."""
    files = {
        "config.yaml": Path("config.yaml"),
        ".env": Path(".env"),
        "pyproject.toml": Path("pyproject.toml"),
    }

    all_exist = True
    for name, path in files.items():
        exists = path.exists()
        print_check(f"Config: {name}", exists)
        all_exist = all_exist and exists

    return all_exist


def check_directories() -> bool:
    """Check required directories."""
    dirs = ["data", "models", "logs", "src"]

    all_exist = True
    for dir_name in dirs:
        path = Path(dir_name)
        exists = path.exists() and path.is_dir()
        print_check(f"Directory: {dir_name}", exists)
        all_exist = all_exist and exists

    return all_exist


def check_dependencies() -> bool:
    """Check key dependencies."""
    packages = {
        "transformers": "Transformers",
        "datasets": "Datasets",
        "peft": "PEFT",
        "accelerate": "Accelerate",
        "sentence_transformers": "Sentence Transformers",
        "fastapi": "FastAPI",
        "wandb": "WandB",
        "loguru": "Loguru",
    }

    all_installed = True
    for package, name in packages.items():
        installed = check_package(package)
        print_check(name, installed)
        all_installed = all_installed and installed

    return all_installed


def check_env_variables() -> bool:
    """Check required environment variables."""
    required_vars = {
        "QDRANT_URL": "Qdrant connection URL",
        "HF_TOKEN": "HuggingFace token",
    }

    optional_vars = {
        "WANDB_API_KEY": "WandB API key",
        "ZENML_SERVER_URL": "ZenML server URL",
        "DATABASE_URL": "PostgreSQL connection",
    }

    print("\nRequired environment variables:")
    all_set = True
    for var, description in required_vars.items():
        is_set = os.getenv(var) is not None
        print_check(var, is_set, description)
        all_set = all_set and is_set

    print("\nOptional environment variables:")
    for var, description in optional_vars.items():
        is_set = os.getenv(var) is not None
        status = "Set" if is_set else "Not set"
        print_check(var, True, f"{description} - {status}")

    return all_set


def check_disk_space() -> bool:
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")

        free_gb = free // (2**30)
        min_required = 50  # GB

        sufficient = free_gb >= min_required
        print_check("Disk Space", sufficient, f"{free_gb} GB free (min {min_required} GB)")

        return sufficient
    except Exception as e:
        print_check("Disk Space", False, f"Cannot check: {e}")
        return False


def main() -> int:
    """Main check routine."""
    print_header("ScriptGuard Podrun Environment Check")

    checks = {
        "Python Environment": check_python,
        "Core Dependencies": check_dependencies,
        "PyTorch & CUDA": check_pytorch,
        "ZenML": check_zenml,
        "Qdrant": check_qdrant,
        "Configuration Files": check_config_files,
        "Project Structure": check_directories,
        "Environment Variables": check_env_variables,
        "Disk Space": check_disk_space,
    }

    results = {}

    for category, check_func in checks.items():
        print_header(category)
        try:
            results[category] = check_func()
        except Exception as e:
            print_check(category, False, f"Error: {e}")
            results[category] = False

    # Summary
    print_header("Summary")

    passed = sum(results.values())
    total = len(results)

    for category, result in results.items():
        print_check(category, result)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ Environment is ready for training!")
        return 0
    else:
        print("\n✗ Some checks failed. Please resolve issues before training.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user.")
        sys.exit(1)
