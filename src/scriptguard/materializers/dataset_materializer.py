"""Custom materializer for HuggingFace Dataset objects.

This materializer properly handles HuggingFace Dataset serialization/deserialization
and fixes path issues on Windows.
"""
import logging
from pathlib import Path
from typing import Type, Any

from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
from datasets import Dataset, load_from_disk

logger = logging.getLogger(__name__)


class HuggingFaceDatasetMaterializer(BaseMaterializer):
    """Materializer for HuggingFace Dataset objects.

    Uses HuggingFace's native save_to_disk/load_from_disk methods
    which properly handle cross-platform paths.
    """

    ASSOCIATED_TYPES = (Dataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> Dataset:
        """Load a HuggingFace Dataset from disk.

        Args:
            data_type: The type of the data to load.

        Returns:
            The loaded Dataset object.
        """
        # Use Path to properly handle cross-platform paths
        dataset_path = Path(self.uri)
        logger.info(f"Loading dataset from: {dataset_path}")

        # Try direct loading first
        try:
            dataset = load_from_disk(str(dataset_path))
            logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
            return dataset
        except Exception as e:
            logger.warning(f"Direct loading failed: {e}")

            # If direct loading fails, search for dataset in subdirectories
            # Sometimes ZenML adds extra subdirectories
            if dataset_path.is_dir():
                # Look for dataset_dict.json or dataset_info.json in immediate subdirectories first
                for subdir in sorted(dataset_path.iterdir()):
                    if subdir.is_dir():
                        if (subdir / "dataset_dict.json").exists() or (subdir / "dataset_info.json").exists():
                            logger.info(f"Found dataset in subdirectory: {subdir}")
                            try:
                                dataset = load_from_disk(str(subdir))
                                logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
                                return dataset
                            except Exception as subdir_error:
                                logger.warning(f"Failed to load from {subdir}: {subdir_error}")
                                continue

                # If not found in immediate subdirectories, try recursive search
                logger.info("Searching recursively for dataset files...")
                for subdir in dataset_path.rglob("dataset_info.json"):
                    dataset_dir = subdir.parent
                    logger.info(f"Found dataset in: {dataset_dir}")
                    try:
                        dataset = load_from_disk(str(dataset_dir))
                        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
                        return dataset
                    except Exception as recursive_error:
                        logger.warning(f"Failed to load from {dataset_dir}: {recursive_error}")
                        continue

            # If still fails, raise the original error
            raise RuntimeError(
                f"Failed to load HuggingFace Dataset from {dataset_path}. "
                f"Original error: {e}"
            ) from e

    def save(self, data: Dataset) -> None:
        """Save a HuggingFace Dataset to disk.

        Args:
            data: The Dataset to save.
        """
        # Use Path to properly handle cross-platform paths
        dataset_path = Path(self.uri)
        logger.info(f"Saving dataset to: {dataset_path}")

        # Create the directory if it doesn't exist
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Save using HuggingFace's native saver
        try:
            data.save_to_disk(str(dataset_path))
            logger.info(f"Successfully saved dataset with {len(data)} samples")
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise RuntimeError(
                f"Failed to save HuggingFace Dataset to {dataset_path}. "
                f"Error: {e}"
            ) from e
